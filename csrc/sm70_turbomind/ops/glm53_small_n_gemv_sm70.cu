// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GLM-5.3 decode: small-N FP16 projections (router gate, shared-expert MLP)
// for M <= 8 rows, y[M][N] = x[M][K] @ W[N][K]^T over row-major FP16
// weights, FP32 accumulation on the Volta tensor core (mma.m8n8k4), FP16 or
// FP32 output. cuBLAS split-Ks these shapes into 100-200 GB/s and the router
// then needs a cast kernel; this writes the FP32 logits directly. Each warp
// streams a 32-column strip of W along k with 16 B loads per lane and an 8-deep
// prefetch; the eight A rows of the MMA carry the eight activation rows, so M
// = 1..8 costs the same instruction stream. Split-K across blocks writes FP32
// partials that a fixed-order reduce turns into FP16, so the result is
// deterministic.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#include <type_traits>

namespace {

constexpr int kWarps = 4;
constexpr int kPrefetch = 8;
constexpr int kMaxM = 8;

__device__ __forceinline__ void mma884(float (&d)[8], const uint32_t (&a)[2],
                                       const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
}

// grid (N / 32, ksplit), block kWarps warps. Block (strip, s) covers columns
// strip*32.. and k in [s*kper, (s+1)*kper); warp w takes a quarter of that.
// out: FP32 partials [ksplit][M][N] (ksplit > 1) or FP16 y [M][N] (ksplit ==
// 1).
template <bool kDirect, bool kOutF32>
__global__ void __launch_bounds__(kWarps * 32)
    fp16_mma_gemv_kernel(const half* __restrict__ w, const half* __restrict__ x,
                         void* __restrict__ out, int M, int N, int K,
                         int kper) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int n0 = blockIdx.x * 32;
  const int q = (lane >> 2) & 3;                        // 8-column group
  const int nw = 8 * q + (lane >> 4) * 4 + (lane & 3);  // B column of this lane
  const int row = (lane >> 4) * 4 + (lane & 3);         // A row of this lane
  const bool on = row < M;
  const int kw = kper / kWarps;  // k per warp
  const int kbeg = blockIdx.y * kper + warp * kw;
  const int steps = kw >> 4;

  const half* wr = w + static_cast<int64_t>(n0 + nw) * K + kbeg;
  const half* xr = x + static_cast<int64_t>(row) * K + kbeg;

  float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int s0 = 0; s0 < steps; s0 += kPrefetch) {
    uint4 wv[kPrefetch][2];
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      if (s0 + p < steps) {
        const uint4* src = reinterpret_cast<const uint4*>(wr + (s0 + p) * 16);
        wv[p][0] = __ldg(src);
        wv[p][1] = __ldg(src + 1);
      } else {
        wv[p][0] = make_uint4(0, 0, 0, 0);
        wv[p][1] = make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      if (s0 + p >= steps) break;
      uint4 xa = make_uint4(0, 0, 0, 0), xb = make_uint4(0, 0, 0, 0);
      if (on) {
        const uint4* xs = reinterpret_cast<const uint4*>(xr + (s0 + p) * 16);
        xa = __ldg(xs);
        xb = __ldg(xs + 1);
      }
      const uint32_t a0[2] = {xa.x, xa.y}, a1[2] = {xa.z, xa.w};
      const uint32_t a2[2] = {xb.x, xb.y}, a3[2] = {xb.z, xb.w};
      const uint32_t b0[2] = {wv[p][0].x, wv[p][0].y};
      const uint32_t b1[2] = {wv[p][0].z, wv[p][0].w};
      const uint32_t b2[2] = {wv[p][1].x, wv[p][1].y};
      const uint32_t b3[2] = {wv[p][1].z, wv[p][1].w};
      mma884(d, a0, b0);
      mma884(d, a1, b1);
      mma884(d, a2, b2);
      mma884(d, a3, b3);
    }
  }

  // D fragment: lane L, reg i -> row (L&1) + ((i>>1)&1)*2 + (L>=16 ? 4 : 0),
  // col 8q + ((i>>2)&1)*4 + (L&2) + (i&1).
  __shared__ float red[kWarps][kMaxM][32];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = (lane & 1) + ((i >> 1) & 1) * 2 + (lane >= 16 ? 4 : 0);
    const int c = 8 * q + ((i >> 2) & 1) * 4 + (lane & 2) + (i & 1);
    red[warp][r][c] = d[i];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < M * 32; idx += kWarps * 32) {
    const int m = idx >> 5, c = idx & 31;
    float acc = 0.f;
#pragma unroll
    for (int wi = 0; wi < kWarps; ++wi) acc += red[wi][m][c];
    const int64_t o = static_cast<int64_t>(m) * N + n0 + c;
    if constexpr (kDirect && kOutF32) {
      reinterpret_cast<float*>(out)[o] = acc;
    } else if constexpr (kDirect) {
      reinterpret_cast<half*>(out)[o] = __float2half_rn(acc);
    } else {
      reinterpret_cast<float*>(
          out)[static_cast<int64_t>(blockIdx.y) * M * N + o] = acc;
    }
  }
}

template <typename OutT>
__global__ void reduce_partials_kernel(const float* __restrict__ part,
                                       OutT* __restrict__ y, int ksplit,
                                       int64_t mn) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= mn) return;
  float acc = 0.f;
  for (int s = 0; s < ksplit; ++s)
    acc += part[static_cast<int64_t>(s) * mn + i];
  if constexpr (std::is_same<OutT, float>::value)
    y[i] = acc;
  else
    y[i] = __float2half_rn(acc);
}

int choose_ksplit(int N, int K) {
  const int strips = N / 32;
  int ks = 1;
  // at least ~4 blocks per SM (320) while keeping >= 128 k (8 steps) per warp
  while (strips * ks < 320 && (K / (ks * 2)) >= kWarps * 128 && ks < 16)
    ks *= 2;
  return ks;
}

}  // namespace

void sm70_glm53_small_n_gemv_out(torch::Tensor y, torch::Tensor x,
                                 torch::Tensor w) {
  TORCH_CHECK(y.is_cuda() && x.is_cuda() && w.is_cuda(),
              "sm70_glm53_small_n_gemv_out: CUDA tensors");
  TORCH_CHECK(
      (y.scalar_type() == torch::kFloat16 ||
       y.scalar_type() == torch::kFloat32) &&
          x.scalar_type() == torch::kFloat16 &&
          w.scalar_type() == torch::kFloat16,
      "sm70_glm53_small_n_gemv_out: float16 inputs, float16 or float32 output");
  const bool out_f32 = y.scalar_type() == torch::kFloat32;
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && y.dim() == 2,
              "sm70_glm53_small_n_gemv_out: rank 2");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && y.is_contiguous(),
              "sm70_glm53_small_n_gemv_out: contiguous tensors");
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= kMaxM, "sm70_glm53_small_n_gemv_out: 1 to 8 rows");
  TORCH_CHECK(w.size(1) == K && y.size(0) == M && y.size(1) == N,
              "sm70_glm53_small_n_gemv_out: shapes");
  TORCH_CHECK(N % 32 == 0 && K % (kWarps * 16) == 0,
              "sm70_glm53_small_n_gemv_out: N % 32, K % 64");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int ks = choose_ksplit(N, K);
  const int kper = K / ks;
  TORCH_CHECK(kper % (kWarps * 16) == 0, "sm70_glm53_small_n_gemv_out: split");
  const dim3 grid(N / 32, ks);
  const half* wp = reinterpret_cast<const half*>(w.data_ptr<at::Half>());
  const half* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  if (ks == 1) {
    if (out_f32)
      fp16_mma_gemv_kernel<true, true><<<grid, kWarps * 32, 0, stream>>>(
          wp, xp, y.data_ptr(), M, N, K, kper);
    else
      fp16_mma_gemv_kernel<true, false><<<grid, kWarps * 32, 0, stream>>>(
          wp, xp, y.data_ptr(), M, N, K, kper);
  } else {
    auto part = torch::empty({ks, M, N}, x.options().dtype(torch::kFloat32));
    fp16_mma_gemv_kernel<false, false><<<grid, kWarps * 32, 0, stream>>>(
        wp, xp, part.data_ptr(), M, N, K, kper);
    const int64_t mn = static_cast<int64_t>(M) * N;
    const unsigned rb = static_cast<unsigned>((mn + 255) / 256);
    if (out_f32)
      reduce_partials_kernel<float><<<rb, 256, 0, stream>>>(
          part.data_ptr<float>(), y.data_ptr<float>(), ks, mn);
    else
      reduce_partials_kernel<half><<<rb, 256, 0, stream>>>(
          part.data_ptr<float>(),
          reinterpret_cast<half*>(y.data_ptr<at::Half>()), ks, mn);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
