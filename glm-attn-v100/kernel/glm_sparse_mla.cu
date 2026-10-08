// SPDX-License-Identifier: Apache-2.0
// GLM-5.3-Flash sparse MLA on SM70 (V100): packed-E4M3 latent KV, per-query top-k gather,
// absorbed MLA = MQA with head dim 512 (every query head attends over the same 512-wide latent,
// which is both K and V). Tensor-core QK^T and PV with FP32 accumulation; FP32 online softmax
// with exact row maxima; split-K over the selected keys with an FP32 log-sum-exp combine.
//
// Cache layout (matches vllm/models/glm5next/sm70/fp8_kv.py): per block of `block_size` tokens,
//   [block_size x 512 E4M3 bytes][block_size x 8 scale bytes], scale = 2^(byte - 127) per 64 values.
// Dequantization is bit-exact with the Triton gather kernel: fp16(e4m3) * fp16(2^(e-127)) in fp16.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>

namespace glm_attn_v100 {
using namespace nvcuda;

constexpr int kD = 512;          // latent width (K = V)
constexpr int kHG = 16;          // query heads per CTA = one WMMA M tile
constexpr int kBN = 32;          // keys per tile
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kGroup = 64;
constexpr int kGroups = kD / kGroup;
constexpr int kSlotBytes = kD + kGroups;

constexpr int kLdQ = kD + 8;     // half
constexpr int kLdK = kD + 8;     // half
constexpr int kLdO = kD + 4;     // float
constexpr int kLdS = kBN + 4;    // float
constexpr int kLdP = kBN + 8;    // half

constexpr int kOffQ = 0;
constexpr int kOffK = kOffQ + kHG * kLdQ * 2;
constexpr int kOffO = kOffK + kBN * kLdK * 2;
constexpr int kOffS = kOffO + kHG * kLdO * 4;
constexpr int kOffP = kOffS + kHG * kLdS * 4;
constexpr int kOffLut = kOffP + kHG * kLdP * 2;
constexpr int kOffM = kOffLut + 256 * 2;
constexpr int kOffL = kOffM + kHG * 4;
constexpr int kOffAlpha = kOffL + kHG * 4;
constexpr int kOffSlot = kOffAlpha + kHG * 4;
constexpr int kSmemBytes = kOffSlot + kBN * 4;
static_assert(kOffK % 32 == 0 && kOffO % 32 == 0 && kOffS % 32 == 0 && kOffP % 32 == 0,
              "WMMA tiles need 256-bit aligned shared memory");
static_assert(kSmemBytes <= 96 * 1024, "SM70 opt-in shared memory limit");

__device__ __forceinline__ float e4m3fn_to_float(uint32_t bits) {
  const uint32_t exponent = (bits >> 3) & 0xF;
  const uint32_t mantissa = bits & 0x7;
  float value = exponent == 0
                    ? static_cast<float>(mantissa) * 0.001953125f  // 2^-9
                    : __uint_as_float(((exponent + 120u) << 23) | (mantissa << 20));
  if (bits & 0x80) value = -value;
  if (exponent == 15 && mantissa == 7) value = __int_as_float(0x7fc00000);
  return value;
}

__device__ __forceinline__ half scale_from_byte(uint32_t e) {
  // 2^(e - 127) exactly; e = 0 underflows to 0 in fp16 like the Triton reference.
  const float s = e == 0 ? 0.0f : (e == 255 ? INFINITY : __uint_as_float(e << 23));
  return __float2half_rn(s);
}

__global__ void __launch_bounds__(kThreads, 1) glm_sparse_mla_fp8_split_kernel(
    const half* __restrict__ q, int64_t q_stride_t, int64_t q_stride_h,
    const uint8_t* __restrict__ cache, int64_t cache_stride_block, int block_size,
    int64_t num_slots, const int32_t* __restrict__ indices, int64_t idx_stride_t,
    int index_width, const int32_t* __restrict__ lengths, float scale, int keys_per_split,
    float* __restrict__ o_part, float* __restrict__ ml, int num_heads, int num_splits) {
  extern __shared__ __align__(128) unsigned char smem[];
  half* sQ = reinterpret_cast<half*>(smem + kOffQ);
  half* sK = reinterpret_cast<half*>(smem + kOffK);
  float* sO = reinterpret_cast<float*>(smem + kOffO);
  float* sS = reinterpret_cast<float*>(smem + kOffS);
  half* sP = reinterpret_cast<half*>(smem + kOffP);
  half* sLut = reinterpret_cast<half*>(smem + kOffLut);
  float* sM = reinterpret_cast<float*>(smem + kOffM);
  float* sL = reinterpret_cast<float*>(smem + kOffL);
  float* sAlpha = reinterpret_cast<float*>(smem + kOffAlpha);
  int* sSlot = reinterpret_cast<int*>(smem + kOffSlot);

  const int split = blockIdx.x;
  const int h0 = blockIdx.y * kHG;
  const int t = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;

  for (int i = tid; i < 256; i += kThreads) sLut[i] = __float2half_rn(e4m3fn_to_float(i));
  const half* q_base = q + t * q_stride_t + h0 * q_stride_h;
  for (int i = tid; i < kHG * (kD / 8); i += kThreads) {
    const int r = i / (kD / 8), c8 = i % (kD / 8);
    *reinterpret_cast<uint4*>(sQ + r * kLdQ + c8 * 8) =
        *reinterpret_cast<const uint4*>(q_base + r * q_stride_h + c8 * 8);
  }
  for (int i = tid; i < kHG * kLdO; i += kThreads) sO[i] = 0.0f;
  if (tid < kHG) {
    sM[tid] = -INFINITY;
    sL[tid] = 0.0f;
  }
  int len = lengths[t];
  len = len < 0 ? 0 : (len > index_width ? index_width : len);
  const int k_begin = split * keys_per_split;
  const int k_end = min(k_begin + keys_per_split, len);
  const int32_t* idx_row = indices + t * idx_stride_t;
  __syncthreads();

  for (int kt = k_begin; kt < k_end; kt += kBN) {
    if (tid < kBN) {
      const int idx = kt + tid;
      int slot = -1;
      if (idx < k_end) {
        const int s = idx_row[idx];
        if (s >= 0 && static_cast<int64_t>(s) < num_slots) slot = s;
      }
      sSlot[tid] = slot;
    }
    __syncthreads();

    // Gather + exact dequant of 32 latent rows into shared memory (invalid rows -> 0).
    for (int i = tid; i < kBN * (kD / 4); i += kThreads) {
      const int j = i / (kD / 4), w = i % (kD / 4);
      const int slot = sSlot[j];
      half2 lo = __float2half2_rn(0.0f), hi = lo;
      if (slot >= 0) {
        const int64_t blk = slot / block_size;
        const int64_t pos = slot - blk * block_size;
        const uint8_t* base = cache + blk * cache_stride_block;
        const uint32_t word = *reinterpret_cast<const uint32_t*>(base + pos * kD + w * 4);
        const half hs = scale_from_byte(base[static_cast<int64_t>(block_size) * kD + pos * kGroups +
                                             (w * 4) / kGroup]);
        lo = __halves2half2(__hmul(sLut[word & 0xff], hs), __hmul(sLut[(word >> 8) & 0xff], hs));
        hi = __halves2half2(__hmul(sLut[(word >> 16) & 0xff], hs), __hmul(sLut[word >> 24], hs));
      }
      half2* dst = reinterpret_cast<half2*>(sK + j * kLdK + w * 4);
      dst[0] = lo;
      dst[1] = hi;
    }
    __syncthreads();

    // S = Q K^T (16 heads x 32 keys), FP32 accumulation, unscaled.
    if (warp < kBN / 16) {
      const int n0 = warp * 16;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.0f);
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
      for (int k0 = 0; k0 < kD; k0 += 16) {
        wmma::load_matrix_sync(a, sQ + k0, kLdQ);
        wmma::load_matrix_sync(b, sK + n0 * kLdK + k0, kLdK);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(sS + n0, acc, kLdS, wmma::mem_row_major);
    }
    __syncthreads();

    // FP32 online softmax: 16 lanes per head row, 2 keys per lane.
    {
      const int r = tid >> 4;
      const int c = (tid & 15) * 2;
      const bool v0 = sSlot[c] >= 0, v1 = sSlot[c + 1] >= 0;
      const float s0 = v0 ? sS[r * kLdS + c] * scale : -INFINITY;
      const float s1 = v1 ? sS[r * kLdS + c + 1] * scale : -INFINITY;
      float mx = fmaxf(s0, s1);
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
      const float m_old = sM[r];
      const float m_new = fmaxf(m_old, mx);
      float p0 = 0.0f, p1 = 0.0f;
      if (m_new != -INFINITY) {
        p0 = v0 ? expf(s0 - m_new) : 0.0f;
        p1 = v1 ? expf(s1 - m_new) : 0.0f;
      }
      const half hp0 = __float2half_rn(p0), hp1 = __float2half_rn(p1);
      sP[r * kLdP + c] = hp0;
      sP[r * kLdP + c + 1] = hp1;
      // The denominator uses the same rounded probabilities the PV GEMM consumes.
      float sum = __half2float(hp0) + __half2float(hp1);
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
      __syncwarp();
      if ((tid & 15) == 0) {
        float alpha = 1.0f;
        if (m_new != -INFINITY) alpha = m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
        sL[r] = sL[r] * alpha + sum;
        sM[r] = m_new;
        sAlpha[r] = alpha;
      }
    }
    __syncthreads();

    for (int i = tid; i < kHG * kD; i += kThreads) {
      const int r = i / kD, d = i % kD;
      sO[r * kLdO + d] *= sAlpha[r];
    }
    __syncthreads();

    // O += P V (V = the same latent rows), FP32 accumulation.
    for (int nt = warp; nt < kD / 16; nt += kWarps) {
      const int n0 = nt * 16;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::load_matrix_sync(acc, sO + n0, kLdO, wmma::mem_row_major);
#pragma unroll
      for (int k0 = 0; k0 < kBN; k0 += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
        wmma::load_matrix_sync(a, sP + k0, kLdP);
        wmma::load_matrix_sync(b, sK + k0 * kLdK + n0, kLdK);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(sO + n0, acc, kLdO, wmma::mem_row_major);
    }
    __syncthreads();
  }

  for (int i = tid; i < kHG * kD; i += kThreads) {
    const int r = i / kD, d = i % kD;
    o_part[((static_cast<int64_t>(t) * num_heads + h0 + r) * num_splits + split) * kD + d] =
        sO[r * kLdO + d];
  }
  if (tid < kHG) {
    const int64_t o = ((static_cast<int64_t>(t) * num_heads + h0 + tid) * num_splits + split) * 2;
    ml[o] = sM[tid];
    ml[o + 1] = sL[tid];
  }
}

__global__ void glm_sparse_mla_combine_kernel(const float* __restrict__ o_part,
                                              const float* __restrict__ ml, half* __restrict__ out,
                                              int64_t out_stride_t, int64_t out_stride_h,
                                              int num_heads, int num_splits) {
  const int h = blockIdx.x, t = blockIdx.y;
  const int64_t th = static_cast<int64_t>(t) * num_heads + h;
  float m_all = -INFINITY;
  for (int s = 0; s < num_splits; ++s) m_all = fmaxf(m_all, ml[(th * num_splits + s) * 2]);
  float l_all = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const float m = ml[(th * num_splits + s) * 2];
    if (m != -INFINITY) l_all += expf(m - m_all) * ml[(th * num_splits + s) * 2 + 1];
  }
  const float inv = l_all > 0.0f ? 1.0f / l_all : 0.0f;
  half* out_row = out + t * out_stride_t + h * out_stride_h;
  for (int d = threadIdx.x; d < kD; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
      const float m = ml[(th * num_splits + s) * 2];
      if (m != -INFINITY) acc += expf(m - m_all) * o_part[(th * num_splits + s) * kD + d];
    }
    out_row[d] = __float2half_rn(acc * inv);
  }
}

void glm_sparse_mla_fp8_fwd(const at::Tensor& q, const at::Tensor& cache,
                            const at::Tensor& indices, const at::Tensor& lengths, double scale,
                            at::Tensor& out, at::Tensor& o_part, at::Tensor& ml,
                            int64_t num_splits) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 3 && q.size(2) == kD,
              "q must be CUDA fp16 [tokens, heads, 512]");
  TORCH_CHECK(q.stride(2) == 1 && q.stride(1) % 8 == 0 && q.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
              "q rows must be contiguous and 16-byte aligned");
  const int64_t T = q.size(0), H = q.size(1);
  TORCH_CHECK(H % kHG == 0, "heads per rank must be a multiple of 16, got ", H);
  TORCH_CHECK(cache.is_cuda() && cache.scalar_type() == at::kByte && cache.dim() == 3 &&
                  cache.size(2) == kSlotBytes && cache.stride(2) == 1 &&
                  cache.stride(1) == kSlotBytes && cache.stride(0) % 4 == 0,
              "cache must be uint8 [blocks, block_size, 520] with contiguous blocks");
  TORCH_CHECK(indices.is_cuda() && indices.scalar_type() == at::kInt && indices.dim() == 2 &&
                  indices.size(0) == T && indices.stride(1) == 1,
              "indices must be int32 [tokens, index_width]");
  TORCH_CHECK(lengths.is_cuda() && lengths.scalar_type() == at::kInt && lengths.numel() == T &&
                  lengths.is_contiguous(),
              "lengths must be contiguous int32 [tokens]");
  TORCH_CHECK(out.scalar_type() == at::kHalf && out.sizes() == q.sizes() && out.stride(2) == 1,
              "out must be fp16 shaped like q");
  TORCH_CHECK(num_splits >= 1, "num_splits must be positive");
  TORCH_CHECK(o_part.scalar_type() == at::kFloat && o_part.is_contiguous() &&
                  o_part.numel() >= T * H * num_splits * kD,
              "o_part workspace too small");
  TORCH_CHECK(ml.scalar_type() == at::kFloat && ml.is_contiguous() &&
                  ml.numel() >= T * H * num_splits * 2,
              "ml workspace too small");
  if (T == 0) return;
  const at::cuda::OptionalCUDAGuard guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int index_width = static_cast<int>(indices.size(1));
  int keys_per_split = (index_width + static_cast<int>(num_splits) - 1) / static_cast<int>(num_splits);
  keys_per_split = (keys_per_split + kBN - 1) / kBN * kBN;
  if (keys_per_split == 0) keys_per_split = kBN;

  static bool smem_configured = false;
  if (!smem_configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(glm_sparse_mla_fp8_split_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    smem_configured = true;
  }
  const dim3 grid(static_cast<unsigned>(num_splits), static_cast<unsigned>(H / kHG),
                  static_cast<unsigned>(T));
  glm_sparse_mla_fp8_split_kernel<<<grid, kThreads, kSmemBytes, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr()), q.stride(0), q.stride(1),
      cache.data_ptr<uint8_t>(), cache.stride(0), static_cast<int>(cache.size(1)),
      cache.size(0) * cache.size(1), indices.data_ptr<int32_t>(), indices.stride(0), index_width,
      lengths.data_ptr<int32_t>(), static_cast<float>(scale), keys_per_split,
      o_part.data_ptr<float>(), ml.data_ptr<float>(), static_cast<int>(H),
      static_cast<int>(num_splits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  glm_sparse_mla_combine_kernel<<<dim3(static_cast<unsigned>(H), static_cast<unsigned>(T)), 128, 0,
                                  stream>>>(o_part.data_ptr<float>(), ml.data_ptr<float>(),
                                            reinterpret_cast<half*>(out.data_ptr()), out.stride(0),
                                            out.stride(1), static_cast<int>(H),
                                            static_cast<int>(num_splits));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace glm_attn_v100

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_mla_fp8_fwd", &glm_attn_v100::glm_sparse_mla_fp8_fwd,
        "GLM-5.3 sparse MLA over packed E4M3 latent KV (SM70, FP32 softmax)");
  m.attr("latent_dim") = glm_attn_v100::kD;
  m.attr("heads_per_cta") = glm_attn_v100::kHG;
  m.attr("keys_per_tile") = glm_attn_v100::kBN;
  m.attr("precision_version") = 1;
}
