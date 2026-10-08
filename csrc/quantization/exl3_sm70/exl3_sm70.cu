// EXL3 (sm70_colmajor tile order, mcg codebook, K = 2) routed-expert kernels for Volta (sm_70).
// Standalone shared library, called through ctypes with raw device pointers on the caller's CUDA stream.
// Everything is shape-static per slot count -> CUDA-graph capturable; expert ids are read on the device.
//
// One "slot" = one (token, routed expert) assignment, s = t * topk + j.
//   pre   : X'[s]  = had128(x[t] * suh[e])                      fp16 [S, k]
//   gemv  : Y'[s]  = X'[s] @ T[e]      (trellis decode + mma.m8n8k4, fp32 accumulate)   fp32 [S, n]
//   mid   : H'[s]  = had128(silu(had128(Yg')*svh_g[e]) * (had128(Yu')*svh_u[e]) * suh_d[e])   fp16 [S, n]
//   post  : out[t] = sum_j w[s] * had128(O'[s]) * svh_d[e]      fp32 [T, n]
// W = diag(suh) H T H diag(svh), H = blockwise Sylvester-128 / sqrt(128)  (exl3_ref.reconstruct).
//
// Build: nvcc -O3 -arch=sm_70 -std=c++17 -shared -Xcompiler -fPIC -o libexl3_sm70.so exl3_sm70.cu
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ void mma884(float (&d)[8], const uint32_t (&a)[2], const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ half2 mcg2(uint32_t s0, uint32_t s1)
{
    uint32_t x0 = s0 * 0xCBAC1FEDu, x1 = s1 * 0xCBAC1FEDu;
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
    half2 h0 = *reinterpret_cast<half2*>(&x0), h1 = *reinterpret_cast<half2*>(&x1);
    return __hadd2(__lows2half2(h0, h1), __highs2half2(h0, h1));
}

// K = bits per weight (2, 3 or 4). A 16x16 tile is 256K bits = 8K u32 words, MSB-first (the half-swapped int16
// storage read as little-endian u32). Lane-run (column c, half h) = 8 consecutive positions 16c + 8h + i; its window
// starts at bit s = ((16c + 8h + 1)K - 16) mod 256K and spans 16 + 7K bits, which always fits two aligned words.
template <int K>
__device__ __forceinline__ void decode8(uint32_t wa, uint32_t wb, int o, uint32_t (&b)[4])
{
    uint64_t v = ((uint64_t) wa << 32) | wb;
    uint32_t st[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) st[i] = (uint32_t) (v >> (48 - o - i * K)) & 0xffffu;
    #pragma unroll
    for (int j = 0; j < 4; ++j) { half2 r = mcg2(st[2 * j], st[2 * j + 1]); b[j] = *reinterpret_cast<uint32_t*>(&r); }
}

// The warp's two-tile block (16K words) is loaded once, coalesced, then lanes fetch their two window words by shuffle.
template <int K>
struct TileWords
{
    uint32_t r0, r1;
    __device__ __forceinline__ void load(const uint32_t* __restrict__ blk, int L)
    {
        if (K == 2) { r0 = blk[L]; r1 = 0; }
        else if (2 * L < 16 * K) { uint2 v = reinterpret_cast<const uint2*>(blk)[L]; r0 = v.x; r1 = v.y; }
        else { r0 = 0; r1 = 0; }
    }
    __device__ __forceinline__ uint32_t word(int j) const
    {
        if (K == 2) return __shfl_sync(0xffffffffu, r0, j);
        uint32_t x0 = __shfl_sync(0xffffffffu, r0, j >> 1), x1 = __shfl_sync(0xffffffffu, r1, j >> 1);
        return (j & 1) ? x1 : x0;
    }
};

// Lane-constant window geometry for tile t (0/1 within the strip), column c.
template <int K>
struct Win
{
    int ja[2], jb[2], o[2];
    __device__ __forceinline__ Win(int t, int c)
    {
        constexpr int TW = 8 * K, LB = 256 * K;
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            int s = (((16 * c + 8 * h + 1) * K - 16) % LB + LB) % LB;
            ja[h] = t * TW + (s >> 5); jb[h] = t * TW + ((s >> 5) + 1) % TW; o[h] = s & 31;
        }
    }
};

// grid (n / 32, S), block 4 warps. Slot s reads expert eid[s]'s matrix; M = 1 (row 0 of the m8n8k4 tile).
constexpr int WARPS = 4;
template <int K>
__global__ void __launch_bounds__(WARPS * 32)
gemv_kernel(const uint32_t* __restrict__ trellis, const int* __restrict__ eid, const half* __restrict__ x,
            float* __restrict__ y, int k, int n, int ktiles_per_warp, size_t expert_words,
            int S, const uint32_t* __restrict__ trellis2, const half* __restrict__ x2, float* __restrict__ y2)
{
    int L = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int s = blockIdx.y;
    if (s >= S) { s -= S; trellis = trellis2; x = x2; y = y2; }      // second projection (gate+up fused launch)
    int n0 = blockIdx.x * 32;
    int q = (L >> 2) & 3;
    int nw = 8 * q + (L >> 4) * 4 + L % 4;
    int t = nw >> 4, c = nw & 15;
    int row = (L >> 4) * 4 + L % 4;
    int tiles_n = n >> 4;
    int tk0 = warp * ktiles_per_warp;
    const uint32_t* tr = trellis + (size_t) eid[s] * expert_words;
    const half* xr = x + (size_t) s * k;

    constexpr int TW = 8 * K;
    Win<K> win(t, c);

    float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int it = 0; it < ktiles_per_warp; ++it)
    {
        int tk = tk0 + it;
        const uint32_t* blk = tr + ((size_t) tk * tiles_n + (n0 >> 4)) * TW;       // two contiguous tiles
        TileWords<K> tw; tw.load(blk, L);
        uint32_t bb[2][4];
        decode8<K>(tw.word(win.ja[0]), tw.word(win.jb[0]), win.o[0], bb[0]);
        decode8<K>(tw.word(win.ja[1]), tw.word(win.jb[1]), win.o[1], bb[1]);
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            int ks = tk * 2 + h;
            uint32_t b0[2] = {bb[h][0], bb[h][1]}, b1[2] = {bb[h][2], bb[h][3]};
            uint4 xa = row == 0 ? *reinterpret_cast<const uint4*>(xr + ks * 8) : make_uint4(0, 0, 0, 0);
            uint32_t a0[2] = {xa.x, xa.y}, a1[2] = {xa.z, xa.w};
            mma884(d, a0, b0);
            mma884(d, a1, b1);
        }
    }

    // D fragment: lane L, reg i -> row (L&1) + ((i>>1)&1)*2 + (L>=16 ? 4 : 0), col ((i>>2)&1)*4 + (L&2) + (i&1)
    __shared__ float red[WARPS][32];
    if (L < 16 && !(L & 1))
    {
        red[warp][8 * q + (L & 2)]         = d[0];
        red[warp][8 * q + (L & 2) + 1]     = d[1];
        red[warp][8 * q + 4 + (L & 2)]     = d[4];
        red[warp][8 * q + 4 + (L & 2) + 1] = d[5];
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float acc = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) acc += red[w][threadIdx.x];
        y[(size_t) s * n + n0 + threadIdx.x] = acc;
    }
}

// Grouped variant for prefill: a group = up to 8 slots routed to the SAME expert. The expert's tiles are decoded
// once and multiplied against all 8 rows (the m8n8k4 A operand carries 8 rows). rows[g * 8 + m] = slot index or -1.
// grid (n / 32, G or 2G), block 4 warps. Groups with rows[g * 8] < 0 exit immediately (static worst-case launches).
template <int K>
__global__ void __launch_bounds__(WARPS * 32)
gemv_grp_kernel(const uint32_t* __restrict__ trellis, const int* __restrict__ gexp, const int* __restrict__ rows,
                const half* __restrict__ x, float* __restrict__ y, int k, int n, int ktiles_per_warp,
                size_t expert_words, int G, const uint32_t* __restrict__ trellis2, const half* __restrict__ x2,
                float* __restrict__ y2)
{
    int L = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int g = blockIdx.y;
    if (g >= G) { g -= G; trellis = trellis2; x = x2; y = y2; }
    const int* gr = rows + (size_t) g * 8;
    if (gr[0] < 0) return;
    int n0 = blockIdx.x * 32;
    int q = (L >> 2) & 3;
    int nw = 8 * q + (L >> 4) * 4 + L % 4;
    int t = nw >> 4, c = nw & 15;
    int row = (L >> 4) * 4 + L % 4;                 // A-operand row held by this lane (0..7)
    int slot = gr[row];
    int tiles_n = n >> 4;
    int tk0 = warp * ktiles_per_warp;
    const uint32_t* tr = trellis + (size_t) gexp[g] * expert_words;
    const half* xr = x + (size_t) (slot < 0 ? 0 : slot) * k;

    constexpr int TW = 8 * K;
    Win<K> win(t, c);

    float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int it = 0; it < ktiles_per_warp; ++it)
    {
        int tk = tk0 + it;
        const uint32_t* blk = tr + ((size_t) tk * tiles_n + (n0 >> 4)) * TW;
        TileWords<K> tw; tw.load(blk, L);
        uint32_t bb[2][4];
        decode8<K>(tw.word(win.ja[0]), tw.word(win.jb[0]), win.o[0], bb[0]);
        decode8<K>(tw.word(win.ja[1]), tw.word(win.jb[1]), win.o[1], bb[1]);
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            int ks = tk * 2 + h;
            uint32_t b0[2] = {bb[h][0], bb[h][1]}, b1[2] = {bb[h][2], bb[h][3]};
            uint4 xa = slot >= 0 ? *reinterpret_cast<const uint4*>(xr + ks * 8) : make_uint4(0, 0, 0, 0);
            uint32_t a0[2] = {xa.x, xa.y}, a1[2] = {xa.z, xa.w};
            mma884(d, a0, b0);
            mma884(d, a1, b1);
        }
    }

    // D fragment: lane L, reg i -> row (L&1) + ((i>>1)&1)*2 + (L>=16 ? 4 : 0), col ((i>>2)&1)*4 + (L&2) + (i&1)
    __shared__ float red[WARPS][8][32];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        red[warp][(L & 1) + ((i >> 1) & 1) * 2 + (L >= 16 ? 4 : 0)][8 * q + ((i >> 2) & 1) * 4 + (L & 2) + (i & 1)] = d[i];
    __syncthreads();
    for (int e = threadIdx.x; e < 256; e += WARPS * 32)
    {
        int m = e >> 5, nn = e & 31;
        int so = gr[m];
        if (so < 0) continue;
        float acc = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) acc += red[w][m][nn];
        y[(size_t) so * n + n0 + nn] = acc;
    }
}

// 128-point Walsh-Hadamard (natural/Sylvester order) in registers: one warp per 128-chunk, lane l holds
// elements 4l .. 4l+3. Stages 1, 2 are in-lane; stages 4 .. 64 exchange with lane l ^ (len / 4). No shared memory.
__device__ __forceinline__ void fwht128w(float (&v)[4])
{
    int lane = threadIdx.x & 31;
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
    #pragma unroll
    for (int dl = 1; dl < 32; dl <<= 1)
    {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float o = __shfl_xor_sync(0xffffffffu, v[i], dl);
            v[i] = (lane & dl) ? o - v[i] : v[i] + o;
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= 0.08838834764831845f;          // 1 / sqrt(128)
}

__device__ __forceinline__ void ld4h(const half* p, float (&v)[4])
{
    uint2 r = *reinterpret_cast<const uint2*>(p);
    half2 h0 = *reinterpret_cast<half2*>(&r.x), h1 = *reinterpret_cast<half2*>(&r.y);
    float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
    v[0] = f0.x; v[1] = f0.y; v[2] = f1.x; v[3] = f1.y;
}

__device__ __forceinline__ void st4h(half* p, const float (&v)[4])
{
    half2 h0 = __floats2half2_rn(v[0], v[1]), h1 = __floats2half2_rn(v[2], v[3]);
    uint2 r; r.x = *reinterpret_cast<uint32_t*>(&h0); r.y = *reinterpret_cast<uint32_t*>(&h1);
    *reinterpret_cast<uint2*>(p) = r;
}

constexpr int GW = 4;            // warps (= 128-chunks) per block in the glue kernels; dims are multiples of GW*128

__global__ void pre_kernel(const half* __restrict__ x, const int* __restrict__ eid, const half* __restrict__ suh,
                           half* __restrict__ out, int k, int topk, int S, const half* __restrict__ suh2,
                           half* __restrict__ out2)
{
    int s = blockIdx.y;
    if (s >= S) { s -= S; suh = suh2; out = out2; }
    int o = (blockIdx.x * GW + (threadIdx.x >> 5)) * 128 + (threadIdx.x & 31) * 4;
    float v[4], g[4];
    ld4h(x + (size_t) (s / topk) * k + o, v);
    ld4h(suh + (size_t) eid[s] * k + o, g);
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= g[i];
    fwht128w(v);
    st4h(out + (size_t) s * k + o, v);
}

__global__ void mid_kernel(const float* __restrict__ yg, const float* __restrict__ yu, const int* __restrict__ eid,
                           const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                           const half* __restrict__ suh_d, half* __restrict__ out, int n)
{
    int s = blockIdx.y;
    int o = (blockIdx.x * GW + (threadIdx.x >> 5)) * 128 + (threadIdx.x & 31) * 4;
    size_t e = (size_t) eid[s] * n + o;
    float g[4], u[4], sg[4], su[4], sd[4];
    float4 a = *reinterpret_cast<const float4*>(yg + (size_t) s * n + o);
    float4 b = *reinterpret_cast<const float4*>(yu + (size_t) s * n + o);
    g[0] = a.x; g[1] = a.y; g[2] = a.z; g[3] = a.w; u[0] = b.x; u[1] = b.y; u[2] = b.z; u[3] = b.w;
    fwht128w(g); fwht128w(u);
    ld4h(svh_g + e, sg); ld4h(svh_u + e, su); ld4h(suh_d + e, sd);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float gg = g[i] * sg[i];
        g[i] = gg / (1.f + __expf(-gg)) * (u[i] * su[i]) * sd[i];
    }
    fwht128w(g);
    st4h(out + (size_t) s * n + o, g);
}

__global__ void post_kernel(const float* __restrict__ yd, const int* __restrict__ eid, const float* __restrict__ w,
                            const half* __restrict__ svh_d, float* __restrict__ out, int n, int topk)
{
    int t = blockIdx.y;
    int o = (blockIdx.x * GW + (threadIdx.x >> 5)) * 128 + (threadIdx.x & 31) * 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < topk; ++j)
    {
        int s = t * topk + j;
        float v[4], sv[4];
        float4 a = *reinterpret_cast<const float4*>(yd + (size_t) s * n + o);
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
        fwht128w(v);
        ld4h(svh_d + (size_t) eid[s] * n + o, sv);
        float ww = w[s];
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] += ww * v[i] * sv[i];
    }
    *reinterpret_cast<float4*>(out + (size_t) t * n + o) = make_float4(acc[0], acc[1], acc[2], acc[3]);
}

extern "C" {

int exl3_sm70_gemv(const void* trellis, const int* eid, const void* x, float* y, int S, int k, int n,
                   long long expert_words, void* stream, int K)
{
    int ktiles = k / 16;
    if (k % 16 || n % 32 || ktiles % WARPS || S < 1 || S > 65535) return -1;
    if (K == 2) gemv_kernel<2><<<dim3(n / 32, S), WARPS * 32, 0, (cudaStream_t) stream>>>(
        (const uint32_t*) trellis, eid, (const half*) x, y, k, n, ktiles / WARPS, (size_t) expert_words,
        S, nullptr, nullptr, nullptr);
    else if (K == 3) gemv_kernel<3><<<dim3(n / 32, S), WARPS * 32, 0, (cudaStream_t) stream>>>(
        (const uint32_t*) trellis, eid, (const half*) x, y, k, n, ktiles / WARPS, (size_t) expert_words,
        S, nullptr, nullptr, nullptr);
    else if (K == 4) gemv_kernel<4><<<dim3(n / 32, S), WARPS * 32, 0, (cudaStream_t) stream>>>(
        (const uint32_t*) trellis, eid, (const half*) x, y, k, n, ktiles / WARPS, (size_t) expert_words,
        S, nullptr, nullptr, nullptr);
    else return -2;
    return (int) cudaPeekAtLastError();
}

// gate + up in one launch: slots [0, S) use (trellis, x, y), slots [S, 2S) use (trellis2, x2, y2).
int exl3_sm70_gemv2(const void* trellis, const void* trellis2, const int* eid, const void* x, const void* x2,
                    float* y, float* y2, int S, int k, int n, long long expert_words, void* stream, int K)
{
    int ktiles = k / 16;
    if (k % 16 || n % 32 || ktiles % WARPS || S < 1 || 2 * S > 65535) return -1;
    if (K == 2) gemv_kernel<2><<<dim3(n / 32, 2 * S), WARPS * 32, 0, (cudaStream_t) stream>>>(
        (const uint32_t*) trellis, eid, (const half*) x, y, k, n, ktiles / WARPS, (size_t) expert_words,
        S, (const uint32_t*) trellis2, (const half*) x2, y2);
    else if (K == 3) gemv_kernel<3><<<dim3(n / 32, 2 * S), WARPS * 32, 0, (cudaStream_t) stream>>>(
        (const uint32_t*) trellis, eid, (const half*) x, y, k, n, ktiles / WARPS, (size_t) expert_words,
        S, (const uint32_t*) trellis2, (const half*) x2, y2);
    else if (K == 4) gemv_kernel<4><<<dim3(n / 32, 2 * S), WARPS * 32, 0, (cudaStream_t) stream>>>(
        (const uint32_t*) trellis, eid, (const half*) x, y, k, n, ktiles / WARPS, (size_t) expert_words,
        S, (const uint32_t*) trellis2, (const half*) x2, y2);
    else return -2;
    return (int) cudaPeekAtLastError();
}

// Grouped launches. G groups; gate+up fused variant covers 2G rows of the grid.
int exl3_sm70_gemvg(const void* trellis, const int* gexp, const int* rows, const void* x, float* y, int G, int k, int n,
                    long long expert_words, void* stream, int K)
{
    int ktiles = k / 16;
    if (k % 16 || n % 32 || ktiles % WARPS || G < 1 || G > 65535) return -1;
    if (K == 2) gemv_grp_kernel<2><<<dim3(n / 32, G), WARPS * 32, 0, (cudaStream_t) stream>>>(
        (const uint32_t*) trellis, gexp, rows, (const half*) x, y, k, n, ktiles / WARPS, (size_t) expert_words,
        G, nullptr, nullptr, nullptr);
    else if (K == 3) gemv_grp_kernel<3><<<dim3(n / 32, G), WARPS * 32, 0, (cudaStream_t) stream>>>(
        (const uint32_t*) trellis, gexp, rows, (const half*) x, y, k, n, ktiles / WARPS, (size_t) expert_words,
        G, nullptr, nullptr, nullptr);
    else if (K == 4) gemv_grp_kernel<4><<<dim3(n / 32, G), WARPS * 32, 0, (cudaStream_t) stream>>>(
        (const uint32_t*) trellis, gexp, rows, (const half*) x, y, k, n, ktiles / WARPS, (size_t) expert_words,
        G, nullptr, nullptr, nullptr);
    else return -2;
    return (int) cudaPeekAtLastError();
}

int exl3_sm70_gemvg2(const void* trellis, const void* trellis2, const int* gexp, const int* rows, const void* x,
                     const void* x2, float* y, float* y2, int G, int k, int n, long long expert_words, void* stream, int K)
{
    int ktiles = k / 16;
    if (k % 16 || n % 32 || ktiles % WARPS || G < 1 || 2 * G > 65535) return -1;
    if (K == 2) gemv_grp_kernel<2><<<dim3(n / 32, 2 * G), WARPS * 32, 0, (cudaStream_t) stream>>>(
        (const uint32_t*) trellis, gexp, rows, (const half*) x, y, k, n, ktiles / WARPS, (size_t) expert_words,
        G, (const uint32_t*) trellis2, (const half*) x2, y2);
    else if (K == 3) gemv_grp_kernel<3><<<dim3(n / 32, 2 * G), WARPS * 32, 0, (cudaStream_t) stream>>>(
        (const uint32_t*) trellis, gexp, rows, (const half*) x, y, k, n, ktiles / WARPS, (size_t) expert_words,
        G, (const uint32_t*) trellis2, (const half*) x2, y2);
    else if (K == 4) gemv_grp_kernel<4><<<dim3(n / 32, 2 * G), WARPS * 32, 0, (cudaStream_t) stream>>>(
        (const uint32_t*) trellis, gexp, rows, (const half*) x, y, k, n, ktiles / WARPS, (size_t) expert_words,
        G, (const uint32_t*) trellis2, (const half*) x2, y2);
    else return -2;
    return (int) cudaPeekAtLastError();
}

int exl3_sm70_pre(const void* x, const int* eid, const void* suh, void* out, int S, int k, int topk, void* stream)
{
    if (k % 512 || S < 1 || S > 65535) return -1;
    pre_kernel<<<dim3(k / 512, S), 128, 0, (cudaStream_t) stream>>>((const half*) x, eid, (const half*) suh, (half*) out, k, topk,
                                                                  S, nullptr, nullptr);
    return (int) cudaPeekAtLastError();
}

int exl3_sm70_pre2(const void* x, const int* eid, const void* suh, const void* suh2, void* out, void* out2, int S, int k,
                   int topk, void* stream)
{
    if (k % 512 || S < 1 || 2 * S > 65535) return -1;
    pre_kernel<<<dim3(k / 512, 2 * S), 128, 0, (cudaStream_t) stream>>>((const half*) x, eid, (const half*) suh, (half*) out, k, topk,
                                                                      S, (const half*) suh2, (half*) out2);
    return (int) cudaPeekAtLastError();
}

int exl3_sm70_mid(const float* yg, const float* yu, const int* eid, const void* svh_g, const void* svh_u,
                  const void* suh_d, void* out, int S, int n, void* stream)
{
    if (n % 512 || S < 1 || S > 65535) return -1;
    mid_kernel<<<dim3(n / 512, S), 128, 0, (cudaStream_t) stream>>>(yg, yu, eid, (const half*) svh_g, (const half*) svh_u,
                                                                  (const half*) suh_d, (half*) out, n);
    return (int) cudaPeekAtLastError();
}

int exl3_sm70_post(const float* yd, const int* eid, const float* w, const void* svh_d, float* out, int T, int n,
                   int topk, void* stream)
{
    if (n % 512 || T < 1 || T > 65535) return -1;
    post_kernel<<<dim3(n / 512, T), 128, 0, (cudaStream_t) stream>>>(yd, eid, w, (const half*) svh_d, out, n, topk);
    return (int) cudaPeekAtLastError();
}

}
