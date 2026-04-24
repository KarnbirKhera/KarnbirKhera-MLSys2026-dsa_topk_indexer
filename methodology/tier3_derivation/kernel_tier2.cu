// dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
// Torch binding. Fully batched: one launch per phase (dequant, bmm, scoring, topk,
// translate) across the entire batch; no per-batch host loop, seq_lens stays on GPU.

#include <cstdint>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

constexpr int INDEX_HEAD_DIM  = 128;
constexpr int PAGE_SIZE       = 64;
constexpr int PAGE_DATA_BYTES = 8192;
constexpr int PAGE_BYTES      = 8448;
constexpr int TOPK            = 2048;

__device__ __forceinline__ float fp8_e4m3_to_fp32(uint8_t byte) {
    const __nv_fp8_storage_t s  = static_cast<__nv_fp8_storage_t>(byte);
    const __half_raw         hr = __nv_cvt_fp8_to_halfraw(s, __NV_E4M3);
    return __half2float(__half(hr));
}

// Batched dequant: writes [B, max_num_pages*PAGE_SIZE, 128] FP32. Padding rows
// (row >= seq_lens[b]) are zero-filled — zeros in K mean matmul produces 0 at
// those columns, which are later overwritten with -INF in the scoring kernel.
__global__ void dequant_gather_batched_kernel(
    const uint8_t* __restrict__ k_cache,
    const int32_t* __restrict__ block_table,   // [B, max_num_pages]
    const int32_t* __restrict__ seq_lens,      // [B]
    float*         __restrict__ K_deq_all,     // [B, max_rows, 128]
    int max_num_pages)
{
    const int page_id  = blockIdx.x;
    const int tok      = blockIdx.y;
    const int b        = blockIdx.z;
    const int lane     = threadIdx.x;
    const int max_rows = max_num_pages * PAGE_SIZE;
    const int row      = page_id * PAGE_SIZE + tok;
    const int k_base   = lane * 4;

    float* K_out = K_deq_all
                   + size_t(b) * max_rows * INDEX_HEAD_DIM
                   + row * INDEX_HEAD_DIM
                   + k_base;

    const int sl = seq_lens[b];
    if (row >= sl) {
        *reinterpret_cast<float4*>(K_out) = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }
    const int32_t phys_i32 = block_table[b * max_num_pages + page_id];
    const int64_t phys     = static_cast<int64_t>(phys_i32);
    const uint8_t* pb      = k_cache + phys * PAGE_BYTES;
    const float scale = *reinterpret_cast<const float*>(pb + PAGE_DATA_BYTES + tok * 4);
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(
        pb + tok * INDEX_HEAD_DIM + k_base);
    const float f0 = fp8_e4m3_to_fp32(static_cast<uint8_t>((packed >>  0) & 0xff)) * scale;
    const float f1 = fp8_e4m3_to_fp32(static_cast<uint8_t>((packed >>  8) & 0xff)) * scale;
    const float f2 = fp8_e4m3_to_fp32(static_cast<uint8_t>((packed >> 16) & 0xff)) * scale;
    const float f3 = fp8_e4m3_to_fp32(static_cast<uint8_t>((packed >> 24) & 0xff)) * scale;
    *reinterpret_cast<float4*>(K_out) = make_float4(f0, f1, f2, f3);
}

// Batched fused scoring: final_scores[b, t] = sum_{h} max(0, scores[b, h, t]) * weights[b, h],
// with positions t >= seq_lens[b] stamped to -INF so at::topk orders them last.
__global__ void weighted_relu_sum_batched_kernel(
    const float*   __restrict__ scores_all,      // [B, 64, N_max]
    const float*   __restrict__ weights_all,     // [B, 64]
    const int32_t* __restrict__ seq_lens,        // [B]
    float*         __restrict__ final_scores,    // [B, N_max]
    int N_max)
{
    constexpr int H = 64;
    const int b   = blockIdx.y;
    const int tid = threadIdx.x;
    const int t   = blockIdx.x * blockDim.x + tid;

    __shared__ float w_smem[H];
    if (tid < H) w_smem[tid] = weights_all[b * H + tid];
    __syncthreads();

    if (t >= N_max) return;
    const int sl = seq_lens[b];
    if (t >= sl) {
        final_scores[b * N_max + t] = -INFINITY;
        return;
    }

    const float* scores_b = scores_all + size_t(b) * H * N_max;
    float acc = 0.f;
    #pragma unroll 8
    for (int h = 0; h < H; ++h) {
        const float s = scores_b[h * N_max + t];
        acc = fmaf(fmaxf(s, 0.f), w_smem[h], acc);
    }
    final_scores[b * N_max + t] = acc;
}

// Batched index translation. local_idx_all is [B, k_req] where k_req = min(TOPK, N_max);
// its batch stride is k_req, not TOPK. Output topk_indices is [B, TOPK]; positions
// t >= actual_topk_b are left as -1 by the caller's fill_ and skipped here.
__global__ void translate_indices_batched_kernel(
    const int64_t* __restrict__ local_idx_all,   // [B, k_req]
    const int32_t* __restrict__ block_table,     // [B, max_num_pages]
    const int32_t* __restrict__ seq_lens,        // [B]
    int32_t*       __restrict__ topk_indices,    // [B, TOPK]
    int max_num_pages,
    int k_req)
{
    const int b = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= k_req) return;

    const int sl = seq_lens[b];
    const int actual_topk = sl < TOPK ? sl : TOPK;
    if (t >= actual_topk) return;  // leave -1 from caller's fill_

    const int64_t lid  = local_idx_all[size_t(b) * k_req + t];
    const int32_t phys = block_table[b * max_num_pages + static_cast<int>(lid >> 6)];
    topk_indices[b * TOPK + t] = phys * PAGE_SIZE + static_cast<int32_t>(lid & 63);
}

void launch_topk_c(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor topk_indices)
{
    const at::cuda::CUDAGuard device_guard(q_index_fp8.device());
    const int B             = static_cast<int>(q_index_fp8.size(0));
    const int max_num_pages = static_cast<int>(block_table.size(1));
    const int N_max         = max_num_pages * PAGE_SIZE;
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto q_fp32_all = q_index_fp8.to(torch::kFloat32);                  // [B, 64, 128]
    auto k_cache_u8 = k_index_cache_fp8.view(torch::kUInt8);
    const uint8_t* k_cache_ptr = k_cache_u8.data_ptr<uint8_t>();
    const int32_t* bt_ptr  = block_table.data_ptr<int32_t>();
    const int32_t* sl_ptr  = seq_lens.data_ptr<int32_t>();

    auto opts_f32 = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(q_index_fp8.device());
    auto K_deq_all = torch::empty({B, N_max, INDEX_HEAD_DIM}, opts_f32);

    // Phase 1: batched dequant over (page, tok, batch).
    dequant_gather_batched_kernel<<<
        dim3(max_num_pages, PAGE_SIZE, B), 32, 0, stream>>>(
        k_cache_ptr, bt_ptr, sl_ptr, K_deq_all.data_ptr<float>(), max_num_pages);
    AT_CUDA_CHECK(cudaGetLastError());

    // Phase 2: batched matmul [B,64,128] x [B,128,N_max] -> [B,64,N_max].
    auto scores_all = torch::bmm(q_fp32_all, K_deq_all.transpose(-2, -1));

    // Phase 3: batched fused relu * weights * sum-over-heads, with -INF padding mask.
    auto final_scores = torch::empty({B, N_max}, opts_f32);
    weighted_relu_sum_batched_kernel<<<
        dim3((N_max + 255) / 256, B), 256, 0, stream>>>(
        scores_all.data_ptr<float>(),
        weights.data_ptr<float>(),
        sl_ptr,
        final_scores.data_ptr<float>(),
        N_max);
    AT_CUDA_CHECK(cudaGetLastError());

    // Phase 4: single batched top-K along dim -1.
    const int k_req = std::min(TOPK, N_max);
    auto result = at::topk(final_scores, k_req, -1, true, true);
    auto local_idx = std::get<1>(result).contiguous();  // [B, k_req] int64

    // Phase 5: batched index translation. Fills -1 for t >= actual_topk_b.
    topk_indices.fill_(-1);
    translate_indices_batched_kernel<<<
        dim3((k_req + 255) / 256, B), 256, 0, stream>>>(
        local_idx.data_ptr<int64_t>(),
        bt_ptr, sl_ptr,
        topk_indices.data_ptr<int32_t>(),
        max_num_pages,
        k_req);
    AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_topk_c", &launch_topk_c, "DSA TopK FP8 indexer");
}
