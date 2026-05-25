#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../../utils/check.h"
#include "attention_iluvatar.cuh"

#include <cmath>
#include <limits>

namespace {
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kMaxHeadDim = 128;
constexpr int kMaxTotalSeqLen = 8;
constexpr int kMaxHeadChunks = (kMaxHeadDim + kWarpSize - 1) / kWarpSize;
constexpr unsigned kFullWarpMask = 0xffffffffu;

__device__ __forceinline__ float to_float_half(half v) {
    return __half2float(v);
}

__device__ __forceinline__ half from_float_half(float v) {
    return __float2half(v);
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}
} // namespace

__global__ void __launch_bounds__(kWarpSize * kWarpsPerBlock)
attention_fused_kernel_half(const half *q,
                            const half *k_cache,
                            const half *v_cache,
                            half *out,
                            int n_head,
                            int seq_len,
                            int head_dim,
                            int total_seq_len,
                            int cache_len,
                            int pos) {
    const int lane = threadIdx.x;
    const int warp_idx = threadIdx.y;
    const int q_idx = blockIdx.x;
    const int h = blockIdx.y * blockDim.y + warp_idx;
    if (q_idx >= seq_len || h >= n_head) {
        return;
    }

    int max_t = pos + q_idx;
    if (max_t >= total_seq_len) {
        max_t = total_seq_len - 1;
    }
    const int active_t = max_t + 1;
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));
    const half *q_ptr = q + (h * seq_len + q_idx) * head_dim;

    int dims[kMaxHeadChunks];
    float q_vals[kMaxHeadChunks];
    float acc[kMaxHeadChunks];
    int chunk_count = 0;
    for (int d = lane; d < head_dim; d += kWarpSize) {
        dims[chunk_count] = d;
        q_vals[chunk_count] = to_float_half(q_ptr[d]);
        acc[chunk_count] = 0.0f;
        ++chunk_count;
    }

    float scores[kMaxTotalSeqLen];
    float max_score = -INFINITY;
    for (int t = 0; t < active_t; ++t) {
        const half *k_ptr = k_cache + (h * cache_len + t) * head_dim;
        float partial = 0.0f;
        for (int i = 0; i < chunk_count; ++i) {
            partial += q_vals[i] * to_float_half(k_ptr[dims[i]]);
        }
        float dot = warp_sum(partial);
        if (lane == 0) {
            const float score = dot * inv_sqrt;
            scores[t] = score;
            max_score = fmaxf(max_score, score);
        }
    }

    float sum = 0.0f;
    if (lane == 0) {
        for (int t = 0; t < active_t; ++t) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
    }
    sum = __shfl_sync(kFullWarpMask, sum, 0);
    const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int t = 0; t < active_t; ++t) {
        float weight = 0.0f;
        if (lane == 0) {
            weight = scores[t] * inv_sum;
        }
        weight = __shfl_sync(kFullWarpMask, weight, 0);

        const half *v_ptr = v_cache + (h * cache_len + t) * head_dim;
        for (int i = 0; i < chunk_count; ++i) {
            acc[i] += weight * to_float_half(v_ptr[dims[i]]);
        }
    }

    half *out_ptr = out + (q_idx * n_head + h) * head_dim;
    for (int i = 0; i < chunk_count; ++i) {
        out_ptr[dims[i]] = from_float_half(acc[i]);
    }
}

namespace op::attention::iluvatar {

infiniStatus_t fused_attention(infiniDtype_t dtype,
                               void *out,
                               const void *q,
                               const void *k,
                               const void *v,
                               void *k_cache,
                               void *v_cache,
                               size_t n_head,
                               size_t seq_len,
                               size_t head_dim,
                               size_t total_seq_len,
                               size_t cache_len,
                               size_t pos,
                               void *stream) {
    (void)k;
    (void)v;
    if (dtype != INFINI_DTYPE_F16) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (head_dim == 0 || head_dim > kMaxHeadDim || total_seq_len == 0 || total_seq_len > kMaxTotalSeqLen || cache_len < total_seq_len) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const int n_head_i = static_cast<int>(n_head);
    const int seq_len_i = static_cast<int>(seq_len);
    const int head_dim_i = static_cast<int>(head_dim);
    const int total_seq_len_i = static_cast<int>(total_seq_len);
    const int cache_len_i = static_cast<int>(cache_len);
    const int pos_i = static_cast<int>(pos);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    dim3 block(kWarpSize, kWarpsPerBlock);
    dim3 grid(seq_len_i,
              (n_head_i + block.y - 1) / block.y);

    attention_fused_kernel_half<<<grid, block, 0, cuda_stream>>>(
        reinterpret_cast<const half *>(q),
        reinterpret_cast<const half *>(k_cache),
        reinterpret_cast<const half *>(v_cache),
        reinterpret_cast<half *>(out),
        n_head_i,
        seq_len_i,
        head_dim_i,
        total_seq_len_i,
        cache_len_i,
        pos_i);
    CHECK_CUDA(cudaGetLastError());

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::attention::iluvatar
