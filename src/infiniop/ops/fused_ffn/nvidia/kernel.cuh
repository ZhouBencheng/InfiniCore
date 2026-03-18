#ifndef __FUSED_FFN_CUDA_KERNEL_H__
#define __FUSED_FFN_CUDA_KERNEL_H__

#include <cub/block/block_reduce.cuh>

// RMSNorm preprocessing kernel
// Each block processes one token
// Computes: normalized = x * rsqrt(mean(x^2) + eps) * w
template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata, typename Tweight>
__device__ void rmsnormBlock(
    Tdata *__restrict__ normalized,
    const Tdata *__restrict__ x,
    const Tweight *__restrict__ w,
    size_t dim,
    float epsilon,
    ptrdiff_t in_stride,
    ptrdiff_t out_stride) {

    size_t token_idx = blockIdx.x;
    auto x_ptr = x + token_idx * in_stride;
    auto norm_ptr = normalized + token_idx * out_stride;

    // Compute sum of squares
    Tcompute sum_squared = 0;
    for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        Tcompute val = Tcompute(x_ptr[i]);
        sum_squared += val * val;
    }

    // Block-reduce sum of squares
    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum_squared = BlockReduce(temp_storage).Sum(sum_squared);

    // Thread 0 computes RMS = 1/sqrt(ss/dim + epsilon)
    __shared__ Tcompute rms;
    if (threadIdx.x == 0) {
        rms = Tcompute(rsqrtf(sum_squared / Tcompute(dim) + epsilon));
    }
    __syncthreads();

    // Apply normalization: normalized = x * w * rms
    for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        norm_ptr[i] = Tdata(Tcompute(x_ptr[i]) * Tcompute(w[i]) * rms);
    }
}

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata, typename Tweight>
INFINIOP_CUDA_KERNEL rmsnormKernel(
    Tdata *__restrict__ normalized,
    const Tdata *__restrict__ x,
    const Tweight *__restrict__ w,
    size_t ntok,
    size_t dim,
    float epsilon,
    ptrdiff_t in_stride,
    ptrdiff_t out_stride) {
    if (blockIdx.x < ntok) {
        rmsnormBlock<BLOCK_SIZE, Tcompute>(
            normalized, x, w, dim, epsilon, in_stride, out_stride);
    }
}

// SwiGLU transform kernel
// Computes: out[i] = silu(gate[i]) * up[i] where silu(x) = x * sigmoid(x)
// gate_up buffer has shape [ntok, 2*di], we transform in-place to [ntok, di]
template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata>
__device__ void swigluBlock(
    Tdata *__restrict__ out,
    const Tdata *__restrict__ gate_up,
    size_t intermediate_dim,
    ptrdiff_t stride) {

    size_t token_idx = blockIdx.x;
    auto gate_up_ptr = gate_up + token_idx * stride;
    auto out_ptr = out + token_idx * stride / 2; // Output stride is half

    for (size_t i = threadIdx.x; i < intermediate_dim; i += BLOCK_SIZE) {
        Tcompute gate = Tcompute(gate_up_ptr[i]);
        Tcompute up = Tcompute(gate_up_ptr[intermediate_dim + i]);

        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        Tcompute sigmoid = Tcompute(1.0f) / (Tcompute(1.0f) + exp_(-gate));
        Tcompute silu = gate * sigmoid;

        out_ptr[i] = Tdata(silu * up);
    }
}

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata>
INFINIOP_CUDA_KERNEL swigluKernel(
    Tdata *__restrict__ out,
    const Tdata *__restrict__ gate_up,
    size_t ntok,
    size_t intermediate_dim,
    ptrdiff_t stride) {
    if (blockIdx.x < ntok) {
        swigluBlock<BLOCK_SIZE, Tcompute>(
            out, gate_up, intermediate_dim, stride);
    }
}

// Residual add kernel
// Computes: out = gemm_out + residual
template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata>
__device__ void residualAddBlock(
    Tdata *__restrict__ out,
    const Tdata *__restrict__ gemm_out,
    const Tdata *__restrict__ residual,
    size_t dim,
    ptrdiff_t out_stride,
    ptrdiff_t residual_stride) {

    size_t token_idx = blockIdx.x;
    auto out_ptr = out + token_idx * out_stride;
    auto gemm_ptr = gemm_out + token_idx * out_stride;
    auto residual_ptr = residual + token_idx * residual_stride;

    for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        out_ptr[i] = Tdata(Tcompute(gemm_ptr[i]) + Tcompute(residual_ptr[i]));
    }
}

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata>
INFINIOP_CUDA_KERNEL residualAddKernel(
    Tdata *__restrict__ out,
    const Tdata *__restrict__ gemm_out,
    const Tdata *__restrict__ residual,
    size_t ntok,
    size_t dim,
    ptrdiff_t out_stride,
    ptrdiff_t residual_stride) {
    if (blockIdx.x < ntok) {
        residualAddBlock<BLOCK_SIZE, Tcompute>(
            out, gemm_out, residual, dim, out_stride, residual_stride);
    }
}

#endif // __FUSED_FFN_CUDA_KERNEL_H__
