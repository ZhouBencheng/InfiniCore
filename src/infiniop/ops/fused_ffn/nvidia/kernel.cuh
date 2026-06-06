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

// SwiGLU transform kernel (in-place)
// Computes: gate_up[i] = silu(gate_up[i]) * gate_up[di+i] for i in [0, di)
// Writes result to the gate half of gate_up, overwriting gate values.
// This matches the non-fused path's buffer layout (hidden at stride 2*di).
//
// Used by the metax backend's fused_ffn path. The nvidia backend uses the
// out-of-place variant below (`swigluOutKernel`) to write into a separate
// hidden_buf with stride di — saves the down-GEMM from having to read a
// stride-2*di operand.
template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata>
__device__ void swigluBlock(
    Tdata *__restrict__ gate_up,
    size_t intermediate_dim,
    ptrdiff_t stride) {

    size_t token_idx = blockIdx.x;
    auto gate_up_ptr = gate_up + token_idx * stride;

    for (size_t i = threadIdx.x; i < intermediate_dim; i += BLOCK_SIZE) {
        Tcompute gate = Tcompute(gate_up_ptr[i]);
        Tcompute up = Tcompute(gate_up_ptr[intermediate_dim + i]);

        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        Tcompute sigmoid = Tcompute(1.0f) / (Tcompute(1.0f) + exp_(-gate));
        Tcompute silu = gate * sigmoid;

        gate_up_ptr[i] = Tdata(silu * up);
    }
}

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata>
INFINIOP_CUDA_KERNEL swigluKernel(
    Tdata *__restrict__ gate_up,
    size_t ntok,
    size_t intermediate_dim,
    ptrdiff_t stride) {
    if (blockIdx.x < ntok) {
        swigluBlock<BLOCK_SIZE, Tcompute>(
            gate_up, intermediate_dim, stride);
    }
}

// SwiGLU transform kernel (out-of-place)
// Computes: hidden[token, j] = silu(gate_up[token, j]) * gate_up[token, di + j]
// Reads gate (first half) and up (second half) from a row of gate_up_buf with
// stride `gate_up_row_stride`; writes silu(gate)*up to hidden_buf with stride
// `hidden_row_stride`. Caller guarantees the gate_up layout is [gate|up].
//
// This is what the nvidia FusedFFN calculate() launches directly to skip the
// standalone op::swiglu::nvidia::Descriptor's elementwise-framework overhead.
template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata>
__device__ void swigluOutBlock(
    Tdata *__restrict__ hidden,
    const Tdata *__restrict__ gate_up,
    size_t intermediate_dim,
    ptrdiff_t hidden_row_stride,
    ptrdiff_t gate_up_row_stride) {

    size_t token_idx = blockIdx.x;
    auto h_ptr = hidden + token_idx * hidden_row_stride;
    auto gu_ptr = gate_up + token_idx * gate_up_row_stride;

    for (size_t i = threadIdx.x; i < intermediate_dim; i += BLOCK_SIZE) {
        Tcompute gate = Tcompute(gu_ptr[i]);
        Tcompute up = Tcompute(gu_ptr[intermediate_dim + i]);

        Tcompute sigmoid = Tcompute(1.0f) / (Tcompute(1.0f) + exp_(-gate));
        Tcompute silu = gate * sigmoid;

        h_ptr[i] = Tdata(silu * up);
    }
}

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata>
INFINIOP_CUDA_KERNEL swigluOutKernel(
    Tdata *__restrict__ hidden,
    const Tdata *__restrict__ gate_up,
    size_t ntok,
    size_t intermediate_dim,
    ptrdiff_t hidden_row_stride,
    ptrdiff_t gate_up_row_stride) {
    if (blockIdx.x < ntok) {
        swigluOutBlock<BLOCK_SIZE, Tcompute>(
            hidden, gate_up, intermediate_dim,
            hidden_row_stride, gate_up_row_stride);
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

// ── Deep-fused Gate-Up GEMM + SwiGLU ────────────────────────────────
//
// Reproduces the paper-style fusion of DeepFusionKernel (Zhang et al., 2026)
// by streaming both matmul tiles and the SiLU-gated mul through registers,
// eliminating the gate_up_buf HBM round-trip that sits between the separate
// gate_up_gemm and swiglu kernels in the non-fused pipeline.
//
// For each output element hidden[token, col]:
//     gate = Σ_k X[token, k] * W_gate[k, col]
//     up   = Σ_k X[token, k] * W_up  [k, col]
//     hidden[token, col] = (gate * sigmoid(gate)) * up
//
// Grid : (ntok, di)
// Block: BLOCK_SIZE threads cooperatively reducing over K=d.
//
// The weight pointer is the raw user pointer. w_k_stride and w_col_stride are
// the physical element strides when moving along K and along the output
// column axis respectively, so both [2*di, d] (layout A) and [d, 2*di]
// (layout B) storages are supported by choosing (stride_k, stride_col).
template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata, typename Tweight>
__device__ void deepFusedGateUpSiluBlock(
    Tdata *__restrict__ hidden,
    const Tdata *__restrict__ x,
    const Tweight *__restrict__ w,
    size_t d,
    ptrdiff_t x_row_stride,
    ptrdiff_t hidden_row_stride,
    ptrdiff_t w_k_stride,
    ptrdiff_t w_col_stride,
    size_t gate_col_base,
    size_t up_col_base) {

    const size_t token = blockIdx.x;
    const size_t col = blockIdx.y;

    auto x_ptr = x + token * x_row_stride;
    auto h_ptr = hidden + token * hidden_row_stride;

    const Tweight *w_gate_col = w + (gate_col_base + col) * w_col_stride;
    const Tweight *w_up_col = w + (up_col_base + col) * w_col_stride;

    Tcompute gate_acc = Tcompute(0);
    Tcompute up_acc = Tcompute(0);

    for (size_t k = threadIdx.x; k < d; k += BLOCK_SIZE) {
        Tcompute x_val = Tcompute(x_ptr[k]);
        Tcompute w_g = Tcompute(w_gate_col[k * w_k_stride]);
        Tcompute w_u = Tcompute(w_up_col[k * w_k_stride]);
        gate_acc += x_val * w_g;
        up_acc += x_val * w_u;
    }

    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    Tcompute gate_sum = BlockReduce(temp_storage).Sum(gate_acc);
    __syncthreads();
    Tcompute up_sum = BlockReduce(temp_storage).Sum(up_acc);

    if (threadIdx.x == 0) {
        Tcompute sig = Tcompute(1.0f) / (Tcompute(1.0f) + exp_(-gate_sum));
        Tcompute silu = gate_sum * sig;
        h_ptr[col] = Tdata(silu * up_sum);
    }
}

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata, typename Tweight>
INFINIOP_CUDA_KERNEL deepFusedGateUpSiluKernel(
    Tdata *__restrict__ hidden,
    const Tdata *__restrict__ x,
    const Tweight *__restrict__ w,
    size_t ntok,
    size_t d,
    size_t di,
    ptrdiff_t x_row_stride,
    ptrdiff_t hidden_row_stride,
    ptrdiff_t w_k_stride,
    ptrdiff_t w_col_stride,
    size_t gate_col_base,
    size_t up_col_base) {
    if (blockIdx.x < ntok && blockIdx.y < di) {
        deepFusedGateUpSiluBlock<BLOCK_SIZE, Tcompute>(
            hidden, x, w, d,
            x_row_stride, hidden_row_stride,
            w_k_stride, w_col_stride,
            gate_col_base, up_col_base);
    }
}

#endif // __FUSED_FFN_CUDA_KERNEL_H__
