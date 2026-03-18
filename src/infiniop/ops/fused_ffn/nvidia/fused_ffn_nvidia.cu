#include "../../../devices/nvidia/nvidia_common.cuh"
#include "fused_ffn_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cublas_v2.h>

#include "kernel.cuh"

namespace op::fused_ffn::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    infiniopTensorDescriptor_t residual_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    infiniopTensorDescriptor_t gate_up_weight_desc,
    infiniopTensorDescriptor_t down_weight_desc,
    float epsilon) {

    auto result = FusedFFNInfo::create(
        out_desc, in_desc, residual_desc,
        norm_weight_desc, gate_up_weight_desc, down_weight_desc, epsilon);
    CHECK_RESULT(result);
    auto info = result.take();

    // Workspace size:
    // - normalized_buf: ntok * d * sizeof(dtype)  (for RMSNorm output -> GEMM input)
    // - gate_up_buf: ntok * 2*di * sizeof(dtype)  (for GateUp GEMM output)
    // - hidden_buf: ntok * di * sizeof(dtype)     (for SwiGLU output -> Down GEMM input)
    // - down_out_buf: ntok * d * sizeof(dtype)    (for Down GEMM output, if residual exists)
    size_t dtype_size = infiniSizeOf(info.dtype);
    size_t ntok = info.ntok();
    size_t d = info.d();
    size_t di = info.di();

    size_t normalized_size = ntok * d * dtype_size;
    size_t gate_up_size = ntok * 2 * di * dtype_size;
    size_t hidden_size = ntok * di * dtype_size;
    size_t down_out_size = info.has_residual ? ntok * d * dtype_size : 0;

    // Total workspace
    size_t workspace_size = normalized_size + gate_up_size + hidden_size + down_out_size;

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        workspace_size,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// Launch RMSNorm kernel with different data types
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchRmsnormKernel(
    size_t ntok, size_t dim,
    void *normalized, const void *x, const void *w,
    infiniDtype_t dtype, infiniDtype_t wtype,
    float epsilon,
    ptrdiff_t in_stride, ptrdiff_t out_stride,
    cudaStream_t cuda_stream) {

#define LAUNCH_KERNEL(Tdata, Tweight, Tcompute)                                        \
    rmsnormKernel<BLOCK_SIZE, Tcompute, Tdata, Tweight>                                \
        <<<ntok, BLOCK_SIZE, 0, cuda_stream>>>(                                        \
            reinterpret_cast<Tdata *>(normalized),                                     \
            reinterpret_cast<const Tdata *>(x),                                        \
            reinterpret_cast<const Tweight *>(w),                                      \
            ntok, dim, epsilon, in_stride, out_stride)

    if (dtype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_F16) {
        LAUNCH_KERNEL(half, half, float);
    } else if (dtype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_BF16) {
        LAUNCH_KERNEL(half, __nv_bfloat16, float);
    } else if (dtype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(half, float, float);
    } else if (dtype == INFINI_DTYPE_BF16 && wtype == INFINI_DTYPE_BF16) {
        LAUNCH_KERNEL(__nv_bfloat16, __nv_bfloat16, float);
    } else if (dtype == INFINI_DTYPE_BF16 && wtype == INFINI_DTYPE_F16) {
        LAUNCH_KERNEL(__nv_bfloat16, half, float);
    } else if (dtype == INFINI_DTYPE_BF16 && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(__nv_bfloat16, float, float);
    } else if (dtype == INFINI_DTYPE_F32 && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(float, float, float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_KERNEL

    return INFINI_STATUS_SUCCESS;
}

// Launch SwiGLU kernel with different data types
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchSwigluKernel(
    size_t ntok, size_t intermediate_dim,
    void *out, const void *gate_up,
    infiniDtype_t dtype,
    ptrdiff_t stride,
    cudaStream_t cuda_stream) {

#define LAUNCH_KERNEL(Tdata, Tcompute)                                                 \
    swigluKernel<BLOCK_SIZE, Tcompute, Tdata>                                          \
        <<<ntok, BLOCK_SIZE, 0, cuda_stream>>>(                                        \
            reinterpret_cast<Tdata *>(out),                                            \
            reinterpret_cast<const Tdata *>(gate_up),                                  \
            ntok, intermediate_dim, stride)

    if (dtype == INFINI_DTYPE_F16) {
        LAUNCH_KERNEL(half, float);
    } else if (dtype == INFINI_DTYPE_BF16) {
        LAUNCH_KERNEL(__nv_bfloat16, float);
    } else if (dtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(float, float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_KERNEL

    return INFINI_STATUS_SUCCESS;
}

// Launch Residual Add kernel with different data types
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchResidualAddKernel(
    size_t ntok, size_t dim,
    void *out, const void *gemm_out, const void *residual,
    infiniDtype_t dtype,
    ptrdiff_t out_stride, ptrdiff_t residual_stride,
    cudaStream_t cuda_stream) {

#define LAUNCH_KERNEL(Tdata, Tcompute)                                                 \
    residualAddKernel<BLOCK_SIZE, Tcompute, Tdata>                                     \
        <<<ntok, BLOCK_SIZE, 0, cuda_stream>>>(                                        \
            reinterpret_cast<Tdata *>(out),                                            \
            reinterpret_cast<const Tdata *>(gemm_out),                                 \
            reinterpret_cast<const Tdata *>(residual),                                 \
            ntok, dim, out_stride, residual_stride)

    if (dtype == INFINI_DTYPE_F16) {
        LAUNCH_KERNEL(half, float);
    } else if (dtype == INFINI_DTYPE_BF16) {
        LAUNCH_KERNEL(__nv_bfloat16, float);
    } else if (dtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(float, float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_KERNEL

    return INFINI_STATUS_SUCCESS;
}

// Helper function to get cuBLAS types for activation and weight matrices
static void getCublasTypes(infiniDtype_t dtype, infiniDtype_t wtype,
                          cudaDataType &a_type, cudaDataType &b_type, cudaDataType &c_type,
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
                          cudaDataType &compute_type
#else
                          cublasComputeType_t &compute_type
#endif
) {
    // Set a_type and c_type based on activation dtype
    switch (dtype) {
    case INFINI_DTYPE_F16:
        a_type = CUDA_R_16F;
        c_type = CUDA_R_16F;
        break;
    case INFINI_DTYPE_BF16:
        a_type = CUDA_R_16BF;
        c_type = CUDA_R_16BF;
        break;
    case INFINI_DTYPE_F32:
        a_type = CUDA_R_32F;
        c_type = CUDA_R_32F;
        break;
    default:
        break;
    }

    // Set b_type based on weight wtype
    switch (wtype) {
    case INFINI_DTYPE_F16:
        b_type = CUDA_R_16F;
        break;
    case INFINI_DTYPE_BF16:
        b_type = CUDA_R_16BF;
        break;
    case INFINI_DTYPE_F32:
        b_type = CUDA_R_32F;
        break;
    default:
        break;
    }

    // Set compute type based on activation dtype
    switch (dtype) {
    case INFINI_DTYPE_F16:
    case INFINI_DTYPE_BF16:
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_F32:
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#endif
        break;
    default:
        break;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out,
    const void *in,
    const void *residual,
    const void *norm_weight,
    const void *gate_up_weight,
    const void *down_weight,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    size_t ntok = _info.ntok();
    size_t d = _info.d();
    size_t di = _info.di();
    size_t dtype_size = infiniSizeOf(_info.dtype);

    // Partition workspace
    char *ws_ptr = static_cast<char *>(workspace);
    void *normalized_buf = ws_ptr;
    ws_ptr += ntok * d * dtype_size;
    void *gate_up_buf = ws_ptr;
    ws_ptr += ntok * 2 * di * dtype_size;
    void *hidden_buf = ws_ptr;
    ws_ptr += ntok * di * dtype_size;
    void *down_out_buf = _info.has_residual ? ws_ptr : out;

    // Stage 1: RMSNorm
    if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchRmsnormKernel<1024>(
            ntok, d,
            normalized_buf, in, norm_weight,
            _info.dtype, _info.wtype,
            _info.epsilon,
            _info.in_stride, d,
            cuda_stream));
    } else {
        CHECK_STATUS(launchRmsnormKernel<512>(
            ntok, d,
            normalized_buf, in, norm_weight,
            _info.dtype, _info.wtype,
            _info.epsilon,
            _info.in_stride, d,
            cuda_stream));
    }

    // Stage 2: GateUp GEMM using cuBLAS
    // gate_up = normalized @ gate_up_weight^T
    // normalized: [ntok, d], gate_up_weight: [2*di, d] -> gate_up: [ntok, 2*di]
    {
        cudaDataType a_type, b_type, c_type;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        cudaDataType compute_type;
#else
        cublasComputeType_t compute_type;
#endif
        getCublasTypes(_info.dtype, _info.wtype, a_type, b_type, c_type, compute_type);

        float alpha = 1.0f, beta = 0.0f;
        CHECK_STATUS(_opaque->internal->useCublas(
            cuda_stream,
            [&](cublasHandle_t handle) {
                // For row-major C = A @ B^T with cuBLAS:
                // Row-major A[ntok,d], B[2*di,d], C[ntok,2*di]
                // -> Column-major A_col[d,ntok], B_col[d,2*di], C_col[2*di,ntok]
                // C_col = B_col^T @ A_col^T
                // Use transa=T, transb=N, m=2*di, n=ntok, k=d
                CHECK_CUBLAS(
                    cublasGemmStridedBatchedEx(
                        handle,
                        CUBLAS_OP_T,  // A (weight): transpose
                        CUBLAS_OP_N,  // B (normalized): no transpose
                        static_cast<int>(2 * di),  // m
                        static_cast<int>(ntok),     // n
                        static_cast<int>(d),        // k
                        &alpha,
                        gate_up_weight, b_type, static_cast<int>(d), 0,
                        normalized_buf, a_type, static_cast<int>(d), 0,
                        &beta,
                        gate_up_buf, c_type, static_cast<int>(2 * di), 0,
                        1,  // batch count
                        compute_type,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                return INFINI_STATUS_SUCCESS;
            }));
    }

    // Stage 3: SwiGLU transform
    if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchSwigluKernel<1024>(
            ntok, di,
            hidden_buf, gate_up_buf,
            _info.dtype,
            2 * di,
            cuda_stream));
    } else {
        CHECK_STATUS(launchSwigluKernel<512>(
            ntok, di,
            hidden_buf, gate_up_buf,
            _info.dtype,
            2 * di,
            cuda_stream));
    }

    // Stage 4: Down GEMM using cuBLAS
    // down_out = hidden @ down_weight^T
    // hidden: [ntok, di], down_weight: [d, di] -> down_out: [ntok, d]
    {
        cudaDataType a_type, b_type, c_type;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        cudaDataType compute_type;
#else
        cublasComputeType_t compute_type;
#endif
        getCublasTypes(_info.dtype, _info.wtype, a_type, b_type, c_type, compute_type);

        float alpha = 1.0f, beta = 0.0f;
        CHECK_STATUS(_opaque->internal->useCublas(
            cuda_stream,
            [&](cublasHandle_t handle) {
                // For row-major C = A @ B^T with cuBLAS:
                // Row-major A[ntok,di], B[d,di], C[ntok,d]
                // -> Column-major A_col[di,ntok], B_col[di,d], C_col[d,ntok]
                // C_col = B_col^T @ A_col^T
                // Use transa=T, transb=N, m=d, n=ntok, k=di
                CHECK_CUBLAS(
                    cublasGemmStridedBatchedEx(
                        handle,
                        CUBLAS_OP_T,  // A (weight): transpose
                        CUBLAS_OP_N,  // B (hidden): no transpose
                        static_cast<int>(d),     // m
                        static_cast<int>(ntok),  // n
                        static_cast<int>(di),    // k
                        &alpha,
                        down_weight, b_type, static_cast<int>(di), 0,
                        hidden_buf, a_type, static_cast<int>(di), 0,
                        &beta,
                        down_out_buf, c_type, static_cast<int>(d), 0,
                        1,  // batch count
                        compute_type,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                return INFINI_STATUS_SUCCESS;
            }));
    }

    // Stage 5: Residual Add (if residual is provided)
    if (_info.has_residual) {
        if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_1024) {
            CHECK_STATUS(launchResidualAddKernel<1024>(
                ntok, d,
                out, down_out_buf, residual,
                _info.dtype,
                _info.out_stride, _info.residual_stride,
                cuda_stream));
        } else {
            CHECK_STATUS(launchResidualAddKernel<512>(
                ntok, d,
                out, down_out_buf, residual,
                _info.dtype,
                _info.out_stride, _info.residual_stride,
                cuda_stream));
        }
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::fused_ffn::nvidia
