#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "grouped_gemm_nvidia.cuh"

namespace op::grouped_gemm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::Descriptor(
    int group_count,
    infiniDtype_t dtype,
    size_t workspace_size,
    Opaque *opaque,
    infiniDevice_t device_type,
    int device_id)
    : InfiniopDescriptor{device_type, device_id}
    , _opaque(opaque)
    , _group_count(group_count)
    , _dtype(dtype)
    , _workspace_size(workspace_size) {}

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    int group_count,
    const int *m_array, const int *n_array, const int *k_array,
    const int *lda_array, const int *ldb_array, const int *ldc_array,
    infiniDtype_t dtype) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

#ifdef ENABLE_NVIDIA_API
    size_t ptr_size = 3 * group_count * sizeof(void *);
    size_t int_size = 6 * group_count * sizeof(int);
    size_t workspace_size = ptr_size + int_size;
#else
    size_t workspace_size = 0;
#endif

    auto desc = new Descriptor(
        group_count, dtype, workspace_size,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);

    desc->_m_arr.assign(m_array, m_array + group_count);
    desc->_n_arr.assign(n_array, n_array + group_count);
    desc->_k_arr.assign(k_array, k_array + group_count);
    desc->_lda_arr.assign(lda_array, lda_array + group_count);
    desc->_ldb_arr.assign(ldb_array, ldb_array + group_count);
    desc->_ldc_arr.assign(ldc_array, ldc_array + group_count);

    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void * const *c_ptr_array,
    void const * const *a_ptr_array,
    void const * const *b_ptr_array,
    float alpha, float beta,
    void *stream) const {

    cudaStream_t cuda_stream = (cudaStream_t)stream;

    cudaDataType a_type, b_type, c_type;

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        a_type = b_type = c_type = CUDA_R_16F;
        break;
    case INFINI_DTYPE_BF16:
        a_type = b_type = c_type = CUDA_R_16BF;
        break;
    case INFINI_DTYPE_F32:
        a_type = b_type = c_type = CUDA_R_32F;
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#ifdef ENABLE_NVIDIA_API
    {
        cublasComputeType_t compute_type;
        switch (_dtype) {
        case INFINI_DTYPE_F16:
            compute_type = CUBLAS_COMPUTE_32F;
            break;
        case INFINI_DTYPE_BF16:
            compute_type = CUBLAS_COMPUTE_32F;
            break;
        case INFINI_DTYPE_F32:
            compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
            break;
        }

        void const **d_A_array = (void const **)workspace;
        void const **d_B_array = d_A_array + _group_count;
        void **d_C_array = (void **)(d_B_array + _group_count);

        int *d_m_array = (int *)(d_C_array + _group_count);
        int *d_n_array = d_m_array + _group_count;
        int *d_k_array = d_n_array + _group_count;
        int *d_lda_array = d_k_array + _group_count;
        int *d_ldb_array = d_lda_array + _group_count;
        int *d_ldc_array = d_ldb_array + _group_count;

        cudaMemcpyAsync(d_A_array, a_ptr_array, _group_count * sizeof(void *), cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync(d_B_array, b_ptr_array, _group_count * sizeof(void *), cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync(d_C_array, c_ptr_array, _group_count * sizeof(void *), cudaMemcpyHostToDevice, cuda_stream);

        cudaMemcpyAsync(d_m_array, _m_arr.data(), _group_count * sizeof(int), cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync(d_n_array, _n_arr.data(), _group_count * sizeof(int), cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync(d_k_array, _k_arr.data(), _group_count * sizeof(int), cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync(d_lda_array, _lda_arr.data(), _group_count * sizeof(int), cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync(d_ldb_array, _ldb_arr.data(), _group_count * sizeof(int), cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync(d_ldc_array, _ldc_arr.data(), _group_count * sizeof(int), cudaMemcpyHostToDevice, cuda_stream);

        CHECK_STATUS(_opaque->internal->useCublas(
            cuda_stream,
            [&](cublasHandle_t cb_handle) {
                CHECK_CUBLAS(
                    cublasGemmGroupedBatchedEx(
                        cb_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        d_m_array, d_n_array, d_k_array,
                        &alpha,
                        d_A_array, a_type, d_lda_array,
                        d_B_array, b_type, d_ldb_array,
                        &beta,
                        d_C_array, c_type, d_ldc_array,
                        _group_count,
                        compute_type,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                return INFINI_STATUS_SUCCESS;
            }));
    }
#else
    // Fallback for platforms without cublasGemmGroupedBatchedEx (Iluvatar, Hygon, Ali):
    // loop over each group and call cublasGemmEx individually
    for (int i = 0; i < _group_count; i++) {
        CHECK_STATUS(_opaque->internal->useCublas(
            cuda_stream,
            [&](cublasHandle_t cb_handle) {
                CHECK_CUBLAS(
                    cublasGemmEx(
                        cb_handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        _n_arr[i], _m_arr[i], _k_arr[i],
                        &alpha,
                        b_ptr_array[i], b_type, _ldb_arr[i],
                        a_ptr_array[i], a_type, _lda_arr[i],
                        &beta,
                        c_ptr_array[i], c_type, _ldc_arr[i],
                        CUDA_R_32F,
                        CUBLAS_GEMM_DFALT));
                return INFINI_STATUS_SUCCESS;
            }));
    }
#endif
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::grouped_gemm::nvidia
