#include "../../handle.h"
#include "infiniop/ops/grouped_gemm.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/grouped_gemm_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateGroupedGemmDescriptor(
    infiniopHandle_t handle,
    infiniopGroupedGemmDescriptor_t *desc_ptr,
    int group_count,
    const int *m_array, const int *n_array, const int *k_array,
    const int *lda_array, const int *ldb_array, const int *ldc_array,
    infiniDtype_t dtype) {

#define CREATE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        return op::grouped_gemm::NAMESPACE::Descriptor::create(                          \
            handle,                                                                      \
            reinterpret_cast<op::grouped_gemm::NAMESPACE::Descriptor **>(desc_ptr),      \
            group_count,                                                                 \
            m_array, n_array, k_array,                                                   \
            lda_array, ldb_array, ldc_array,                                             \
            dtype)

    switch (handle->device) {

#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetGroupedGemmWorkspaceSize(
    infiniopGroupedGemmDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                          \
    case CASE:                                                                                        \
        *size = reinterpret_cast<const op::grouped_gemm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__INFINI_C infiniStatus_t infiniopGroupedGemm(
    infiniopGroupedGemmDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void * const *c_ptr_array,
    void const * const *a_ptr_array,
    void const * const *b_ptr_array,
    float alpha, float beta,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                           \
        return reinterpret_cast<const op::grouped_gemm::NAMESPACE::Descriptor *>(desc)   \
            ->calculate(workspace, workspace_size,                                       \
                        c_ptr_array, a_ptr_array, b_ptr_array,                           \
                        alpha, beta,                                                     \
                        stream)

    switch (desc->device_type) {

#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyGroupedGemmDescriptor(
    infiniopGroupedGemmDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        delete reinterpret_cast<const op::grouped_gemm::NAMESPACE::Descriptor *>(desc);  \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DELETE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DELETE(INFINI_DEVICE_HYGON, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
