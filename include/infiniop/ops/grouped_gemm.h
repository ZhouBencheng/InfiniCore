#ifndef __INFINIOP_GROUPED_GEMM_API_H__
#define __INFINIOP_GROUPED_GEMM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGroupedGemmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateGroupedGemmDescriptor(
    infiniopHandle_t handle,
    infiniopGroupedGemmDescriptor_t *desc_ptr,
    int group_count,
    const int *m_array, const int *n_array, const int *k_array,
    const int *lda_array, const int *ldb_array, const int *ldc_array,
    infiniDtype_t dtype);

__INFINI_C __export infiniStatus_t infiniopGetGroupedGemmWorkspaceSize(
    infiniopGroupedGemmDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopGroupedGemm(
    infiniopGroupedGemmDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void * const *c_ptr_array,
    void const * const *a_ptr_array,
    void const * const *b_ptr_array,
    float alpha, float beta,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyGroupedGemmDescriptor(infiniopGroupedGemmDescriptor_t desc);

#endif
