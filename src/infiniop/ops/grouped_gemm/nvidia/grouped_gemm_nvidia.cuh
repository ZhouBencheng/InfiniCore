#ifndef __GROUPED_GEMM_NVIDIA_CUH__
#define __GROUPED_GEMM_NVIDIA_CUH__

#include "../../../operator.h"
#include <vector>

namespace op::grouped_gemm::nvidia {

class Descriptor final : public InfiniopDescriptor {
    struct Opaque;
    Opaque *_opaque;

    int _group_count;
    infiniDtype_t _dtype;
    size_t _workspace_size;

    std::vector<int> _m_arr, _n_arr, _k_arr;
    std::vector<int> _lda_arr, _ldb_arr, _ldc_arr;

    Descriptor(
        int group_count,
        infiniDtype_t dtype,
        size_t workspace_size,
        Opaque *opaque,
        infiniDevice_t device_type,
        int device_id);

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        int group_count,
        const int *m_array, const int *n_array, const int *k_array,
        const int *lda_array, const int *ldb_array, const int *ldc_array,
        infiniDtype_t dtype);

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void * const *c_ptr_array,
        void const * const *a_ptr_array,
        void const * const *b_ptr_array,
        float alpha, float beta,
        void *stream) const;
};

} // namespace op::grouped_gemm::nvidia

#endif
