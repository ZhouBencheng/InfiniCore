#ifndef FUSED_FFN_H
#define FUSED_FFN_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::fused_ffn::NAMESPACE {                         \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        FusedFFNInfo _info;                                      \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            FusedFFNInfo info,                                   \
            size_t workspace_size,                               \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t out_desc,                 \
            infiniopTensorDescriptor_t in_desc,                  \
            infiniopTensorDescriptor_t residual_desc,            \
            infiniopTensorDescriptor_t norm_weight_desc,         \
            infiniopTensorDescriptor_t gate_up_weight_desc,      \
            infiniopTensorDescriptor_t down_weight_desc,         \
            float epsilon);                                      \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *out,                                           \
            const void *in,                                      \
            const void *residual,                                \
            const void *norm_weight,                             \
            const void *gate_up_weight,                          \
            const void *down_weight,                             \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // FUSED_FFN_H
