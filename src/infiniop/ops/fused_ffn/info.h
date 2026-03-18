#ifndef __FUSED_FFN_INFO_H__
#define __FUSED_FFN_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::fused_ffn {

class FusedFFNInfo {
    FusedFFNInfo() = default;

public:
    infiniDtype_t dtype;
    infiniDtype_t wtype;
    float epsilon;
    std::vector<size_t> shape;
    ptrdiff_t in_stride;
    ptrdiff_t out_stride;
    ptrdiff_t residual_stride;
    size_t hidden_dim;
    size_t intermediate_dim;
    bool has_residual;

    size_t ntok() const { return shape[0]; }
    size_t d() const { return hidden_dim; }
    size_t di() const { return intermediate_dim; }

    static utils::Result<FusedFFNInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc,
        infiniopTensorDescriptor_t residual_desc,
        infiniopTensorDescriptor_t norm_weight_desc,
        infiniopTensorDescriptor_t gate_up_weight_desc,
        infiniopTensorDescriptor_t down_weight_desc,
        float epsilon) {

        auto dtype = out_desc->dtype();
        auto wtype = norm_weight_desc->dtype();

        // Check that input and output have the same dtype
        if (in_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // Check weight dtypes
        if (gate_up_weight_desc->dtype() != wtype || down_weight_desc->dtype() != wtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // For half-precision types, weights can be same type or FP32
        if (dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16) {
            if (wtype != dtype && wtype != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else if (dtype == INFINI_DTYPE_F32) {
            if (wtype != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // Check tensor dimensions
        const size_t out_ndim = out_desc->ndim();
        const size_t in_ndim = in_desc->ndim();

        // Must be 2D tensors [ntok, hidden_dim]
        if (out_ndim != 2 || in_ndim != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t ntok = out_desc->dim(0);
        size_t hidden_dim = out_desc->dim(1);

        // Check input shape matches output
        if (in_desc->dim(0) != ntok || in_desc->dim(1) != hidden_dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check norm_weight is 1D [hidden_dim]
        if (norm_weight_desc->ndim() != 1 || norm_weight_desc->dim(0) != hidden_dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check gate_up_weight is 2D [2*intermediate_dim, hidden_dim]
        if (gate_up_weight_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        size_t gate_up_rows = gate_up_weight_desc->dim(0);
        size_t gate_up_cols = gate_up_weight_desc->dim(1);
        if (gate_up_cols != hidden_dim || gate_up_rows % 2 != 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        size_t intermediate_dim = gate_up_rows / 2;

        // Check down_weight is 2D [hidden_dim, intermediate_dim]
        if (down_weight_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (down_weight_desc->dim(0) != hidden_dim || down_weight_desc->dim(1) != intermediate_dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check contiguity of the last dimension
        if (out_desc->stride(out_ndim - 1) != 1 ||
            in_desc->stride(in_ndim - 1) != 1 ||
            norm_weight_desc->stride(0) != 1 ||
            gate_up_weight_desc->stride(1) != 1 ||
            down_weight_desc->stride(1) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        // Check residual if provided
        bool has_residual = residual_desc != nullptr;
        ptrdiff_t residual_stride = 0;
        if (has_residual) {
            if (residual_desc->ndim() != 2) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            if (residual_desc->dim(0) != ntok || residual_desc->dim(1) != hidden_dim) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            if (residual_desc->dtype() != dtype) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
            if (residual_desc->stride(1) != 1) {
                return INFINI_STATUS_BAD_TENSOR_STRIDES;
            }
            residual_stride = residual_desc->stride(0);
        }

        FusedFFNInfo info;
        info.dtype = dtype;
        info.wtype = wtype;
        info.epsilon = epsilon;
        info.shape = out_desc->shape();
        info.in_stride = in_desc->stride(0);
        info.out_stride = out_desc->stride(0);
        info.residual_stride = residual_stride;
        info.hidden_dim = hidden_dim;
        info.intermediate_dim = intermediate_dim;
        info.has_residual = has_residual;

        return utils::Result<FusedFFNInfo>(info);
    }
};

} // namespace op::fused_ffn

#endif // __FUSED_FFN_INFO_H__
