#include "fused_ffn_cpu.h"
#include "../../../../utils.h"
#include "../../../devices/cpu/common_cpu.h"
#include <cmath>
#include <cstring>

namespace op::fused_ffn::cpu {

Descriptor::~Descriptor() = default;

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

    // Workspace size (same as NVIDIA implementation)
    size_t dtype_size = infiniSizeOf(info.dtype);
    size_t ntok = info.ntok();
    size_t d = info.d();
    size_t di = info.di();

    size_t normalized_size = ntok * d * dtype_size;
    size_t gate_up_size = ntok * 2 * di * dtype_size;
    size_t hidden_size = ntok * di * dtype_size;
    size_t down_out_size = info.has_residual ? ntok * d * dtype_size : 0;

    size_t workspace_size = normalized_size + gate_up_size + hidden_size + down_out_size;

    *desc_ptr = new Descriptor(
        nullptr,
        std::move(info),
        workspace_size,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tweight>
infiniStatus_t calculateTyped(
    const FusedFFNInfo &info,
    void *workspace, size_t workspace_size,
    void *out,
    const void *in,
    const void *residual,
    const void *norm_weight,
    const void *gate_up_weight,
    const void *down_weight) {

    size_t ntok = info.ntok();
    size_t d = info.d();
    size_t di = info.di();

    // Partition workspace
    char *ws_ptr = static_cast<char *>(workspace);
    Tdata *normalized_buf = reinterpret_cast<Tdata *>(ws_ptr);
    ws_ptr += ntok * d * sizeof(Tdata);
    Tdata *gate_up_buf = reinterpret_cast<Tdata *>(ws_ptr);
    ws_ptr += ntok * 2 * di * sizeof(Tdata);
    Tdata *hidden_buf = reinterpret_cast<Tdata *>(ws_ptr);
    ws_ptr += ntok * di * sizeof(Tdata);
    Tdata *down_out_buf = info.has_residual ? reinterpret_cast<Tdata *>(ws_ptr) : reinterpret_cast<Tdata *>(out);

    const Tdata *in_ptr = reinterpret_cast<const Tdata *>(in);
    const Tdata *residual_ptr = reinterpret_cast<const Tdata *>(residual);
    const Tweight *norm_w_ptr = reinterpret_cast<const Tweight *>(norm_weight);
    const Tweight *gate_up_w_ptr = reinterpret_cast<const Tweight *>(gate_up_weight);
    const Tweight *down_w_ptr = reinterpret_cast<const Tweight *>(down_weight);
    Tdata *out_ptr = reinterpret_cast<Tdata *>(out);

    // Stage 1: RMSNorm
    for (size_t t = 0; t < ntok; t++) {
        const Tdata *x = in_ptr + t * info.in_stride;
        Tdata *norm = normalized_buf + t * d;

        // Compute variance
        float sum_sq = 0.0f;
        for (size_t i = 0; i < d; i++) {
            float val = utils::cast<float>(x[i]);
            sum_sq += val * val;
        }

        // Normalize
        float rms = 1.0f / std::sqrt(sum_sq / d + info.epsilon);
        for (size_t i = 0; i < d; i++) {
            float val = utils::cast<float>(x[i]) * utils::cast<float>(norm_w_ptr[i]) * rms;
            norm[i] = utils::cast<Tdata>(val);
        }
    }

    // Stage 2: GateUp GEMM (C = A @ B^T)
    // normalized: [ntok, d], gate_up_weight: [2*di, d] -> gate_up: [ntok, 2*di]
    for (size_t t = 0; t < ntok; t++) {
        const Tdata *norm = normalized_buf + t * d;
        Tdata *gate_up = gate_up_buf + t * 2 * di;

        for (size_t j = 0; j < 2 * di; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < d; k++) {
                sum += utils::cast<float>(norm[k]) * utils::cast<float>(gate_up_w_ptr[j * d + k]);
            }
            gate_up[j] = utils::cast<Tdata>(sum);
        }
    }

    // Stage 3: SwiGLU
    for (size_t t = 0; t < ntok; t++) {
        const Tdata *gate_up = gate_up_buf + t * 2 * di;
        Tdata *hidden = hidden_buf + t * di;

        for (size_t i = 0; i < di; i++) {
            float gate = utils::cast<float>(gate_up[i]);
            float up = utils::cast<float>(gate_up[di + i]);
            // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            float silu = gate / (1.0f + std::exp(-gate));
            hidden[i] = utils::cast<Tdata>(silu * up);
        }
    }

    // Stage 4: Down GEMM (C = A @ B^T)
    // hidden: [ntok, di], down_weight: [d, di] -> down_out: [ntok, d]
    for (size_t t = 0; t < ntok; t++) {
        const Tdata *hidden = hidden_buf + t * di;
        Tdata *down_out = down_out_buf + t * d;

        for (size_t j = 0; j < d; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < di; k++) {
                sum += utils::cast<float>(hidden[k]) * utils::cast<float>(down_w_ptr[j * di + k]);
            }
            down_out[j] = utils::cast<Tdata>(sum);
        }
    }

    // Stage 5: Residual Add (if residual is provided)
    if (info.has_residual) {
        for (size_t t = 0; t < ntok; t++) {
            const Tdata *down_out = down_out_buf + t * d;
            const Tdata *res = residual_ptr + t * info.residual_stride;
            Tdata *o = out_ptr + t * info.out_stride;

            for (size_t i = 0; i < d; i++) {
                float val = utils::cast<float>(down_out[i]) + utils::cast<float>(res[i]);
                o[i] = utils::cast<Tdata>(val);
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
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

    // Dispatch based on dtype and wtype
    if (_info.dtype == INFINI_DTYPE_F16) {
        if (_info.wtype == INFINI_DTYPE_F16) {
            return calculateTyped<fp16_t, fp16_t>(_info, workspace, workspace_size, out, in, residual, norm_weight, gate_up_weight, down_weight);
        } else if (_info.wtype == INFINI_DTYPE_F32) {
            return calculateTyped<fp16_t, float>(_info, workspace, workspace_size, out, in, residual, norm_weight, gate_up_weight, down_weight);
        }
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        if (_info.wtype == INFINI_DTYPE_BF16) {
            return calculateTyped<bf16_t, bf16_t>(_info, workspace, workspace_size, out, in, residual, norm_weight, gate_up_weight, down_weight);
        } else if (_info.wtype == INFINI_DTYPE_F32) {
            return calculateTyped<bf16_t, float>(_info, workspace, workspace_size, out, in, residual, norm_weight, gate_up_weight, down_weight);
        }
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        return calculateTyped<float, float>(_info, workspace, workspace_size, out, in, residual, norm_weight, gate_up_weight, down_weight);
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::fused_ffn::cpu
