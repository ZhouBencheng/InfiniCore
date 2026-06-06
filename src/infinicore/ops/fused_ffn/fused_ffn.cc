#include "infinicore/ops/fused_ffn.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FusedFFN);

FusedFFN::FusedFFN(Tensor out, const Tensor &in,
                   std::optional<Tensor> residual,
                   const Tensor &norm_weight,
                   const Tensor &gate_up_weight,
                   const Tensor &down_weight,
                   float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, in, norm_weight, gate_up_weight, down_weight);
    if (residual.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, residual.value());
    }
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, in, residual,
                                 norm_weight, gate_up_weight, down_weight,
                                 epsilon);
}

void FusedFFN::execute(Tensor out, const Tensor &in,
                       std::optional<Tensor> residual,
                       const Tensor &norm_weight,
                       const Tensor &gate_up_weight,
                       const Tensor &down_weight,
                       float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        FusedFFN,
        out, in, residual, norm_weight, gate_up_weight, down_weight, epsilon);
}

Tensor fused_ffn(const Tensor &in,
                 std::optional<Tensor> residual,
                 const Tensor &norm_weight,
                 const Tensor &gate_up_weight,
                 const Tensor &down_weight,
                 float epsilon) {
    auto out = Tensor::empty(in->shape(), in->dtype(), in->device());
    fused_ffn_(out, in, residual, norm_weight, gate_up_weight, down_weight, epsilon);
    return out;
}

void fused_ffn_(Tensor out,
                const Tensor &in,
                std::optional<Tensor> residual,
                const Tensor &norm_weight,
                const Tensor &gate_up_weight,
                const Tensor &down_weight,
                float epsilon) {
    FusedFFN::execute(out, in, residual, norm_weight, gate_up_weight, down_weight, epsilon);
}

} // namespace infinicore::op
