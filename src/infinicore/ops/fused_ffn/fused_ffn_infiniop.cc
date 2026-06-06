#include "infinicore/ops/fused_ffn.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::fused_ffn_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, FusedFFN, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, out, in, norm_weight, gate_up_weight, down_weight;
    std::optional<graph::GraphTensor> residual;
};

void *plan(Tensor out, const Tensor &in,
           std::optional<Tensor> residual,
           const Tensor &norm_weight,
           const Tensor &gate_up_weight,
           const Tensor &down_weight,
           float epsilon) {
    size_t seed = hash_combine(out, in, residual,
                               norm_weight, gate_up_weight, down_weight,
                               epsilon);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, FusedFFN,
        seed,
        out->desc(),
        in->desc(),
        residual.has_value() ? residual.value()->desc() : nullptr,
        norm_weight->desc(),
        gate_up_weight->desc(),
        down_weight->desc(),
        epsilon);

    INFINIOP_WORKSPACE_TENSOR(workspace, FusedFFN, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(in),
        graph::GraphTensor(norm_weight),
        graph::GraphTensor(gate_up_weight),
        graph::GraphTensor(down_weight),
        residual.has_value()
            ? std::optional<graph::GraphTensor>(graph::GraphTensor(*residual))
            : std::nullopt};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(
        infiniopFusedFFN(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->out->data(),
            p->in->data(),
            p->residual.has_value() ? p->residual.value()->data() : nullptr,
            p->norm_weight->data(),
            p->gate_up_weight->data(),
            p->down_weight->data(),
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(FusedFFN, &plan, &run, &cleanup);

} // namespace infinicore::op::fused_ffn_impl::infiniop
