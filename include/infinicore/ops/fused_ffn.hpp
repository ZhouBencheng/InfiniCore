#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

// Fused RMSNorm + Gate-Up GEMM + SwiGLU + Down GEMM (+ optional residual add).
//
// Computes:  out = down(swiglu(gate_up(rms_norm(in)))) [+ residual]
//
// Shapes:
//   in            : [..., hidden]
//   residual      : same as in (optional; when nullopt, no add is performed)
//   norm_weight   : [hidden]
//   gate_up_weight: [2 * intermediate, hidden]   (gate and up concatenated row-wise)
//   down_weight   : [hidden, intermediate]
//   out           : same shape/dtype as in
INFINICORE_GRAPH_OP_CLASS(FusedFFN, Tensor, const Tensor &,
                          std::optional<Tensor>,
                          const Tensor &, const Tensor &, const Tensor &,
                          float);

Tensor fused_ffn(const Tensor &in,
                 std::optional<Tensor> residual,
                 const Tensor &norm_weight,
                 const Tensor &gate_up_weight,
                 const Tensor &down_weight,
                 float epsilon);

void fused_ffn_(Tensor out,
                const Tensor &in,
                std::optional<Tensor> residual,
                const Tensor &norm_weight,
                const Tensor &gate_up_weight,
                const Tensor &down_weight,
                float epsilon);

} // namespace infinicore::op
