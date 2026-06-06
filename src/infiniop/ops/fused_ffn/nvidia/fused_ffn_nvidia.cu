// ────────────────────────────────────────────────────────────────────────────
// FusedFFN — nvidia / iluvatar BI-V150 implementation
// ────────────────────────────────────────────────────────────────────────────
//
// Five sub-descriptors used to be wired up here: RMSNorm + Gemm +
// SwiGLU + Gemm + Add. Each elementwise sub-descriptor's calculate()
// re-queries workspace size, re-walks tensor descriptors, does a dtype
// switch, and finally launches its kernel. For sub-millisecond ops on
// BI-V150 that host-side path costs ≈ 5-15 µs per stage, and three of
// them stacked together turned what should have been a launch-count
// saving (1 fused call vs 4 unfused calls) into a per-op regression vs
// the externally-chained unfused path.
//
// We now keep only the two cuBLAS GEMM sub-descriptors (cuBLAS is the
// floor on this hardware's tensor-core path) and launch RMSNorm,
// SwiGLU and the residual Add directly from calculate() using the
// templates in kernel.cuh. Launch params (block size, grid, strides)
// resolve from FusedFFNInfo at create() time; per-call host work is one
// dtype switch plus the kernel launch. The elementwise kernels are the
// same templates the standalone ops would have launched, so output is
// bit-identical to the old path.
// ────────────────────────────────────────────────────────────────────────────

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "fused_ffn_nvidia.cuh"
#include "kernel.cuh"

#undef DESCRIPTOR
#include "../../gemm/nvidia/gemm_nvidia.cuh"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>

namespace op::fused_ffn::nvidia {

namespace {

// 256-byte alignment is safe for both cublas working buffers and elementwise
// meta blobs; higher would waste workspace on small shapes.
constexpr size_t kWsAlign = 256;

inline size_t alignUp(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

// Heap-allocate a 2-D tensor descriptor with explicit element strides.
// Returned pointer must be deleted by the caller.
inline infiniopTensorDescriptor_t make2D(infiniDtype_t dtype,
                                         size_t d0, size_t d1,
                                         ptrdiff_t s0, ptrdiff_t s1) {
    const size_t shape[2] = {d0, d1};
    const ptrdiff_t strides[2] = {s0, s1};
    return new InfiniopTensorDescriptor(dtype, 2, shape, strides);
}

// Synthesize a GEMM B-matrix view with logical shape [k, n] regardless of
// whether the original weight was stored as [n, k] (Layout A) or [k, n]
// (Layout B). FusedFFNInfo::create already guarantees one of the two strides
// is 1, so this only needs to decide which dim is which and swap accordingly.
inline infiniopTensorDescriptor_t makeWeightAsKN(infiniDtype_t dtype,
                                                 size_t k, size_t n,
                                                 infiniopTensorDescriptor_t orig) {
    const size_t d0 = orig->dim(0);
    const ptrdiff_t s0 = orig->stride(0);
    const ptrdiff_t s1 = orig->stride(1);
    if (d0 == n) {
        // original [n, k] -> view as [k, n] by swapping axes
        return make2D(dtype, k, n, s1, s0);
    }
    // original already [k, n]
    return make2D(dtype, k, n, s0, s1);
}

// RAII wrapper: owns a list of synthesized tensor descriptors and deletes
// them on scope exit. Sub-descriptor create() calls copy out what they need,
// so the temporaries only need to outlive the create() call.
class DescScope {
    std::vector<infiniopTensorDescriptor_t> _owned;

public:
    ~DescScope() {
        for (auto *t : _owned) {
            delete t;
        }
    }
    infiniopTensorDescriptor_t adopt(infiniopTensorDescriptor_t t) {
        _owned.push_back(t);
        return t;
    }
};

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;

    // Workspace slab sizes (bytes), padded to kWsAlign.
    size_t normalized_bytes = 0;
    size_t gate_up_bytes = 0;
    size_t hidden_bytes = 0;
    size_t inner_ws_bytes = 0; // max of sub-descriptor workspaceSize()

    bool has_residual = false;

    // Deep-fused Gate-Up + SwiGLU config. Resolved from INFINIOP_FUSED_FFN_DEEP
    // at create() — see the env-var dispatch block in Descriptor::create() for
    // the full mode/threshold semantics.
    bool use_deep_fused = false;
    ptrdiff_t gate_up_w_k_stride = 0;
    ptrdiff_t gate_up_w_col_stride = 0;

    // GEMM sub-descriptors stay (cuBLAS path). RMSNorm / SwiGLU / Add are
    // launched directly via the kernel.cuh templates in calculate() so they
    // skip the per-call workspace-size query + virtual-call + dtype-switch
    // overhead the standalone sub-descriptors would otherwise pay.
    std::unique_ptr<op::gemm::nvidia::Descriptor> gate_up_gemm;
    std::unique_ptr<op::gemm::nvidia::Descriptor> down_gemm;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    infiniopTensorDescriptor_t residual_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    infiniopTensorDescriptor_t gate_up_weight_desc,
    infiniopTensorDescriptor_t down_weight_desc,
    float epsilon) {

    auto info_result = FusedFFNInfo::create(
        out_desc, in_desc, residual_desc,
        norm_weight_desc, gate_up_weight_desc, down_weight_desc, epsilon);
    CHECK_RESULT(info_result);
    auto info = info_result.take();

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto opaque = std::make_unique<Opaque>();
    opaque->internal = handle->internal();
    opaque->has_residual = info.has_residual;

    const size_t ntok = info.ntok();
    const size_t d = info.d();
    const size_t di = info.di();
    const size_t dtype_sz = infiniSizeOf(info.dtype);

    // Profile-driven scheduler for the deep-fused kernel path.
    //   INFINIOP_FUSED_FFN_DEEP=0/unset -> always 5-stage (default)
    //   INFINIOP_FUSED_FFN_DEEP=1       -> scheduler: use deep-fused only
    //       when ntok <= max_ntok threshold (default 4); fall back to
    //       5-stage for larger shapes where cuBLAS tensor-core GEMM wins.
    //   INFINIOP_FUSED_FFN_DEEP=2       -> force deep-fused always (debug)
    // Threshold is tunable via INFINIOP_FUSED_FFN_DEEP_MAX_NTOK (default 4).
    {
        const char *env = std::getenv("INFINIOP_FUSED_FFN_DEEP");
        if (env != nullptr && env[0] == '2') {
            // Mode 2: force deep-fused regardless of shape
            opaque->use_deep_fused = true;
        } else if (env != nullptr && env[0] == '1') {
            // Mode 1: scheduler — deep-fused only for small ntok
            size_t max_ntok = 4;
            const char *thr = std::getenv("INFINIOP_FUSED_FFN_DEEP_MAX_NTOK");
            if (thr != nullptr) {
                max_ntok = static_cast<size_t>(std::atol(thr));
                if (max_ntok == 0) {
                    max_ntok = 4;
                }
            }
            opaque->use_deep_fused = (ntok <= max_ntok);
        }
        // Mode 0 / unset: use_deep_fused stays false (5-stage)
    }

    // Extract gate_up weight strides at create time so the deep-fused kernel
    // can index [k, j] regardless of whether storage is Layout A [2*di, d]
    // or Layout B [d, 2*di]. Layout is identified by which descriptor dim
    // equals 2*di (the output column axis) vs d (the K axis).
    {
        const size_t gu_dim0 = gate_up_weight_desc->dim(0);
        const ptrdiff_t gu_s0 = gate_up_weight_desc->stride(0);
        const ptrdiff_t gu_s1 = gate_up_weight_desc->stride(1);
        if (gu_dim0 == 2 * di) {
            // Layout A: [2*di, d] — output column is dim0, K is dim1.
            opaque->gate_up_w_col_stride = gu_s0;
            opaque->gate_up_w_k_stride = gu_s1;
        } else {
            // Layout B: [d, 2*di] — K is dim0, output column is dim1.
            opaque->gate_up_w_k_stride = gu_s0;
            opaque->gate_up_w_col_stride = gu_s1;
        }
    }

    // ── Workspace layout ──
    //   normalized : [ntok, d]     contiguous   -> RMSNorm out,   GateUp in
    //   gate_up    : [ntok, 2*di]  contiguous   -> GateUp out,    SwiGLU in
    //   hidden     : [ntok, di]    contiguous   -> SwiGLU out,    Down  in
    //   inner_ws   : max(sub->workspaceSize())  shared by sub-descriptors
    //
    // The compact hidden slab (stride=di instead of stride=2*di) gives the
    // Down-GEMM a tightly packed K dimension, which matters on BIV150 where
    // cuBLAS 10.2 tensor-core paths prefer aligned contiguous leading dims.
    opaque->normalized_bytes = alignUp(ntok * d * dtype_sz, kWsAlign);
    opaque->gate_up_bytes = alignUp(ntok * 2 * di * dtype_sz, kWsAlign);
    opaque->hidden_bytes = alignUp(ntok * di * dtype_sz, kWsAlign);

    DescScope scope;

    // RMSNorm, SwiGLU and the residual Add no longer get their own sub-
    // descriptors — they are launched directly inside calculate() with
    // params resolved from FusedFFNInfo. Their workspaces would have been 0
    // anyway (the standalone implementations are workspace-free for these
    // contiguous shapes), so dropping them does not change inner_ws_bytes.

    auto normalized_desc = scope.adopt(
        make2D(info.dtype, ntok, d, static_cast<ptrdiff_t>(d), 1));

    // ── GateUp GEMM sub-descriptor (cuBLAS path) ──
    //   [ntok, 2*di] = [ntok, d] @ [d, 2*di]
    auto gate_up_c_desc = scope.adopt(
        make2D(info.dtype, ntok, 2 * di, static_cast<ptrdiff_t>(2 * di), 1));
    auto gate_up_b_desc = scope.adopt(
        makeWeightAsKN(info.mtype, d, 2 * di, gate_up_weight_desc));

    {
        op::gemm::nvidia::Descriptor *sub = nullptr;
        CHECK_STATUS(op::gemm::nvidia::Descriptor::create(
            handle_, &sub, gate_up_c_desc, normalized_desc, gate_up_b_desc));
        opaque->gate_up_gemm.reset(sub);
        opaque->inner_ws_bytes = std::max(opaque->inner_ws_bytes, sub->workspaceSize());
    }

    // hidden_desc is still needed: it describes the A-matrix of the Down GEMM.
    auto hidden_desc = scope.adopt(
        make2D(info.dtype, ntok, di, static_cast<ptrdiff_t>(di), 1));

    // ── Down GEMM sub-descriptor (cuBLAS path) ──
    //   out = [beta * out] + 1.0 * hidden @ down_weight
    auto out_view = scope.adopt(
        make2D(info.dtype, ntok, d, info.out_stride, 1));
    auto down_b_desc = scope.adopt(
        makeWeightAsKN(info.mtype, di, d, down_weight_desc));

    {
        op::gemm::nvidia::Descriptor *sub = nullptr;
        CHECK_STATUS(op::gemm::nvidia::Descriptor::create(
            handle_, &sub, out_view, hidden_desc, down_b_desc));
        opaque->down_gemm.reset(sub);
        opaque->inner_ws_bytes = std::max(opaque->inner_ws_bytes, sub->workspaceSize());
    }

    const size_t workspace_size = opaque->normalized_bytes + opaque->gate_up_bytes + opaque->hidden_bytes + alignUp(opaque->inner_ws_bytes, kWsAlign);

    *desc_ptr = new Descriptor(
        opaque.release(),
        std::move(info),
        workspace_size,
        handle->device, handle->device_id);
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

    const size_t ntok = _info.ntok();
    const size_t d = _info.d();
    const size_t di = _info.di();
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // Partition the workspace into the three persistent slabs plus an
    // inner scratch buffer shared by the two GEMM sub-descriptors.
    char *ws = static_cast<char *>(workspace);
    void *normalized_buf = ws;
    ws += _opaque->normalized_bytes;
    void *gate_up_buf = ws;
    ws += _opaque->gate_up_bytes;
    void *hidden_buf = ws;
    ws += _opaque->hidden_bytes;
    void *inner_ws = ws;
    const size_t inner_ws_size = _opaque->inner_ws_bytes;

    // ── Stage 1: RMSNorm — direct kernel launch. ──
    // Block size is hardcoded to 1024. On BI-V150 maxThreadsPerBlock is 8192,
    // and every other backend that compiles this TU supports 1024; doing this
    // statically lets calculate() skip the standalone RMSNorm's per-call
    // maxThreadsPerBlock() lookup + 4-way block-size switch.
    {
        constexpr unsigned int kRmsBlock = 1024;
        const ptrdiff_t in_stride = _info.in_stride;
        const ptrdiff_t out_stride = static_cast<ptrdiff_t>(d);

#define LAUNCH_RMSNORM(TD, TW)                                       \
    rmsnormKernel<kRmsBlock, float, TD, TW>                          \
        <<<static_cast<unsigned>(ntok), kRmsBlock, 0, cuda_stream>>>(\
            reinterpret_cast<TD *>(normalized_buf),                  \
            reinterpret_cast<const TD *>(in),                        \
            reinterpret_cast<const TW *>(norm_weight),               \
            ntok, d, _info.epsilon,                                  \
            in_stride, out_stride)

        if (_info.dtype == INFINI_DTYPE_F16 && _info.wtype == INFINI_DTYPE_F16) {
            LAUNCH_RMSNORM(half, half);
        } else if (_info.dtype == INFINI_DTYPE_F16 && _info.wtype == INFINI_DTYPE_F32) {
            LAUNCH_RMSNORM(half, float);
        } else if (_info.dtype == INFINI_DTYPE_BF16 && _info.wtype == INFINI_DTYPE_BF16) {
            LAUNCH_RMSNORM(__nv_bfloat16, __nv_bfloat16);
        } else if (_info.dtype == INFINI_DTYPE_BF16 && _info.wtype == INFINI_DTYPE_F32) {
            LAUNCH_RMSNORM(__nv_bfloat16, float);
        } else if (_info.dtype == INFINI_DTYPE_F32 && _info.wtype == INFINI_DTYPE_F32) {
            LAUNCH_RMSNORM(float, float);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
#undef LAUNCH_RMSNORM
    }

    if (_opaque->use_deep_fused) {
        // Stage 2+3 fused: one kernel produces hidden = SiLU(norm@Wg) * (norm@Wu)
        // directly, eliminating the gate_up_buf HBM round-trip. Row strides
        // for X and hidden are d and di respectively (contiguous buffers
        // allocated at create time).
        constexpr unsigned int kBlock = 256;
        dim3 grid(static_cast<unsigned>(ntok), static_cast<unsigned>(di));
        dim3 block(kBlock);

#define DEEP_FUSED_LAUNCH(TD, TW)                         \
    deepFusedGateUpSiluKernel<kBlock, float, TD, TW>      \
        <<<grid, block, 0, cuda_stream>>>(                \
            reinterpret_cast<TD *>(hidden_buf),           \
            reinterpret_cast<const TD *>(normalized_buf), \
            reinterpret_cast<const TW *>(gate_up_weight), \
            ntok, d, di,                                  \
            static_cast<ptrdiff_t>(d),                    \
            static_cast<ptrdiff_t>(di),                   \
            _opaque->gate_up_w_k_stride,                  \
            _opaque->gate_up_w_col_stride,                \
            /*gate_col_base=*/0u, /*up_col_base=*/di)

        if (_info.dtype == INFINI_DTYPE_F16 && _info.mtype == INFINI_DTYPE_F16) {
            DEEP_FUSED_LAUNCH(half, half);
        } else if (_info.dtype == INFINI_DTYPE_BF16 && _info.mtype == INFINI_DTYPE_BF16) {
            DEEP_FUSED_LAUNCH(__nv_bfloat16, __nv_bfloat16);
        } else if (_info.dtype == INFINI_DTYPE_F32 && _info.mtype == INFINI_DTYPE_F32) {
            DEEP_FUSED_LAUNCH(float, float);
        } else if (_info.dtype == INFINI_DTYPE_F16 && _info.mtype == INFINI_DTYPE_F32) {
            DEEP_FUSED_LAUNCH(half, float);
        } else if (_info.dtype == INFINI_DTYPE_BF16 && _info.mtype == INFINI_DTYPE_F32) {
            DEEP_FUSED_LAUNCH(__nv_bfloat16, float);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
#undef DEEP_FUSED_LAUNCH
    } else {
        // Stage 2: GateUp GEMM  -->  gate_up_buf = normalized_buf @ gate_up_weight
        CHECK_STATUS(_opaque->gate_up_gemm->calculate(
            inner_ws, inner_ws_size,
            gate_up_buf, /*beta=*/0.f,
            normalized_buf, gate_up_weight,
            /*alpha=*/1.f, stream));

        // ── Stage 3: SwiGLU — direct kernel launch. ──
        // Reads two halves of the interleaved gate_up_buf row (stride 2*di)
        // and writes hidden_buf (stride di). Skips the standalone SwiGLU's
        // elementwise framework metadata setup per call.
        {
            constexpr unsigned int kSwigluBlock = 256;
            const ptrdiff_t hidden_row_stride = static_cast<ptrdiff_t>(di);
            const ptrdiff_t gate_up_row_stride = static_cast<ptrdiff_t>(2 * di);

#define LAUNCH_SWIGLU(TD)                                            \
    swigluOutKernel<kSwigluBlock, float, TD>                         \
        <<<static_cast<unsigned>(ntok), kSwigluBlock, 0, cuda_stream>>>( \
            reinterpret_cast<TD *>(hidden_buf),                      \
            reinterpret_cast<const TD *>(gate_up_buf),               \
            ntok, di,                                                \
            hidden_row_stride, gate_up_row_stride)

            switch (_info.dtype) {
            case INFINI_DTYPE_F16:
                LAUNCH_SWIGLU(half);
                break;
            case INFINI_DTYPE_BF16:
                LAUNCH_SWIGLU(__nv_bfloat16);
                break;
            case INFINI_DTYPE_F32:
                LAUNCH_SWIGLU(float);
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
#undef LAUNCH_SWIGLU
        }
    }

    // Stage 4: Down GEMM, with optional in-place residual fuse via beta=1.
    //   fuse path : out = 1.0 * out + hidden_buf @ down_weight
    //   plain path: out = 0.0 * out + hidden_buf @ down_weight
    const bool fuse_residual = _opaque->has_residual && (out == residual);
    CHECK_STATUS(_opaque->down_gemm->calculate(
        inner_ws, inner_ws_size,
        out, /*beta=*/fuse_residual ? 1.f : 0.f,
        hidden_buf, down_weight,
        /*alpha=*/1.f, stream));

    // ── Stage 5: Residual Add — direct kernel launch. ──
    // Only fires when out != residual (the (out == residual) case is already
    // absorbed by the Down GEMM's beta=1 epilogue above).
    if (_opaque->has_residual && !fuse_residual) {
        constexpr unsigned int kAddBlock = 256;

#define LAUNCH_ADD(TD)                                               \
    residualAddKernel<kAddBlock, float, TD>                          \
        <<<static_cast<unsigned>(ntok), kAddBlock, 0, cuda_stream>>>(\
            reinterpret_cast<TD *>(out),                             \
            reinterpret_cast<const TD *>(out),                       \
            reinterpret_cast<const TD *>(residual),                  \
            ntok, d,                                                 \
            _info.out_stride,                                        \
            _info.residual_stride)

        switch (_info.dtype) {
        case INFINI_DTYPE_F16:
            LAUNCH_ADD(half);
            break;
        case INFINI_DTYPE_BF16:
            LAUNCH_ADD(__nv_bfloat16);
            break;
        case INFINI_DTYPE_F32:
            LAUNCH_ADD(float);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
#undef LAUNCH_ADD
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::fused_ffn::nvidia
