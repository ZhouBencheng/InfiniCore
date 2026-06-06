"""
Performance benchmark: infiniopFusedFFN vs. the equivalent unfused chain.

Plain warmup + repeat per-call timing. For each (shape, dtype) case we
warmup NUM_PRERUN times, then time NUM_ITERATIONS individual calls with a
synchronize_device() barrier before and after each one, and report the
mean fused vs baseline per-call latency plus the speedup ratio. PyTorch is
timed as an informational reference and is NOT used for the speedup
number.

================================================================================
Two comparison modes — pick with --strict / default
================================================================================

DEFAULT MODE — "realistic" (out == residual)
--------------------------------------------
Baseline = 4 ops:
    rmsnorm(normalized, in, norm_w, eps)
    gemm   (gate_up,   normalized, gate_up_w,  alpha=1, beta=0)   # GateUp
    swiglu (hidden,    up_half,    gate_half)
    gemm   (out_res,   hidden,     down_w,     alpha=1, beta=1)   # Down + residual
The final Down GEMM uses beta=1 against a buffer that already holds the
residual content, so the residual add is folded into the GEMM epilogue — no
standalone Add call.

Fused = 1 call:
    infiniopFusedFFN(out=buf, in=..., residual=buf, ...)   # out == residual
Inside FusedFFN, `out == residual` triggers the same beta=1 fuse on the
internal Down GEMM (see fused_ffn_nvidia.cu :: Stage 4) and Stage 5 Add is
skipped.

STRICT MODE — --strict (out != residual)
----------------------------------------
Baseline = 5 ops: same first 4 with the down GEMM running beta=0, plus an
explicit infiniopAdd as Stage 5. Fused = 1 infiniopFusedFFN call with
distinct out / residual pointers, so its internal Stage 5 Add fires too.
Useful for isolating the "pure orchestration delta" of merging 5 ops into
one descriptor.

================================================================================
Stages (numbering used in source comments and output)
================================================================================
    Stage 1  infiniopRMSNorm    x [N,d]                -> normalized [N,d]
    Stage 2  infiniopGemm       normalized @ W_gu^T    -> gate_up [N,2*di]
    Stage 3  infiniopSwiGLU     up * silu(gate)        -> hidden [N,di]
    Stage 4  infiniopGemm       hidden @ W_d^T         -> out [N,d]
                                  realistic: beta=1, c = out_residual_buf
                                  strict:    beta=0, separate out + Add
    Stage 5  infiniopAdd        out + residual -> out  (strict mode only)

================================================================================
Controlled variables — identical on both candidates
================================================================================
  - x, norm_w, gate_up_w, down_w shared by pointer (allocated ONCE)
  - realistic mode: out_residual_buf is one buffer initialised with
    residual_init once; subsequent iters let it drift — the GPU work is
    determined by shape, not by buffer contents, so drift does not change
    the comparison. Correctness checks reset between candidates.
  - strict mode: out / residual are distinct (out_buf, residual_buf)
  - all intermediate buffers (normalized, gate_up, hidden) allocated ONCE
  - all descriptors and workspaces created ONCE outside the timed region
  - same dtype throughout (no upcast tricks)
  - same warmup count, same iter count, same stream (NULL)
  - synchronize_device() barrier inside timed_loop per iter, identical for both
  - baseline GEMMs view weight tensors with zero-copy transposed strides
    (shape=[K,N], strides=[1,K]) — same as FusedFFN's internal makeWeightAsKN
  - Stage 3 in baseline uses infiniopSwiGLU (same op FusedFFN uses internally),
    NOT a mul+silu surrogate

What this benchmark does NOT do, to keep the comparison honest:
  - does NOT count per-iter intermediate-buffer allocation
  - does NOT add artificial syncs between baseline stages
  - does NOT use a slower mul+silu surrogate for SwiGLU
  - does NOT include descriptor create/destroy in per-iter time
  - does NOT use PyTorch as the baseline timing comparator

================================================================================
Environment knob
================================================================================
INFINIOP_FUSED_FFN_DEEP=1 enables the deep-fused Stage 2+3 single-kernel path
for ntok <= 4 (default 5-stage internal path otherwise). Speedup numbers
will reflect that path when set. Strict mode is otherwise unaffected.
"""

import ctypes
import os
import sys
import time
from ctypes import c_uint64

import torch
import torch.nn.functional as F

from libinfiniop import (
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    LIBINFINIOP,
    TestTensor,
    TestWorkspace,
    check_error,
    get_args,
    get_test_devices,
    infiniopOperatorDescriptor_t,
    infiniopTensorDescriptor_t,
)
from libinfiniop.utils import (
    create_handle,
    destroy_handle,
    get_sync_func,
    synchronize_device,
)

# ------------------------------------------------------------------------------
# Benchmark configuration
# ------------------------------------------------------------------------------
#
# Shape grid covers ntok = 1, 4, 8, 16, 32, 128, 512, 2048 against a few
# (hidden_dim, intermediate_dim) pairs that exercise the small / medium /
# GEMM-bound regimes. The intent is to show the speedup curve across ntok
# at op level, not to fit any particular inference pipeline.

# (ntok, hidden_dim, intermediate_dim) — case identified by shape params only,
# no model code names. ntok is the leading "rows" dimension (batch * seq).
_BENCH_CASES = [
    (1,    2048, 5632),
    (4,    2048, 5632),
    (8,    2048, 5632),
    (16,   2048, 5632),
    (32,   2048, 5632),
    (128,  2048, 5632),
    (1,    4096, 11008),
    (8,    4096, 11008),
    (32,   4096, 11008),
    (128,  4096, 11008),
    (1,    5120, 13824),
    (16,   3584, 18944),
    (512,  4096, 11008),
    (2048, 4096, 11008),
]


def case_label(ntok, d, di):
    return f"ntok={ntok:<4} d={d:<5} di={di}"

_BENCH_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16:  {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-2, "rtol": 5e-2},
    InfiniDtype.F32:  {"atol": 1e-5, "rtol": 1e-5},
}

# warmup + repeat:
#   - run the op NUM_PRERUN times (untimed) to prime kernel caches / GPU clock
#   - run the op NUM_ITERATIONS times, each one timed individually with a
#     sync before and after the call
# The mean per-call latency is the headline number. stddev is reported so
# you can see the noise band — sub-millisecond ops sit close to the
# perf_counter+sync resolution floor and will read noisy.
NUM_PRERUN = 30
NUM_ITERATIONS = 200

# Set at startup from CLI / env.
STRICT_MODE = False


def make_tensor_desc(dt, shape, strides):
    desc = infiniopTensorDescriptor_t()
    ndim = len(shape)
    c_shape = (ctypes.c_size_t * ndim)(*shape)
    c_strides = (ctypes.c_ssize_t * ndim)(*strides)
    check_error(
        LIBINFINIOP.infiniopCreateTensorDescriptor(
            ctypes.byref(desc), ndim, c_shape, c_strides, dt
        )
    )
    return desc


def destroy_tensor_desc(desc):
    if desc is not None:
        check_error(LIBINFINIOP.infiniopDestroyTensorDescriptor(desc))


def reference_fused_ffn(x, residual, norm_w, gate_up_w, down_w, eps):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + eps)
    normalized = (normalized * norm_w.float()).to(x.dtype)
    gate_up = F.linear(normalized, gate_up_w)
    di = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :di], gate_up[..., di:]
    hidden = F.silu(gate) * up
    out = F.linear(hidden, down_w)
    if residual is not None:
        out = out + residual
    return out


def timed_loop(func, num_iters, device):
    """Plain warmup+repeat timing.

    Each iteration is one func() call timed end-to-end with a
    synchronize_device() barrier before and after. Returns (mean, stddev)
    of the per-call latencies in seconds.
    """
    synchronize_device(device)
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        func()
        synchronize_device(device)
        times.append(time.perf_counter() - t0)
    mean = sum(times) / num_iters
    var = sum((t - mean) ** 2 for t in times) / num_iters
    return mean, var ** 0.5


def warmup(func, n):
    for _ in range(n):
        func()


def bench_one_case(handle, device, ntok, d, di, dtype, sync):
    dtype_name = InfiniDtypeNames[dtype]
    mode_name = "strict-5op" if STRICT_MODE else "realistic-4op"
    print()
    print("-" * 78)
    print(f"[{case_label(ntok, d, di)}]  {InfiniDeviceNames[device]}  "
          f"dtype={dtype_name}  mode={mode_name}")
    print("-" * 78)

    wscale = 1.0 / (d ** 0.5)

    # Inputs / weights — shared by both candidates.
    x = TestTensor((ntok, d), None, dtype, device)
    norm_w = TestTensor((d,), None, dtype, device)
    gate_up_w = TestTensor((2 * di, d), None, dtype, device, scale=wscale)
    down_w = TestTensor((d, di), None, dtype, device, scale=wscale)
    residual_init = TestTensor((ntok, d), None, dtype, device)

    # Intermediate buffers — pre-allocated, reused.
    normalized_buf = TestTensor((ntok, d), None, dtype, device, mode="zeros")
    gate_up_buf = TestTensor((ntok, 2 * di), None, dtype, device, mode="zeros")
    hidden_buf = TestTensor((ntok, di), None, dtype, device, mode="zeros")

    # ── Realistic vs strict mode buffer layout ──
    #   Realistic: out and residual share one buffer (`out_residual_buf`).
    #              For both candidates, the Down GEMM writes back to this
    #              buffer with beta=1, so the previous contents are the
    #              residual being added in.  `residual_init` is the saved
    #              original residual value used to reset for correctness.
    #   Strict:    out (out_residual_buf used as plain out) and residual
    #              (residual_init) are distinct buffers; the explicit Stage 5
    #              Add runs on the baseline side and is forced on the fused
    #              side too (because out != residual disables the beta=1
    #              fuse inside FusedFFN).
    out_residual_buf = TestTensor((ntok, d), None, dtype, device, mode="zeros")

    epsilon = 1e-6

    # Reset out_residual_buf := residual_init via a D2D copy. Real inference
    # engines pay no equivalent cost (logits_in IS the residual buffer); we
    # do this in our standalone benchmark only to reset state between
    # candidates for the correctness check.
    def reset_out_buf():
        out_residual_buf.actual_tensor().copy_(residual_init.actual_tensor())
        if sync is not None:
            sync()

    # ── PyTorch reference ──
    ref = reference_fused_ffn(
        x.torch_tensor(),
        residual_init.torch_tensor(),
        norm_w.torch_tensor(),
        gate_up_w.torch_tensor(),
        down_w.torch_tensor(),
        epsilon,
    )
    if sync is not None:
        sync()

    # ============================================================
    # Fused descriptor.  Created the same way for both modes; the
    # out / residual pointers are what differ at call time.
    # ============================================================
    fused_desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFusedFFNDescriptor(
            handle,
            ctypes.byref(fused_desc),
            out_residual_buf.descriptor,
            x.descriptor,
            residual_init.descriptor,  # ok: same shape/dtype/strides regardless of mode
            norm_w.descriptor,
            gate_up_w.descriptor,
            down_w.descriptor,
            ctypes.c_float(epsilon),
        )
    )
    fused_ws_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetFusedFFNWorkspaceSize(
            fused_desc, ctypes.byref(fused_ws_size)
        )
    )
    fused_ws = TestWorkspace(fused_ws_size.value, device)

    if STRICT_MODE:
        # out != residual; FusedFFN will execute its internal Stage 5 Add.
        def run_fused():
            check_error(LIBINFINIOP.infiniopFusedFFN(
                fused_desc, fused_ws.data(), fused_ws_size.value,
                out_residual_buf.data(), x.data(), residual_init.data(),
                norm_w.data(), gate_up_w.data(), down_w.data(), None))
    else:
        # out == residual; FusedFFN will use the down-GEMM beta=1 trick and
        # skip Stage 5 entirely. The actual residual content lives in
        # out_residual_buf at call time.
        def run_fused():
            check_error(LIBINFINIOP.infiniopFusedFFN(
                fused_desc, fused_ws.data(), fused_ws_size.value,
                out_residual_buf.data(), x.data(), out_residual_buf.data(),
                norm_w.data(), gate_up_w.data(), down_w.data(), None))

    # ============================================================
    # Baseline descriptors / workspaces.
    # ============================================================
    rmsnorm_desc = infiniopOperatorDescriptor_t()
    check_error(LIBINFINIOP.infiniopCreateRMSNormDescriptor(
        handle, ctypes.byref(rmsnorm_desc),
        normalized_buf.descriptor, x.descriptor, norm_w.descriptor,
        ctypes.c_float(epsilon)))
    rmsnorm_ws_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetRMSNormWorkspaceSize(
        rmsnorm_desc, ctypes.byref(rmsnorm_ws_size)))
    rmsnorm_ws = TestWorkspace(rmsnorm_ws_size.value, device)

    # GateUp GEMM
    gate_up_w_T_desc = make_tensor_desc(dtype, [d, 2 * di], [1, d])
    gateup_desc = infiniopOperatorDescriptor_t()
    check_error(LIBINFINIOP.infiniopCreateGemmDescriptor(
        handle, ctypes.byref(gateup_desc),
        gate_up_buf.descriptor, normalized_buf.descriptor, gate_up_w_T_desc))
    gateup_ws_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetGemmWorkspaceSize(
        gateup_desc, ctypes.byref(gateup_ws_size)))
    gateup_ws = TestWorkspace(gateup_ws_size.value, device)

    # SwiGLU (Stage 3) — operates on two strided halves of gate_up_buf.
    swiglu_hidden_desc = make_tensor_desc(dtype, [ntok, di], [di, 1])
    half_desc = make_tensor_desc(dtype, [ntok, di], [2 * di, 1])
    swiglu_desc = infiniopOperatorDescriptor_t()
    check_error(LIBINFINIOP.infiniopCreateSwiGLUDescriptor(
        handle, ctypes.byref(swiglu_desc),
        swiglu_hidden_desc, half_desc, half_desc))
    swiglu_ws_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetSwiGLUWorkspaceSize(
        swiglu_desc, ctypes.byref(swiglu_ws_size)))
    swiglu_ws = TestWorkspace(swiglu_ws_size.value, device)

    _DT_BYTES = {InfiniDtype.F16: 2, InfiniDtype.BF16: 2, InfiniDtype.F32: 4}
    up_byte_offset = di * _DT_BYTES[dtype]

    # Down GEMM (Stage 4). Destination is out_residual_buf for both modes;
    # only beta differs.
    down_w_T_desc = make_tensor_desc(dtype, [di, d], [1, di])
    down_desc = infiniopOperatorDescriptor_t()
    check_error(LIBINFINIOP.infiniopCreateGemmDescriptor(
        handle, ctypes.byref(down_desc),
        out_residual_buf.descriptor, hidden_buf.descriptor, down_w_T_desc))
    down_ws_size = c_uint64(0)
    check_error(LIBINFINIOP.infiniopGetGemmWorkspaceSize(
        down_desc, ctypes.byref(down_ws_size)))
    down_ws = TestWorkspace(down_ws_size.value, device)

    add_desc = None
    add_ws = None
    add_ws_size = c_uint64(0)
    if STRICT_MODE:
        # Stage 5 explicit Add: out = out + residual.
        add_desc = infiniopOperatorDescriptor_t()
        check_error(LIBINFINIOP.infiniopCreateAddDescriptor(
            handle, ctypes.byref(add_desc),
            out_residual_buf.descriptor,
            out_residual_buf.descriptor,
            residual_init.descriptor))
        check_error(LIBINFINIOP.infiniopGetAddWorkspaceSize(
            add_desc, ctypes.byref(add_ws_size)))
        add_ws = TestWorkspace(add_ws_size.value, device)

    c_alpha = ctypes.c_float(1.0)
    c_beta_zero = ctypes.c_float(0.0)
    c_beta_one = ctypes.c_float(1.0)

    def run_baseline_stage1():
        check_error(LIBINFINIOP.infiniopRMSNorm(
            rmsnorm_desc, rmsnorm_ws.data(), rmsnorm_ws_size.value,
            normalized_buf.data(), x.data(), norm_w.data(), None))

    def run_baseline_stage2():
        check_error(LIBINFINIOP.infiniopGemm(
            gateup_desc, gateup_ws.data(), gateup_ws_size.value,
            gate_up_buf.data(), normalized_buf.data(), gate_up_w.data(),
            c_alpha, c_beta_zero, None))

    def run_baseline_stage3():
        check_error(LIBINFINIOP.infiniopSwiGLU(
            swiglu_desc, swiglu_ws.data(), swiglu_ws_size.value,
            hidden_buf.data(),
            gate_up_buf.data() + up_byte_offset,  # a = up
            gate_up_buf.data(),                   # b = gate
            None))

    if STRICT_MODE:
        def run_baseline_stage4():
            # beta=0: out_residual_buf = hidden @ down_w (no residual yet)
            check_error(LIBINFINIOP.infiniopGemm(
                down_desc, down_ws.data(), down_ws_size.value,
                out_residual_buf.data(), hidden_buf.data(), down_w.data(),
                c_alpha, c_beta_zero, None))

        def run_baseline_stage5():
            # out = out + residual
            check_error(LIBINFINIOP.infiniopAdd(
                add_desc, add_ws.data(), add_ws_size.value,
                out_residual_buf.data(), out_residual_buf.data(),
                residual_init.data(), None))

        def run_baseline():
            run_baseline_stage1()
            run_baseline_stage2()
            run_baseline_stage3()
            run_baseline_stage4()
            run_baseline_stage5()
    else:
        def run_baseline_stage4():
            # beta=1: out_residual_buf = (previous out_residual_buf) +
            # hidden @ down_w. The previous content is the residual
            # (initialised once, then drifts during timing — same as the
            # fused side, which also drifts identically).
            check_error(LIBINFINIOP.infiniopGemm(
                down_desc, down_ws.data(), down_ws_size.value,
                out_residual_buf.data(), hidden_buf.data(), down_w.data(),
                c_alpha, c_beta_one, None))

        def run_baseline():
            run_baseline_stage1()
            run_baseline_stage2()
            run_baseline_stage3()
            run_baseline_stage4()

    # ============================================================
    # Correctness gate — reset out_residual_buf to residual_init
    # before each candidate so both compute against a clean residual.
    # ============================================================
    atol = _TOLERANCE_MAP[dtype]["atol"]
    rtol = _TOLERANCE_MAP[dtype]["rtol"]

    reset_out_buf()
    run_baseline()
    if sync is not None:
        sync()
    base_ok = torch.allclose(out_residual_buf.actual_tensor(), ref,
                             atol=atol, rtol=rtol)

    reset_out_buf()
    run_fused()
    if sync is not None:
        sync()
    fused_ok = torch.allclose(out_residual_buf.actual_tensor(), ref,
                              atol=atol, rtol=rtol)

    if not base_ok:
        print("  [SKIP] baseline output disagrees with reference.")
    if not fused_ok:
        print("  [SKIP] fused output disagrees with reference.")

    result = None
    if base_ok and fused_ok:
        # ============================================================
        # Timed regions.
        # ============================================================
        reset_out_buf()
        warmup(run_baseline, NUM_PRERUN)
        base_mean, base_std = timed_loop(run_baseline, NUM_ITERATIONS, device)

        reset_out_buf()
        warmup(run_fused, NUM_PRERUN)
        fused_mean, fused_std = timed_loop(run_fused, NUM_ITERATIONS, device)

        # PyTorch reference — informational only.
        def run_torch():
            _ = reference_fused_ffn(
                x.torch_tensor(),
                residual_init.torch_tensor(),
                norm_w.torch_tensor(),
                gate_up_w.torch_tensor(),
                down_w.torch_tensor(),
                epsilon,
            )
        warmup(run_torch, max(3, NUM_PRERUN // 4))
        torch_mean, _ = timed_loop(run_torch, max(10, NUM_ITERATIONS // 4), device)

        ms = lambda s: s * 1e3
        speedup = base_mean / fused_mean if fused_mean > 0 else float("nan")
        print(f"  baseline ({'5-op strict' if STRICT_MODE else '4-op realistic'}) : "
              f"{ms(base_mean):8.4f} ms  (stddev {ms(base_std):.4f})")
        print(f"  fused                                : "
              f"{ms(fused_mean):8.4f} ms  (stddev {ms(fused_std):.4f})")
        print(f"  pytorch (ref only)                   : "
              f"{ms(torch_mean):8.4f} ms")
        print(f"  speedup vs baseline                  : {speedup:6.3f}x  "
              f"({(speedup - 1) * 100:+.2f}%)")
        result = (case_label(ntok, d, di), dtype_name,
                  ms(base_mean), ms(fused_mean), ms(torch_mean), speedup)

    # Cleanup
    if STRICT_MODE:
        check_error(LIBINFINIOP.infiniopDestroyAddDescriptor(add_desc))
    check_error(LIBINFINIOP.infiniopDestroyGemmDescriptor(down_desc))
    check_error(LIBINFINIOP.infiniopDestroySwiGLUDescriptor(swiglu_desc))
    check_error(LIBINFINIOP.infiniopDestroyGemmDescriptor(gateup_desc))
    check_error(LIBINFINIOP.infiniopDestroyRMSNormDescriptor(rmsnorm_desc))
    check_error(LIBINFINIOP.infiniopDestroyFusedFFNDescriptor(fused_desc))
    destroy_tensor_desc(gate_up_w_T_desc)
    destroy_tensor_desc(down_w_T_desc)
    destroy_tensor_desc(swiglu_hidden_desc)
    destroy_tensor_desc(half_desc)
    return result


def _print_preamble():
    mode_name = "STRICT 5-op (out != residual)" if STRICT_MODE else "REALISTIC 4-op (out == residual)"
    print("=" * 78)
    print(" FusedFFN performance benchmark vs. unfused chain")
    print("=" * 78)
    print(f" Mode      = {mode_name}")
    if STRICT_MODE:
        print(" Baseline  = RMSNorm -> Gemm -> SwiGLU -> Gemm -> Add")
        print("             (out and residual are distinct buffers; explicit Stage 5 Add)")
    else:
        print(" Baseline  = RMSNorm -> Gemm -> SwiGLU -> Gemm(beta=1, c=residual)")
        print("             (residual fused into Down GEMM via beta=1; no explicit Add)")
    print(" Fused     = single infiniopFusedFFN call")
    if not STRICT_MODE:
        print("             (out == residual; FusedFFN fuses residual into Down GEMM")
        print("              via beta=1 and skips internal Stage 5)")
    print(" Both candidates:")
    print("   - share x / norm_w / gate_up_w / down_w by pointer")
    print("   - reuse pre-allocated intermediate buffers and workspaces")
    print("   - timed per-call: synchronize_device() before and after each iter")
    print(f"   - warmup={NUM_PRERUN} calls, repeat={NUM_ITERATIONS} calls")
    deep = os.environ.get("INFINIOP_FUSED_FFN_DEEP")
    if deep:
        print(f" NOTE: INFINIOP_FUSED_FFN_DEEP={deep} is set; fused side may use")
        print(f"       the deep-fused Stage 2+3 kernel for ntok <= 4.")
    print("=" * 78)


def _print_summary(rows):
    if not rows:
        return
    print()
    print("=" * 78)
    print(" Summary")
    print("=" * 78)
    print(f" {'case':<36s} {'dtype':<6s} {'base':>10s} {'fused':>10s} {'speedup':>10s}")
    print(" " + "-" * 75)
    for label, dt, base_ms, fused_ms, _torch_ms, sp in rows:
        short = label if len(label) <= 36 else label[:33] + "..."
        delta_pct = (sp - 1) * 100
        print(f" {short:<36s} {dt:<6s} {base_ms:>9.4f}m {fused_ms:>9.4f}m "
              f"{sp:>6.3f}x {delta_pct:+6.1f}%")


def main():
    global NUM_PRERUN, NUM_ITERATIONS, STRICT_MODE

    # Strip --strict before passing argv to libinfiniop's get_args (which
    # otherwise rejects unknown flags).
    if "--strict" in sys.argv:
        STRICT_MODE = True
        sys.argv.remove("--strict")

    args = get_args()
    if args.num_prerun != 10:
        NUM_PRERUN = args.num_prerun
    if args.num_iterations != 1000:
        NUM_ITERATIONS = args.num_iterations

    devices = get_test_devices(args)
    if not devices:
        print("No device selected. Pass one of --cpu / --nvidia / --iluvatar / --qy.")
        return

    _print_preamble()

    rows = []
    for device in devices:
        print()
        print(f"### device: {InfiniDeviceNames[device]} ###")
        LIBINFINIOP.infinirtSetDevice(device, ctypes.c_int(0))
        handle = create_handle()
        sync = get_sync_func(device)
        try:
            for ntok, d, di in _BENCH_CASES:
                for dtype in _BENCH_DTYPES:
                    r = bench_one_case(handle, device, ntok, d, di, dtype, sync)
                    if r is not None:
                        rows.append(r)
        finally:
            destroy_handle(handle)

    _print_summary(rows)


if __name__ == "__main__":
    main()
