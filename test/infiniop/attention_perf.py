"""
Performance benchmark: Iluvatar fused infiniopAttention vs the unfused
fallback path inside the SAME operator.

Both code paths live in src/infiniop/ops/attention/operator.cc:
  - fused:     op::attention::iluvatar::fused_attention(...)
               (a single hand-written ivcore11 kernel)
  - baseline:  rearrange(k -> k_cache) + rearrange(v -> v_cache)
             + gemm(q * full_k)
             + causal_softmax
             + gemm(softmax * full_v)
             + rearrange(att_val -> out)
               (the original 4-stage assembly via existing infiniop ops)

The fused path is selected at descriptor-creation time inside
infiniopCreateAttentionDescriptor when can_use_iluvatar_fused_attention()
returns true (see operator.cc). A runtime kill-switch is already built in:

    INFINIOP_DISABLE_ILUVATAR_FUSED_ATTENTION=1  -> force fallback path

This script flips that environment variable around create calls so the
SAME infiniopAttention API exercises both implementations back-to-back.

================================================================================
Fused gate (operator.cc::can_use_iluvatar_fused_attention)
================================================================================
ALL of the following must hold or the fused path silently degrades into
the fallback path:
  - device == ILUVATAR
  - dtype == F16
  - out / q / k / v / k_cache / v_cache all contiguous
  - n_q_head == n_kv_head           (no GQA)
  - head_dim                <= 128
  - seq_len + pos           <= 8    (kIluvatarFusedMaxTotalSeqLen)

If any gate is violated the fused descriptor falls back, so its timing
becomes identical to the baseline descriptor and the speedup ratio
collapses to ~1.0x. The benchmark cases below are chosen so the fused
path always engages; PyTorch is included only as a sanity reference and
is NOT part of the speedup denominator.

================================================================================
Stability notes
================================================================================
  - The env var is read at create time only; the API call itself does not
    re-check. We build BOTH descriptors at the top of each case (one with
    env=unset, one with env="1") and reuse them across warmup and timed
    iterations.
  - infiniopAttention writes the new k/v into k_cache[:, pos:pos+seq_len, :]
    and v_cache accordingly. Because the inputs do not change between
    iterations, the cache content is overwritten with identical bytes each
    call -- no drift, the comparison stays honest across NUM_ITERATIONS.
  - Each per-call timing uses synchronize_device() before and after so
    sub-millisecond decode kernels read at the perf_counter+sync floor;
    that floor is the same on both candidates.
"""

import ctypes
import math
import os
import sys
import time
from ctypes import c_uint64

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
# Each tuple: (n_q_head, n_kv_head, seq_len, head_dim, pos)
# Every case must satisfy the fused gate listed in the header docstring.
_BENCH_CASES = [
    # decode-style: single new token, growing context up to length 8
    (32, 32, 1, 128, 0),    # total_seq_len = 1   (very first token)
    (32, 32, 1, 128, 3),    # total_seq_len = 4   (small KV context)
    (32, 32, 1, 128, 7),    # total_seq_len = 8   (length-bound 8)
    (16, 16, 1,  64, 0),    # smaller MHA, fresh
    (16, 16, 1,  64, 7),    # smaller MHA, length-bound
    ( 8,  8, 1, 128, 7),    # narrow head count, length-bound
    # short prefill (multi-token Q, total still <= 8)
    ( 8,  8, 4, 128, 0),    # 4-token prefill, fresh
    ( 8,  8, 4, 128, 4),    # 4-token prefill on top of 4-token KV
    (32, 32, 4, 128, 0),    # wider MHA, 4-token prefill
    (16, 16, 2,  64, 6),    # 2-token at length-bound 8
]

# The fused gate enforces dtype == F16. Anything else makes the fused
# descriptor fall back, so we don't sweep dtypes.
_BENCH_DTYPES = [InfiniDtype.F16]

# Loose tolerance: both paths reduce in slightly different orders so the
# F16 accumulation noise diverges; a single allclose at atol=1e-2 is enough
# to gate "obviously correct vs obviously broken".
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},
}

NUM_PRERUN = 30
NUM_ITERATIONS = 300

_ENV_KILL = "INFINIOP_DISABLE_ILUVATAR_FUSED_ATTENTION"


# ------------------------------------------------------------------------------
# PyTorch reference (lifted from attention.py, kept self-contained)
# ------------------------------------------------------------------------------
def causal_softmax_ref(x):
    dt = x.dtype
    mask = torch.tril(torch.ones_like(x), diagonal=-1).flip(dims=[-2, -1])
    masked = torch.where(mask == 1, -torch.inf, x.to(torch.float32))
    return torch.nn.functional.softmax(masked, dim=-1).to(dt)


def attention_ref(q, k, v, k_cache, v_cache, pos):
    """Reference computation matching infiniopAttention semantics.

    q / k / v are the *new* tokens at this step. k_cache[:, :pos, :] / v_cache[:,
    :pos, :] hold prior context. Output shape is [seq_len, n_q_head, head_dim].
    """
    dt = q.dtype
    n_q = q.shape[0]
    n_kv = k.shape[0]
    k_full = torch.cat([k_cache[:, :pos, :], k], dim=1)
    v_full = torch.cat([v_cache[:, :pos, :], v], dim=1)
    total_seq_len = k_full.shape[1]
    head_dim = v_full.shape[-1]
    if n_q != n_kv:
        q = q.reshape(n_kv, -1, head_dim)
    scores = (
        torch.einsum("hqd,hkd->hqk", q.to(torch.float32), k_full.to(torch.float32))
        .to(dt)
        .reshape(n_q, -1, total_seq_len)
    )
    scores = scores / (head_dim ** 0.5)
    weights = causal_softmax_ref(scores).reshape(n_kv, -1, total_seq_len)
    out = (
        torch.einsum("hqk,hkd->hqd", weights.to(torch.float32), v_full.to(torch.float32))
        .to(dt)
        .reshape(n_q, -1, head_dim)
        .permute(1, 0, 2)
        .contiguous()
    )
    return out


# ------------------------------------------------------------------------------
# Timing
# ------------------------------------------------------------------------------
def timed_loop(func, num_iters, device):
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


# ------------------------------------------------------------------------------
# Per-case bench
# ------------------------------------------------------------------------------
def case_label(n_q, n_kv, seq_len, head_dim, pos):
    return (
        f"n_q={n_q:<3d} n_kv={n_kv:<3d} seq_len={seq_len:<2d} "
        f"head_dim={head_dim:<3d} pos={pos:<2d}"
    )


def _check_gate(n_q, n_kv, seq_len, head_dim, pos, dtype):
    """Mirror operator.cc::can_use_iluvatar_fused_attention. Raise if violated."""
    msgs = []
    if dtype != InfiniDtype.F16:
        msgs.append(f"dtype must be F16, got {InfiniDtypeNames[dtype]}")
    if n_q != n_kv:
        msgs.append(f"n_q_head ({n_q}) must equal n_kv_head ({n_kv})")
    if head_dim > 128:
        msgs.append(f"head_dim ({head_dim}) must be <= 128")
    if seq_len + pos > 8:
        msgs.append(f"seq_len + pos ({seq_len + pos}) must be <= 8")
    if msgs:
        raise AssertionError("case violates fused gate: " + "; ".join(msgs))


def _build_attention_descriptor(handle, device, out_t, q, k, v, k_cache, v_cache, pos):
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAttentionDescriptor(
            handle,
            ctypes.byref(desc),
            out_t.descriptor,
            q.descriptor,
            k.descriptor,
            v.descriptor,
            k_cache.descriptor,
            v_cache.descriptor,
            pos,
        )
    )
    ws_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAttentionWorkspaceSize(desc, ctypes.byref(ws_size))
    )
    ws = TestWorkspace(ws_size.value, device)
    return desc, ws, ws_size


def bench_one_case(handle, device, n_q, n_kv, seq_len, head_dim, pos, dtype, sync):
    _check_gate(n_q, n_kv, seq_len, head_dim, pos, dtype)
    total_seq_len = seq_len + pos
    cache_buf_len = max(total_seq_len, 16)
    dtype_name = InfiniDtypeNames[dtype]

    print()
    print("-" * 78)
    print(f"[{case_label(n_q, n_kv, seq_len, head_dim, pos)}]  "
          f"{InfiniDeviceNames[device]}  dtype={dtype_name}")
    print("-" * 78)

    # Inputs (shared by both descriptors via the same tensor pointers).
    # Two distinct output buffers so we can keep both result tensors around
    # for the correctness check before timing kicks in.
    out_fused = TestTensor((seq_len, n_q, head_dim), None, dtype, device, mode="zeros")
    out_base = TestTensor((seq_len, n_q, head_dim), None, dtype, device, mode="zeros")
    q = TestTensor((n_q, seq_len, head_dim), None, dtype, device, scale=0.1)
    k = TestTensor((n_kv, seq_len, head_dim), None, dtype, device, scale=0.1)
    v = TestTensor((n_kv, seq_len, head_dim), None, dtype, device, scale=0.1)
    k_cache = TestTensor((n_kv, cache_buf_len, head_dim), None, dtype, device, scale=0.1)
    v_cache = TestTensor((n_kv, cache_buf_len, head_dim), None, dtype, device, scale=0.1)

    # Snapshot the original cache state before any infiniopAttention call;
    # the API writes new k/v into cache[:, pos:pos+seq_len, :], so we must
    # restore the snapshot between the two correctness runs.
    k_cache_snap = k_cache.actual_tensor().clone()
    v_cache_snap = v_cache.actual_tensor().clone()

    # PyTorch reference uses the *original* cache + the new k/v.
    ref = attention_ref(
        q.torch_tensor(),
        k.torch_tensor(),
        v.torch_tensor(),
        k_cache_snap,
        v_cache_snap,
        pos,
    )
    if sync is not None:
        sync()

    def reset_caches():
        k_cache.actual_tensor().copy_(k_cache_snap)
        v_cache.actual_tensor().copy_(v_cache_snap)
        if sync is not None:
            sync()

    # ------------------------------------------------------------------
    # Build descriptors. The env var is read inside the create call only,
    # so we toggle, create, toggle, create. Both descriptors are then
    # frozen and the env var no longer matters for the rest of the case.
    # ------------------------------------------------------------------
    os.environ.pop(_ENV_KILL, None)  # ensure fused path engages
    fused_desc, fused_ws, fused_ws_size = _build_attention_descriptor(
        handle, device, out_fused, q, k, v, k_cache, v_cache, pos
    )

    os.environ[_ENV_KILL] = "1"  # force fallback path
    base_desc, base_ws, base_ws_size = _build_attention_descriptor(
        handle, device, out_base, q, k, v, k_cache, v_cache, pos
    )
    os.environ.pop(_ENV_KILL, None)  # reset (no longer matters but tidy)

    def run_fused():
        check_error(
            LIBINFINIOP.infiniopAttention(
                fused_desc,
                fused_ws.data(),
                fused_ws_size.value,
                out_fused.data(),
                q.data(),
                k.data(),
                v.data(),
                k_cache.data(),
                v_cache.data(),
                None,
            )
        )

    def run_base():
        check_error(
            LIBINFINIOP.infiniopAttention(
                base_desc,
                base_ws.data(),
                base_ws_size.value,
                out_base.data(),
                q.data(),
                k.data(),
                v.data(),
                k_cache.data(),
                v_cache.data(),
                None,
            )
        )

    # ------------------------------------------------------------------
    # Correctness gate.
    # ------------------------------------------------------------------
    atol = _TOLERANCE_MAP[dtype]["atol"]
    rtol = _TOLERANCE_MAP[dtype]["rtol"]

    reset_caches()
    run_fused()
    if sync is not None:
        sync()
    fused_ok = torch.allclose(out_fused.actual_tensor(), ref, atol=atol, rtol=rtol)

    reset_caches()
    run_base()
    if sync is not None:
        sync()
    base_ok = torch.allclose(out_base.actual_tensor(), ref, atol=atol, rtol=rtol)

    # Cross-check: fused vs baseline should agree at the same tolerance
    fb_ok = torch.allclose(
        out_fused.actual_tensor(), out_base.actual_tensor(), atol=atol, rtol=rtol
    )

    if not fused_ok:
        print("  [WARN] fused output disagrees with PyTorch reference.")
    if not base_ok:
        print("  [WARN] baseline output disagrees with PyTorch reference.")
    if not fb_ok:
        print("  [WARN] fused vs baseline disagree (different reduction order is normal at F16).")

    result = None
    if fused_ok and base_ok:
        # Timed regions. Cache is repeatedly overwritten with identical
        # bytes by both paths, so no drift between iters.
        reset_caches()
        warmup(run_fused, NUM_PRERUN)
        fused_mean, fused_std = timed_loop(run_fused, NUM_ITERATIONS, device)

        reset_caches()
        warmup(run_base, NUM_PRERUN)
        base_mean, base_std = timed_loop(run_base, NUM_ITERATIONS, device)

        # PyTorch as informational reference only; not on the speedup line.
        def run_torch():
            _ = attention_ref(
                q.torch_tensor(),
                k.torch_tensor(),
                v.torch_tensor(),
                k_cache_snap,
                v_cache_snap,
                pos,
            )
        warmup(run_torch, max(3, NUM_PRERUN // 4))
        torch_mean, _ = timed_loop(run_torch, max(10, NUM_ITERATIONS // 4), device)

        ms = lambda s: s * 1e3
        speedup = base_mean / fused_mean if fused_mean > 0 else float("nan")
        delta_pct = (speedup - 1) * 100 if math.isfinite(speedup) else float("nan")
        print(f"  baseline (unfused 4-stage) : "
              f"{ms(base_mean):9.4f} ms  (stddev {ms(base_std):.4f})")
        print(f"  fused (iluvatar kernel)    : "
              f"{ms(fused_mean):9.4f} ms  (stddev {ms(fused_std):.4f})")
        print(f"  pytorch (ref only)         : "
              f"{ms(torch_mean):9.4f} ms")
        print(f"  speedup vs baseline        : "
              f"{speedup:6.3f}x  ({delta_pct:+.2f}%)")
        result = (
            case_label(n_q, n_kv, seq_len, head_dim, pos),
            dtype_name,
            ms(base_mean),
            ms(fused_mean),
            ms(torch_mean),
            speedup,
        )

    check_error(LIBINFINIOP.infiniopDestroyAttentionDescriptor(fused_desc))
    check_error(LIBINFINIOP.infiniopDestroyAttentionDescriptor(base_desc))
    return result


# ------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------
def _print_preamble():
    print("=" * 78)
    print(" Iluvatar fused Attention performance benchmark")
    print("=" * 78)
    print(" Fused     = single op::attention::iluvatar::fused_attention kernel")
    print(" Baseline  = unfused 4-stage path inside the same operator:")
    print("             rearrange(k->cache) + rearrange(v->cache)")
    print("             + gemm(q*K) + causal_softmax + gemm(weights*V)")
    print("             + rearrange(att_val->out)")
    print(" Toggle    = INFINIOP_DISABLE_ILUVATAR_FUSED_ATTENTION around create")
    print(" Both candidates:")
    print("   - share q / k / v / k_cache / v_cache tensors by pointer")
    print("   - run via infiniopAttention (same API, different descriptor)")
    print("   - timed per-call with synchronize_device() before / after")
    print(f"   - warmup={NUM_PRERUN} calls, repeat={NUM_ITERATIONS} calls")
    print(" Fused gate (operator.cc::can_use_iluvatar_fused_attention):")
    print("   dtype=F16, n_q==n_kv, head_dim<=128, seq_len+pos<=8, contiguous")
    print("=" * 78)


def _print_summary(rows):
    if not rows:
        return
    print()
    print("=" * 78)
    print(" Summary")
    print("=" * 78)
    print(f" {'case':<46s} {'dtype':<5s} {'base':>10s} {'fused':>10s} {'speedup':>10s}")
    print(" " + "-" * 75)
    for label, dt, base_ms, fused_ms, _torch_ms, sp in rows:
        short = label if len(label) <= 46 else label[:43] + "..."
        delta_pct = (sp - 1) * 100
        print(f" {short:<46s} {dt:<5s} {base_ms:>9.4f}m {fused_ms:>9.4f}m "
              f"{sp:>6.3f}x {delta_pct:+6.1f}%")


def main():
    global NUM_PRERUN, NUM_ITERATIONS

    args = get_args()
    if args.num_prerun != 10:
        NUM_PRERUN = args.num_prerun
    if args.num_iterations != 1000:
        NUM_ITERATIONS = args.num_iterations

    devices = get_test_devices(args)
    if not devices:
        print("No device selected. Pass --iluvatar (this benchmark targets ivcore11).")
        return

    _print_preamble()

    rows = []
    for device in devices:
        # The fused path only engages on Iluvatar; on other devices both
        # descriptors land on the same (non-iluvatar) code path and the
        # speedup ratio collapses to ~1.0x. Leave the warning in so a
        # mis-flagged run is visible.
        if "iluvatar" not in InfiniDeviceNames[device].lower():
            print(f"\n[WARN] device {InfiniDeviceNames[device]}: fused gate is "
                  f"iluvatar-only; speedup will read ~1.0x on this device.")
        print()
        print(f"### device: {InfiniDeviceNames[device]} ###")
        LIBINFINIOP.infinirtSetDevice(device, ctypes.c_int(0))
        handle = create_handle()
        sync = get_sync_func(device)
        try:
            for (n_q, n_kv, seq_len, head_dim, pos) in _BENCH_CASES:
                for dtype in _BENCH_DTYPES:
                    r = bench_one_case(
                        handle, device, n_q, n_kv, seq_len, head_dim, pos, dtype, sync
                    )
                    if r is not None:
                        rows.append(r)
        finally:
            destroy_handle(handle)

    _print_summary(rows)


if __name__ == "__main__":
    main()
