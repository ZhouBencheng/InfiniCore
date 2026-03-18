"""
Fused FFN Performance Benchmark

Compares fused FFN operator vs non-fused PyTorch implementation.
Goal: Find the crossover point where fused becomes faster than non-fused.
"""

import torch
import torch.nn.functional as F
import ctypes
from ctypes import c_uint64
from datetime import datetime
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    get_args,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
    create_handle,
    destroy_handle,
)

# ==============================================================================
#  Configuration
# ==============================================================================

BATCH_SIZES = [
    1, 2, 4,          # Very small batches
    8, 16, 32,        # Small batches
    64, 128, 256,     # Medium batches
    512, 1024,        # Large batches
]

MODEL_CONFIGS = [
    # (hidden_dim, intermediate_dim, name)
    (2048, 5632, "Small"),
    (3584, 18944, "Qwen-7B"),
    (4096, 11008, "LLaMA-7B"),
    (5120, 13824, "LLaMA-13B"),
]

DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]
RESIDUAL_OPTIONS = [True, False]

NUM_WARMUP = 10
NUM_ITERATIONS = 100
EPSILON = 1e-6


# ==============================================================================
#  Non-Fused Implementation (PyTorch)
# ==============================================================================

def nonfused_ffn_pytorch(x, residual, norm_w, gate_up_w, down_w, eps):
    """
    Non-fused FFN using PyTorch native operators.
    This represents the baseline performance without fusion.
    """
    # Stage 1: RMSNorm
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + eps)
    normalized = (normalized * norm_w.float()).to(x.dtype)

    # Stage 2: GateUp projection
    gate_up = F.linear(normalized, gate_up_w)
    di = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :di], gate_up[..., di:]

    # Stage 3: SwiGLU
    hidden = F.silu(gate) * up

    # Stage 4: Down projection
    out = F.linear(hidden, down_w)

    # Stage 5: Residual add
    if residual is not None:
        out = out + residual

    return out


# ==============================================================================
#  Fused Implementation (InfiniOP)
# ==============================================================================

class FusedFFNOperator:
    """Wrapper for fused FFN operator."""

    def __init__(self, handle, batch, hidden_dim, intermediate_dim, dtype, has_residual, device):
        self.batch = batch
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dtype = dtype
        self.has_residual = has_residual
        self.device = device

        # Create output tensor
        self.out = TestTensor((batch, hidden_dim), None, dtype, device)

        # Create input tensors (will be set during call)
        self.x = None
        self.residual = None
        self.norm_w = None
        self.gate_up_w = None
        self.down_w = None

        # Create descriptor
        self.descriptor = infiniopOperatorDescriptor_t()

        # Dummy tensors for descriptor creation
        dummy_x = TestTensor((batch, hidden_dim), None, dtype, device)
        dummy_residual = TestTensor((batch, hidden_dim), None, dtype, device) if has_residual else None
        dummy_norm_w = TestTensor((hidden_dim,), None, dtype, device)
        dummy_gate_up_w = TestTensor((2 * intermediate_dim, hidden_dim), None, dtype, device)
        dummy_down_w = TestTensor((hidden_dim, intermediate_dim), None, dtype, device)

        check_error(
            LIBINFINIOP.infiniopCreateFusedFFNDescriptor(
                handle,
                ctypes.byref(self.descriptor),
                self.out.descriptor,
                dummy_x.descriptor,
                dummy_residual.descriptor if dummy_residual else None,
                dummy_norm_w.descriptor,
                dummy_gate_up_w.descriptor,
                dummy_down_w.descriptor,
                ctypes.c_float(EPSILON),
            )
        )

        # Get workspace size
        workspace_size = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetFusedFFNWorkspaceSize(
                self.descriptor, ctypes.byref(workspace_size)
            )
        )
        self.workspace = TestWorkspace(workspace_size.value, device)

        # Destroy dummy descriptors
        dummy_x.destroy_desc()
        if dummy_residual:
            dummy_residual.destroy_desc()
        dummy_norm_w.destroy_desc()
        dummy_gate_up_w.destroy_desc()
        dummy_down_w.destroy_desc()

    def __call__(self, x, residual, norm_w, gate_up_w, down_w):
        """Execute fused FFN."""
        check_error(
            LIBINFINIOP.infiniopFusedFFN(
                self.descriptor,
                self.workspace.data(),
                self.workspace.size(),
                self.out.data(),
                x.data(),
                residual.data() if residual else None,
                norm_w.data(),
                gate_up_w.data(),
                down_w.data(),
                None,
            )
        )
        return self.out

    def destroy(self):
        """Destroy descriptor."""
        check_error(LIBINFINIOP.infiniopDestroyFusedFFNDescriptor(self.descriptor))


# ==============================================================================
#  Benchmark Functions
# ==============================================================================

def benchmark_cuda_event(func, num_warmup, num_iterations, torch_device):
    """Benchmark using CUDA events for high precision timing."""
    # Warmup
    for _ in range(num_warmup):
        func()

    if torch_device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_iterations):
            func()
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end) / num_iterations
    else:
        # CPU fallback using time
        import time
        start = time.time()
        for _ in range(num_iterations):
            func()
        end = time.time()
        return (end - start) * 1000 / num_iterations


def calculate_memory_traffic(batch, hidden_dim, intermediate_dim, dtype_size):
    """Calculate theoretical memory traffic in bytes."""
    input_traffic = batch * hidden_dim * dtype_size
    norm_w_traffic = hidden_dim * dtype_size
    gate_up_w_traffic = 2 * intermediate_dim * hidden_dim * dtype_size
    gate_up_traffic = batch * 2 * intermediate_dim * dtype_size
    down_w_traffic = hidden_dim * intermediate_dim * dtype_size
    output_traffic = batch * hidden_dim * dtype_size

    fused_traffic = input_traffic + norm_w_traffic + gate_up_w_traffic + down_w_traffic + output_traffic
    nonfused_traffic = fused_traffic + gate_up_traffic

    return fused_traffic, nonfused_traffic


# ==============================================================================
#  Main Benchmark
# ==============================================================================

def run_benchmark(handle, device, batch, hidden_dim, intermediate_dim, dtype, has_residual):
    """Run benchmark for a single configuration."""
    torch_device = "cuda" if "nvidia" in InfiniDeviceNames[device].lower() else "cpu"
    dtype_size = 2 if dtype in [InfiniDtype.F16, InfiniDtype.BF16] else 4

    # Weight scaling to avoid overflow
    weight_scale = 1.0 / (hidden_dim ** 0.5)

    # Create tensors
    x = TestTensor((batch, hidden_dim), None, dtype, device)
    residual = TestTensor((batch, hidden_dim), None, dtype, device) if has_residual else None
    norm_w = TestTensor((hidden_dim,), None, dtype, device)
    gate_up_w = TestTensor((2 * intermediate_dim, hidden_dim), None, dtype, device, scale=weight_scale)
    down_w = TestTensor((hidden_dim, intermediate_dim), None, dtype, device, scale=weight_scale)

    # Create fused operator
    fused_op = FusedFFNOperator(handle, batch, hidden_dim, intermediate_dim, dtype, has_residual, device)

    # Benchmark fused
    fused_time = benchmark_cuda_event(
        lambda: fused_op(x, residual, norm_w, gate_up_w, down_w),
        NUM_WARMUP, NUM_ITERATIONS, torch_device
    )

    # Benchmark non-fused (PyTorch)
    x_torch = x.torch_tensor()
    residual_torch = residual.torch_tensor() if residual else None
    norm_w_torch = norm_w.torch_tensor()
    gate_up_w_torch = gate_up_w.torch_tensor()
    down_w_torch = down_w.torch_tensor()

    nonfused_time = benchmark_cuda_event(
        lambda: nonfused_ffn_pytorch(x_torch, residual_torch, norm_w_torch, gate_up_w_torch, down_w_torch, EPSILON),
        NUM_WARMUP, NUM_ITERATIONS, torch_device
    )

    # Calculate metrics
    speedup = nonfused_time / fused_time if fused_time > 0 else 0
    fused_traffic, nonfused_traffic = calculate_memory_traffic(batch, hidden_dim, intermediate_dim, dtype_size)
    fused_bw = (fused_traffic / fused_time / 1e6) if fused_time > 0 else 0  # GB/s
    nonfused_bw = (nonfused_traffic / nonfused_time / 1e6) if nonfused_time > 0 else 0  # GB/s

    # Cleanup
    fused_op.destroy()

    return {
        "batch": batch,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "dtype": InfiniDtypeNames[dtype],
        "has_residual": has_residual,
        "fused_ms": fused_time,
        "nonfused_ms": nonfused_time,
        "speedup": speedup,
        "winner": "Fused" if speedup > 1.0 else "NonFused",
        "fused_bw_gb_s": fused_bw,
        "nonfused_bw_gb_s": nonfused_bw,
    }


def main():
    args = get_args()

    # Parse additional arguments
    batch_sizes = BATCH_SIZES
    model_configs = MODEL_CONFIGS

    if hasattr(args, 'batch_sizes') and args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    if hasattr(args, 'model') and args.model:
        model_map = {c[2].lower(): c for c in MODEL_CONFIGS}
        if args.model.lower() in model_map:
            model_configs = [model_map[args.model.lower()]]

    # Collect results
    results = []

    print("=" * 80)
    print("Fused FFN Performance Benchmark")
    print("=" * 80)
    print(f"Device: NVIDIA GPU")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Warmup: {NUM_WARMUP}, Iterations: {NUM_ITERATIONS}")
    print("=" * 80)
    print()

    for device in get_test_devices(args):
        # Set device and create handle
        LIBINFINIOP.infinirtSetDevice(device, ctypes.c_int(0))
        handle = create_handle()

        for hidden_dim, intermediate_dim, model_name in model_configs:
            for dtype in DTYPES:
                for has_residual in RESIDUAL_OPTIONS:
                    for batch in batch_sizes:
                        print(f"[{model_name}] Batch={batch}, Hidden={hidden_dim}, Inter={intermediate_dim}, "
                              f"{InfiniDtypeNames[dtype]}, Residual={has_residual}")

                        try:
                            result = run_benchmark(
                                handle, device, batch, hidden_dim, intermediate_dim, dtype, has_residual
                            )
                            result["model"] = model_name
                            results.append(result)

                            print(f"  Fused:    {result['fused_ms']:.4f} ms")
                            print(f"  NonFused: {result['nonfused_ms']:.4f} ms")
                            print(f"  Speedup:  {result['speedup']:.2f}x ({result['winner']} faster)")
                            print()
                        except Exception as e:
                            import traceback
                            print(f"  ERROR: {e}")
                            traceback.print_exc()
                            print()

        # Destroy handle
        destroy_handle(handle)

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    if results:
        # Find crossover points
        fused_wins = [r for r in results if r['speedup'] > 1.0]
        nonfused_wins = [r for r in results if r['speedup'] <= 1.0]

        print(f"Total tests: {len(results)}")
        print(f"Fused faster: {len(fused_wins)} ({100*len(fused_wins)/len(results):.1f}%)")
        print(f"NonFused faster: {len(nonfused_wins)} ({100*len(nonfused_wins)/len(results):.1f}%)")

        if fused_wins:
            best = max(fused_wins, key=lambda x: x['speedup'])
            print(f"Best fused speedup: {best['speedup']:.2f}x @ batch={best['batch']}, model={best['model']}")

        # Find crossover by model
        print("\nCrossover points by model:")
        for model_name in [c[2] for c in model_configs]:
            model_results = [r for r in results if r['model'] == model_name and r['dtype'] == 'F16' and r['has_residual']]
            if model_results:
                model_results.sort(key=lambda x: x['batch'])
                crossover = None
                for i, r in enumerate(model_results):
                    if r['speedup'] > 1.0:
                        crossover = r['batch']
                        break
                if crossover:
                    print(f"  {model_name}: batch_size >= {crossover}")
                else:
                    print(f"  {model_name}: No crossover (NonFused always faster)")

    # CSV output
    if hasattr(args, 'output') and args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"\nResults saved to {args.output}")

    print("\033[92mBenchmark completed!\033[0m")


if __name__ == "__main__":
    import sys

    # Parse custom arguments before standard args
    batch_sizes_arg = None
    model_arg = None
    output_arg = None
    plot_arg = False

    # Remove custom args from sys.argv
    filtered_args = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '--batch_sizes' and i + 1 < len(sys.argv):
            batch_sizes_arg = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--model' and i + 1 < len(sys.argv):
            model_arg = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output_arg = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--plot':
            plot_arg = True
            i += 1
        else:
            filtered_args.append(sys.argv[i])
            i += 1

    sys.argv = filtered_args

    # Get standard args
    args = get_args()

    # Add custom args
    args.batch_sizes = batch_sizes_arg
    args.model = model_arg
    args.output = output_arg
    args.plot = plot_arg

    main()
