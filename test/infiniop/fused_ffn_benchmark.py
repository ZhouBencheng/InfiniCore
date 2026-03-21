"""
Fused FFN Performance Benchmark

Compares fused FFN operator vs non-fused InfiniCore single operators.
Goal: Find the crossover point where fused becomes faster than non-fused.

How to test on Nvidia:
    /root/miniconda3/bin/python test/infiniop/fused_ffn_benchmark.py --nvidia

Additional Options:
    --report <file.md>    Generate structured Markdown report for human reading
    --output <file.csv>   Export raw data as CSV
    --batch_sizes <list>  Comma-separated batch sizes (e.g., "1,4,16,64")
    --model <name>        Test specific model (e.g., "LLaMA-7B", "Qwen-7B")

Example:
    /root/miniconda3/bin/python test/infiniop/fused_ffn_benchmark.py --nvidia \\
        --report benchmark_report.md --output benchmark_data.csv
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
    infiniopTensorDescriptor_t,
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
#  Non-Fused Implementation (InfiniCore Single Operators)
# ==============================================================================

class StridedView:
    """A strided view into an existing tensor with optional offset."""

    def __init__(self, base_tensor, shape, strides, offset_elements, dtype, device):
        self.descriptor = infiniopTensorDescriptor_t()
        self.ndims = len(shape)
        self.c_shape = (ctypes.c_size_t * self.ndims)(*shape)
        self.c_strides = (ctypes.c_ssize_t * self.ndims)(*strides)
        self.dt = dtype
        self.device = device

        LIBINFINIOP.infiniopCreateTensorDescriptor(
            ctypes.byref(self.descriptor),
            self.ndims,
            self.c_shape,
            self.c_strides,
            dtype,
        )

        # Calculate data pointer with offset
        dtype_size = 2 if dtype in [InfiniDtype.F16, InfiniDtype.BF16] else 4
        self._data_ptr = base_tensor.data() + offset_elements * dtype_size

    def data(self):
        return self._data_ptr

    def destroy_desc(self):
        if self.descriptor is not None:
            LIBINFINIOP.infiniopDestroyTensorDescriptor(self.descriptor)
            self.descriptor = None


class NonFusedFFNOperator:
    """Non-fused FFN using individual InfiniCore operators."""

    def __init__(self, handle, batch, hidden_dim, intermediate_dim, dtype, has_residual, device):
        self.batch = batch
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dtype = dtype
        self.has_residual = has_residual
        self.device = device

        # Create intermediate tensors
        # normalized: [batch, hidden_dim] - RMSNorm output
        self.normalized = TestTensor((batch, hidden_dim), None, dtype, device, mode="zeros")

        # gate_up: [batch, 2*intermediate_dim] - GateUp GEMM output
        self.gate_up = TestTensor((batch, 2 * intermediate_dim), None, dtype, device, mode="zeros")

        # hidden: [batch, intermediate_dim] - SwiGLU output
        self.hidden_tensor = TestTensor((batch, intermediate_dim), None, dtype, device, mode="zeros")

        # down_out: [batch, hidden_dim] - Down GEMM output
        self.down_out = TestTensor((batch, hidden_dim), None, dtype, device, mode="zeros")

        # Create strided views for gate and up (split gate_up tensor)
        # gate: view of gate_up[:, :intermediate_dim] - offset 0
        # up: view of gate_up[:, intermediate_dim:] - offset intermediate_dim
        gate_shape = (batch, intermediate_dim)
        gate_strides = (2 * intermediate_dim, 1)  # Same strides as gate_up
        self.gate_view = StridedView(
            self.gate_up, gate_shape, gate_strides, 0, dtype, device
        )

        up_shape = (batch, intermediate_dim)
        up_strides = (2 * intermediate_dim, 1)
        up_offset = intermediate_dim
        self.up_view = StridedView(
            self.gate_up, up_shape, up_strides, up_offset, dtype, device
        )

        # =====================================================================
        #  Stage 1: RMSNorm descriptor
        # =====================================================================
        self.rms_norm_desc = infiniopOperatorDescriptor_t()

        # Create dummy tensors for descriptor creation
        dummy_y = TestTensor((batch, hidden_dim), None, dtype, device)
        dummy_x = TestTensor((batch, hidden_dim), None, dtype, device)
        dummy_w = TestTensor((hidden_dim,), None, dtype, device)

        check_error(
            LIBINFINIOP.infiniopCreateRMSNormDescriptor(
                handle,
                ctypes.byref(self.rms_norm_desc),
                dummy_y.descriptor,
                dummy_x.descriptor,
                dummy_w.descriptor,
                ctypes.c_float(EPSILON),
            )
        )

        # Get RMSNorm workspace
        rms_workspace_size = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetRMSNormWorkspaceSize(
                self.rms_norm_desc, ctypes.byref(rms_workspace_size)
            )
        )
        self.rms_workspace = TestWorkspace(rms_workspace_size.value, device)

        dummy_y.destroy_desc()
        dummy_x.destroy_desc()
        dummy_w.destroy_desc()

        # =====================================================================
        #  Stage 2: GateUp GEMM descriptor
        #  C = A @ B where A=normalized [batch, hidden_dim], B=gate_up_w.T [hidden_dim, 2*intermediate_dim]
        # =====================================================================
        self.gate_up_gemm_desc = infiniopOperatorDescriptor_t()

        # Create dummy tensor descriptors for GEMM
        dummy_c = TestTensor((batch, 2 * intermediate_dim), None, dtype, device, mode="zeros")
        dummy_a = TestTensor((batch, hidden_dim), None, dtype, device, mode="zeros")
        dummy_b = TestTensor((hidden_dim, 2 * intermediate_dim), None, dtype, device, mode="zeros")

        check_error(
            LIBINFINIOP.infiniopCreateGemmDescriptor(
                handle,
                ctypes.byref(self.gate_up_gemm_desc),
                dummy_c.descriptor,
                dummy_a.descriptor,
                dummy_b.descriptor,
            )
        )

        # Get GEMM workspace
        gate_up_workspace_size = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetGemmWorkspaceSize(
                self.gate_up_gemm_desc, ctypes.byref(gate_up_workspace_size)
            )
        )
        self.gate_up_workspace = TestWorkspace(gate_up_workspace_size.value, device)

        dummy_c.destroy_desc()
        dummy_a.destroy_desc()
        dummy_b.destroy_desc()

        # =====================================================================
        #  Stage 3: SwiGLU descriptor
        # =====================================================================
        self.swiglu_desc = infiniopOperatorDescriptor_t()
        check_error(
            LIBINFINIOP.infiniopCreateSwiGLUDescriptor(
                handle,
                ctypes.byref(self.swiglu_desc),
                self.hidden_tensor.descriptor,
                self.gate_view.descriptor,
                self.up_view.descriptor,
            )
        )

        # Get SwiGLU workspace
        swiglu_workspace_size = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetSwiGLUWorkspaceSize(
                self.swiglu_desc, ctypes.byref(swiglu_workspace_size)
            )
        )
        self.swiglu_workspace = TestWorkspace(swiglu_workspace_size.value, device)

        # =====================================================================
        #  Stage 4: Down GEMM descriptor
        #  C = A @ B where A=hidden [batch, intermediate_dim], B=down_w.T [intermediate_dim, hidden_dim]
        # =====================================================================
        self.down_gemm_desc = infiniopOperatorDescriptor_t()

        # Create dummy tensor descriptors for Down GEMM
        dummy_c2 = TestTensor((batch, hidden_dim), None, dtype, device, mode="zeros")
        dummy_a2 = TestTensor((batch, intermediate_dim), None, dtype, device, mode="zeros")
        dummy_b2 = TestTensor((intermediate_dim, hidden_dim), None, dtype, device, mode="zeros")

        check_error(
            LIBINFINIOP.infiniopCreateGemmDescriptor(
                handle,
                ctypes.byref(self.down_gemm_desc),
                dummy_c2.descriptor,
                dummy_a2.descriptor,
                dummy_b2.descriptor,
            )
        )

        # Get Down GEMM workspace
        down_workspace_size = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetGemmWorkspaceSize(
                self.down_gemm_desc, ctypes.byref(down_workspace_size)
            )
        )
        self.down_workspace = TestWorkspace(down_workspace_size.value, device)

        dummy_c2.destroy_desc()
        dummy_a2.destroy_desc()
        dummy_b2.destroy_desc()

        # =====================================================================
        #  Stage 5: Residual Add descriptor (optional)
        # =====================================================================
        if has_residual:
            self.add_desc = infiniopOperatorDescriptor_t()

            dummy_out = TestTensor((batch, hidden_dim), None, dtype, device, mode="zeros")
            dummy_residual = TestTensor((batch, hidden_dim), None, dtype, device, mode="zeros")

            check_error(
                LIBINFINIOP.infiniopCreateAddDescriptor(
                    handle,
                    ctypes.byref(self.add_desc),
                    dummy_out.descriptor,
                    dummy_out.descriptor,  # down_out
                    dummy_residual.descriptor,
                )
            )

            # Get Add workspace
            add_workspace_size = c_uint64(0)
            check_error(
                LIBINFINIOP.infiniopGetAddWorkspaceSize(
                    self.add_desc, ctypes.byref(add_workspace_size)
                )
            )
            self.add_workspace = TestWorkspace(add_workspace_size.value, device)

            dummy_out.destroy_desc()
            dummy_residual.destroy_desc()

            # Output tensor for residual add case
            self.out = TestTensor((batch, hidden_dim), None, dtype, device, mode="zeros")
        else:
            # No residual, output is down_out directly
            self.out = self.down_out
            self.add_desc = None

    def __call__(self, x, residual, norm_w, gate_up_w, down_w):
        """Execute the 5-stage non-fused FFN pipeline."""
        # Create transposed weight views for this call
        # These need to point to the actual weight tensors
        gate_up_w_t_shape = (self.hidden_dim, 2 * self.intermediate_dim)
        gate_up_w_t_strides = (1, self.hidden_dim)  # Transposed strides
        gate_up_w_t_view = StridedView(
            gate_up_w, gate_up_w_t_shape, gate_up_w_t_strides, 0, self.dtype, self.device
        )

        down_w_t_shape = (self.intermediate_dim, self.hidden_dim)
        down_w_t_strides = (1, self.intermediate_dim)
        down_w_t_view = StridedView(
            down_w, down_w_t_shape, down_w_t_strides, 0, self.dtype, self.device
        )

        # Stage 1: RMSNorm
        check_error(
            LIBINFINIOP.infiniopRMSNorm(
                self.rms_norm_desc,
                self.rms_workspace.data(),
                self.rms_workspace.size(),
                self.normalized.data(),
                x.data(),
                norm_w.data(),
                None,
            )
        )

        # Stage 2: GateUp GEMM
        check_error(
            LIBINFINIOP.infiniopGemm(
                self.gate_up_gemm_desc,
                self.gate_up_workspace.data(),
                self.gate_up_workspace.size(),
                self.gate_up.data(),
                self.normalized.data(),
                gate_up_w_t_view.data(),
                ctypes.c_float(1.0),
                ctypes.c_float(0.0),
                None,
            )
        )

        # Stage 3: SwiGLU
        check_error(
            LIBINFINIOP.infiniopSwiGLU(
                self.swiglu_desc,
                self.swiglu_workspace.data(),
                self.swiglu_workspace.size(),
                self.hidden_tensor.data(),
                self.gate_view.data(),
                self.up_view.data(),
                None,
            )
        )

        # Stage 4: Down GEMM
        check_error(
            LIBINFINIOP.infiniopGemm(
                self.down_gemm_desc,
                self.down_workspace.data(),
                self.down_workspace.size(),
                self.down_out.data(),
                self.hidden_tensor.data(),
                down_w_t_view.data(),
                ctypes.c_float(1.0),
                ctypes.c_float(0.0),
                None,
            )
        )

        # Stage 5: Residual Add (optional)
        if self.has_residual:
            check_error(
                LIBINFINIOP.infiniopAdd(
                    self.add_desc,
                    self.add_workspace.data(),
                    self.add_workspace.size(),
                    self.out.data(),
                    self.down_out.data(),
                    residual.data(),
                    None,
                )
            )

        # Cleanup temporary views
        gate_up_w_t_view.destroy_desc()
        down_w_t_view.destroy_desc()

        return self.out

    def destroy(self):
        """Destroy all descriptors and views."""
        # Destroy operator descriptors
        check_error(LIBINFINIOP.infiniopDestroyRMSNormDescriptor(self.rms_norm_desc))
        check_error(LIBINFINIOP.infiniopDestroyGemmDescriptor(self.gate_up_gemm_desc))
        check_error(LIBINFINIOP.infiniopDestroySwiGLUDescriptor(self.swiglu_desc))
        check_error(LIBINFINIOP.infiniopDestroyGemmDescriptor(self.down_gemm_desc))
        if self.add_desc is not None:
            check_error(LIBINFINIOP.infiniopDestroyAddDescriptor(self.add_desc))

        # Destroy strided views
        self.gate_view.destroy_desc()
        self.up_view.destroy_desc()

        # Destroy intermediate tensor descriptors
        self.normalized.destroy_desc()
        self.gate_up.destroy_desc()
        self.hidden_tensor.destroy_desc()
        self.down_out.destroy_desc()
        if self.has_residual:
            self.out.destroy_desc()


# Keep PyTorch reference for validation only (not for timing)
def nonfused_ffn_pytorch_ref(x, residual, norm_w, gate_up_w, down_w, eps):
    """
    PyTorch reference implementation for validation purposes only.
    Not used for benchmarking.
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

    # Create non-fused operator (using InfiniCore single operators)
    nonfused_op = NonFusedFFNOperator(handle, batch, hidden_dim, intermediate_dim, dtype, has_residual, device)

    # Benchmark non-fused (InfiniCore single operators)
    nonfused_time = benchmark_cuda_event(
        lambda: nonfused_op(x, residual, norm_w, gate_up_w, down_w),
        NUM_WARMUP, NUM_ITERATIONS, torch_device
    )

    # Calculate metrics
    speedup = nonfused_time / fused_time if fused_time > 0 else 0
    fused_traffic, nonfused_traffic = calculate_memory_traffic(batch, hidden_dim, intermediate_dim, dtype_size)
    fused_bw = (fused_traffic / fused_time / 1e6) if fused_time > 0 else 0  # GB/s
    nonfused_bw = (nonfused_traffic / nonfused_time / 1e6) if nonfused_time > 0 else 0  # GB/s

    # Cleanup
    fused_op.destroy()
    nonfused_op.destroy()

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


# ==============================================================================
#  Report Generation
# ==============================================================================

def generate_structured_report(results, output_path, model_configs):
    """Generate a structured Markdown report for human readability."""
    from datetime import datetime

    lines = []

    # Header
    lines.append("# Fused FFN Performance Benchmark Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Tests:** {len(results)}")
    lines.append(f"**Warmup Iterations:** {NUM_WARMUP}")
    lines.append(f"**Benchmark Iterations:** {NUM_ITERATIONS}")
    lines.append("")

    # Summary Statistics
    lines.append("---")
    lines.append("")
    lines.append("## 📊 Summary Statistics")
    lines.append("")

    fused_wins = [r for r in results if r['speedup'] > 1.0]
    nonfused_wins = [r for r in results if r['speedup'] <= 1.0]

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Fused Faster | {len(fused_wins)} ({100*len(fused_wins)/len(results):.1f}%) |")
    lines.append(f"| NonFused Faster | {len(nonfused_wins)} ({100*len(nonfused_wins)/len(results):.1f}%) |")

    if fused_wins:
        best = max(fused_wins, key=lambda x: x['speedup'])
        lines.append(f"| Best Fused Speedup | **{best['speedup']:.2f}x** (batch={best['batch']}, model={best['model']}) |")

    if nonfused_wins:
        worst = min(fused_wins, key=lambda x: x['speedup']) if fused_wins else None
        if worst:
            lines.append(f"| Worst Fused Speedup | {worst['speedup']:.2f}x (batch={worst['batch']}, model={worst['model']}) |")

    lines.append("")

    # Crossover Analysis
    lines.append("---")
    lines.append("")
    lines.append("## 🔍 Crossover Analysis")
    lines.append("")
    lines.append("> Crossover point: The batch size where fused FFN becomes faster than non-fused.")
    lines.append("")
    lines.append("| Model | Dtype | Residual | Crossover Batch |")
    lines.append("|-------|-------|----------|-----------------|")

    model_names = sorted(set(r['model'] for r in results))
    for model_name in model_names:
        for dtype in DTYPES:
            dtype_name = InfiniDtypeNames[dtype]
            for has_residual in [True, False]:
                model_results = [r for r in results
                               if r['model'] == model_name
                               and r['dtype'] == dtype_name
                               and r['has_residual'] == has_residual]
                if model_results:
                    model_results.sort(key=lambda x: x['batch'])
                    crossover = None
                    for r in model_results:
                        if r['speedup'] > 1.0:
                            crossover = r['batch']
                            break
                    crossover_str = f"**≥ {crossover}**" if crossover else "Never"
                    lines.append(f"| {model_name} | {dtype_name} | {has_residual} | {crossover_str} |")

    lines.append("")

    # Detailed Results by Model
    lines.append("---")
    lines.append("")
    lines.append("## 📈 Detailed Results")
    lines.append("")

    for model_name in model_names:
        model_results = [r for r in results if r['model'] == model_name]
        if not model_results:
            continue

        hidden_dim = model_results[0]['hidden_dim']
        intermediate_dim = model_results[0]['intermediate_dim']

        lines.append(f"### {model_name}")
        lines.append("")
        lines.append(f"- **Hidden Dim:** {hidden_dim}")
        lines.append(f"- **Intermediate Dim:** {intermediate_dim}")
        lines.append("")

        # Group by dtype and residual
        for dtype_name in ['F16', 'BF16']:
            for has_residual in [True, False]:
                subset = [r for r in model_results
                         if r['dtype'] == dtype_name and r['has_residual'] == has_residual]
                if not subset:
                    continue

                subset.sort(key=lambda x: x['batch'])
                residual_str = "with Residual" if has_residual else "w/o Residual"

                lines.append(f"#### {dtype_name} {residual_str}")
                lines.append("")
                lines.append("| Batch | Fused (ms) | NonFused (ms) | Speedup | Winner | BW (GB/s) |")
                lines.append("|-------|------------|---------------|---------|--------|-----------|")

                for r in subset:
                    winner_emoji = "🟢" if r['winner'] == 'Fused' else "🔴"
                    lines.append(f"| {r['batch']} | {r['fused_ms']:.4f} | {r['nonfused_ms']:.4f} | "
                               f"**{r['speedup']:.2f}x** | {winner_emoji} {r['winner']} | "
                               f"{r['fused_bw_gb_s']:.1f} / {r['nonfused_bw_gb_s']:.1f} |")

                lines.append("")

    # Performance Heatmap (Text-based)
    lines.append("---")
    lines.append("")
    lines.append("## 🎯 Performance Matrix")
    lines.append("")
    lines.append("Speedup values (Fused vs NonFused): **>1.0** means Fused is faster.")
    lines.append("")

    # Create a matrix view
    batch_sizes_used = sorted(set(r['batch'] for r in results))
    lines.append("| Model \\ Batch | " + " | ".join(str(b) for b in batch_sizes_used) + " |")
    lines.append("|" + "|".join(["---"] * (len(batch_sizes_used) + 1)) + "|")

    for model_name in model_names:
        row_values = []
        for batch in batch_sizes_used:
            # Find matching result (prefer F16 with residual as representative)
            matches = [r for r in results
                      if r['model'] == model_name
                      and r['batch'] == batch
                      and r['dtype'] == 'F16'
                      and r['has_residual'] == True]
            if matches:
                r = matches[0]
                speedup = r['speedup']
                if speedup > 1.1:
                    cell = f"**{speedup:.2f}x** ✅"
                elif speedup > 1.0:
                    cell = f"{speedup:.2f}x ✅"
                elif speedup > 0.9:
                    cell = f"{speedup:.2f}x ⚠️"
                else:
                    cell = f"{speedup:.2f}x ❌"
                row_values.append(cell)
            else:
                row_values.append("-")
        lines.append(f"| {model_name} | " + " | ".join(row_values) + " |")

    lines.append("")

    # Legend
    lines.append("**Legend:**")
    lines.append("- ✅ Fused faster (>1.0x)")
    lines.append("- ⚠️ Comparable (~1.0x)")
    lines.append("- ❌ NonFused faster (<1.0x)")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by InfiniCore Fused FFN Benchmark*")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main(args):
    # args is passed from __main__ block, don't re-parse

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

    # Generate structured report
    report_path = getattr(args, 'report', None)
    if report_path:
        if not results:
            print(f"\n⚠️  Report generation skipped: No benchmark results collected")
        else:
            import os
            # Resolve to absolute path for clarity
            abs_path = os.path.abspath(report_path)
            generate_structured_report(results, report_path, model_configs)
            print(f"\n✅ Structured report saved to: {abs_path}")

    print("\033[92mBenchmark completed!\033[0m")


if __name__ == "__main__":
    import sys

    # Parse custom arguments before standard args
    batch_sizes_arg = None
    model_arg = None
    output_arg = None
    plot_arg = False
    report_arg = None

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
        elif sys.argv[i] == '--report' and i + 1 < len(sys.argv):
            report_arg = sys.argv[i + 1]
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
    args.report = report_arg

    main(args)
