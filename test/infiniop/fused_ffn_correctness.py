import torch
import torch.nn.functional as F
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================

# Test cases: (batch, hidden_dim, intermediate_dim, has_residual)
_TEST_CASES = [
    # Small batches
    (1, 2048, 5632, True),
    (2, 2048, 5632, True),
    (4, 2048, 5632, True),
    # Medium batches
    (8, 2048, 5632, True),
    (16, 2048, 5632, True),
    (32, 2048, 5632, True),
    # Large batches
    (64, 2048, 5632, True),
    (128, 2048, 5632, True),
    # Different architectures
    (16, 4096, 11008, True),    # LLaMA-7B
    (16, 5120, 13824, True),    # LLaMA-13B
    (16, 3584, 18944, True),    # Qwen
    # Without residual
    (16, 2048, 5632, False),
    (32, 4096, 11008, False),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-2, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


def reference_fused_ffn(x, residual, norm_w, gate_up_w, down_w, eps):
    """
    Reference implementation of fused FFN using PyTorch.

    Args:
        x: Input tensor [ntok, d]
        residual: Residual tensor [ntok, d] (optional)
        norm_w: RMSNorm weight [d]
        gate_up_w: GateUp weight [2*di, d]
        down_w: Down weight [d, di]
        eps: RMSNorm epsilon

    Returns:
        Output tensor [ntok, d]
    """
    # RMSNorm
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + eps)
    normalized = (normalized * norm_w.float()).to(x.dtype)

    # GateUp projection (gate_up = normalized @ gate_up_w^T)
    gate_up = F.linear(normalized, gate_up_w)
    di = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :di], gate_up[..., di:]

    # SwiGLU (out = silu(gate) * up)
    hidden = F.silu(gate) * up

    # Down projection (out = hidden @ down_w^T)
    out = F.linear(hidden, down_w)

    # Residual add
    if residual is not None:
        out = out + residual

    return out


def test(
    handle,
    device,
    batch,
    hidden_dim,
    intermediate_dim,
    has_residual,
    dtype=InfiniDtype.F16,
    sync=None,
):
    """
    Test the fused FFN operator.
    """
    # Create input tensors with scaled weights to avoid overflow
    # For large matrices, use smaller weight values
    weight_scale = 1.0 / (hidden_dim ** 0.5)  # Xavier-like scaling

    x = TestTensor((batch, hidden_dim), None, dtype, device)
    norm_w = TestTensor((hidden_dim,), None, dtype, device)
    gate_up_w = TestTensor((2 * intermediate_dim, hidden_dim), None, dtype, device, scale=weight_scale)
    down_w = TestTensor((hidden_dim, intermediate_dim), None, dtype, device, scale=weight_scale)

    # Create residual tensor if needed
    residual = None
    if has_residual:
        residual = TestTensor((batch, hidden_dim), None, dtype, device)

    # Create output tensor
    out = TestTensor((batch, hidden_dim), None, dtype, device)

    epsilon = 1e-6

    print(
        f"Testing FusedFFN on {InfiniDeviceNames[device]} with "
        f"batch:{batch} hidden_dim:{hidden_dim} intermediate_dim:{intermediate_dim} "
        f"has_residual:{has_residual} dtype:{InfiniDtypeNames[dtype]}"
    )

    # Compute reference output
    ans = reference_fused_ffn(
        x.torch_tensor(),
        residual.torch_tensor() if residual else None,
        norm_w.torch_tensor(),
        gate_up_w.torch_tensor(),
        down_w.torch_tensor(),
        epsilon,
    )

    if sync is not None:
        sync()

    # Create descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateFusedFFNDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            x.descriptor,
            residual.descriptor if residual else None,
            norm_w.descriptor,
            gate_up_w.descriptor,
            down_w.descriptor,
            ctypes.c_float(epsilon),
        )
    )

    # Invalidate descriptors
    x.destroy_desc()
    norm_w.destroy_desc()
    gate_up_w.destroy_desc()
    down_w.destroy_desc()
    out.destroy_desc()
    if residual:
        residual.destroy_desc()

    # Get workspace size
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetFusedFFNWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, out.device)

    def lib_fused_ffn():
        check_error(
            LIBINFINIOP.infiniopFusedFFN(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                x.data(),
                residual.data() if residual else None,
                norm_w.data(),
                gate_up_w.data(),
                down_w.data(),
                None,
            )
        )

    lib_fused_ffn()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        def torch_fused_ffn():
            return reference_fused_ffn(
                x.torch_tensor(),
                residual.torch_tensor() if residual else None,
                norm_w.torch_tensor(),
                gate_up_w.torch_tensor(),
                down_w.torch_tensor(),
                epsilon,
            )

        profile_operation("PyTorch", torch_fused_ffn, device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_fused_ffn, device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyFusedFFNDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
