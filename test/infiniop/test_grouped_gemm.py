import ctypes
import os
import sys
from ctypes import c_void_p, c_int, c_size_t, c_float, POINTER

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
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
)

# ==============================================================================
#  Low-level C interface registration
# ==============================================================================
# GroupedGemm passes raw device pointers and plain int arrays (m/n/k and the
# leading dimensions) instead of tensor descriptors, so it is not handled by the
# automatic op_register machinery and must be declared explicitly here.
LIBINFINIOP.infiniopCreateGroupedGemmDescriptor.argtypes = [
    c_void_p, POINTER(c_void_p), c_int,
    POINTER(c_int), POINTER(c_int), POINTER(c_int),
    POINTER(c_int), POINTER(c_int), POINTER(c_int),
    c_int,
]
LIBINFINIOP.infiniopGetGroupedGemmWorkspaceSize.argtypes = [c_void_p, POINTER(c_size_t)]
LIBINFINIOP.infiniopGroupedGemm.argtypes = [
    c_void_p, c_void_p, c_size_t,
    POINTER(c_void_p), POINTER(c_void_p), POINTER(c_void_p),
    c_float, c_float, c_void_p,
]
LIBINFINIOP.infiniopDestroyGroupedGemmDescriptor.argtypes = [c_void_p]

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# Each test case is a single GroupedGemm invocation that batches several
# independently shaped GEMMs ("groups"). It is described as:
#     (alpha, beta, groups)
# where every group is either a 3-tuple or a 6-tuple:
#     (m, k, n)                      -> contiguous, lda/ldb/ldc default to k/n/n
#     (m, k, n, lda, ldb, ldc)       -> explicit (padded) leading dimensions
# For every group:  A:(m, k)  B:(k, n)  C:(m, n),  all row-major.
# Each group computes:  C = alpha * (A @ B) + beta * C
_TEST_CASES = [
    # 1. single small square group (basic sanity)
    (1.0, 0.0, [(64, 64, 64)]),
    # 2. multiple groups with varied rectangular shapes
    (1.0, 0.0, [(128, 256, 64), (64, 128, 128), (256, 64, 256)]),
    # 3. alpha / beta scaling with C accumulation across groups
    (0.5, 1.0, [(128, 256, 128), (96, 128, 64)]),
    # 4. many groups with a shared inner shape (expert-batching style)
    (1.0, 0.0, [(m, 512, 1024) for m in (16, 32, 8, 64, 24, 40, 12, 48)]),
    # 5. GEMV-like group: m = 1 with a large contraction dimension
    (1.0, 0.0, [(1, 4096, 4096)]),
    # 6. padded leading dimensions (strided storage)
    (1.0, 0.0, [(64, 128, 96, 160, 112, 128), (32, 64, 48, 96, 64, 80)]),
    # 7. alpha-only large single group
    (2.0, 0.0, [(512, 512, 512)]),
    # 8. degenerate inner / output dimensions (k = 1 and n = 1)
    (1.0, 0.0, [(8, 1, 16), (16, 32, 1)]),
    # 9. irregular odd sizes with accumulation
    (1.0, 1.0, [(33, 65, 17), (50, 50, 50), (7, 200, 9)]),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 0, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 0, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def _parse_group(group):
    """Normalize a group spec into (m, k, n, lda, ldb, ldc) with effective strides."""
    m, k, n = group[0], group[1], group[2]
    lda = group[3] if len(group) > 3 else None
    ldb = group[4] if len(group) > 4 else None
    ldc = group[5] if len(group) > 5 else None
    return m, k, n, lda, ldb, ldc


# The argument list should be (handle, torch_device, <param list>, dtype, sync)
# The <param list> should keep the same order as the one specified in _TEST_CASES
def test(
    handle,
    device,
    alpha,
    beta,
    groups,
    dtype=InfiniDtype.F16,
    sync=None,
):
    group_count = len(groups)
    shapes_str = ", ".join(f"({g[0]},{g[1]},{g[2]})" for g in groups)
    print(
        f"Testing GroupedGemm on {InfiniDeviceNames[device]} with group_count:{group_count}"
        f" alpha:{alpha} beta:{beta} shapes(M,K,N):[{shapes_str}] dtype:{InfiniDtypeNames[dtype]}"
    )

    a_tensors, b_tensors, c_tensors, ans_tensors = [], [], [], []
    a_ptrs, b_ptrs, c_ptrs = [], [], []
    m_list, n_list, k_list = [], [], []
    lda_list, ldb_list, ldc_list = [], [], []

    for group in groups:
        m, k, n, lda, ldb, ldc = _parse_group(group)
        a_stride = [lda, 1] if lda is not None else None
        b_stride = [ldb, 1] if ldb is not None else None
        c_stride = [ldc, 1] if ldc is not None else None

        # Initialize tensors
        a = TestTensor((m, k), a_stride, dtype, device)
        b = TestTensor((k, n), b_stride, dtype, device)
        c = TestTensor((m, n), c_stride, dtype, device, mode="ones")
        ans = TestTensor((m, n), c_stride, dtype, device, mode="zeros")

        a_tensors.append(a)
        b_tensors.append(b)
        c_tensors.append(c)
        ans_tensors.append(ans)
        a_ptrs.append(a.data())
        b_ptrs.append(b.data())
        c_ptrs.append(c.data())

        m_list.append(m)
        n_list.append(n)
        k_list.append(k)
        lda_list.append(lda if lda is not None else k)
        ldb_list.append(ldb if ldb is not None else n)
        ldc_list.append(ldc if ldc is not None else n)

    # Compute the PyTorch reference result: ans = beta * C + alpha * (A @ B)
    def torch_grouped_gemm():
        for i in range(group_count):
            torch.addmm(
                c_tensors[i].torch_tensor(),
                a_tensors[i].torch_tensor(),
                b_tensors[i].torch_tensor(),
                beta=beta,
                alpha=alpha,
                out=ans_tensors[i].torch_tensor(),
            )

    torch_grouped_gemm()

    if sync is not None:
        sync()

    IntArray = c_int * group_count
    PtrArray = c_void_p * group_count
    m_arr, n_arr, k_arr = IntArray(*m_list), IntArray(*n_list), IntArray(*k_list)
    lda_arr, ldb_arr, ldc_arr = IntArray(*lda_list), IntArray(*ldb_list), IntArray(*ldc_list)
    a_ptr_arr, b_ptr_arr, c_ptr_arr = PtrArray(*a_ptrs), PtrArray(*b_ptrs), PtrArray(*c_ptrs)

    descriptor = c_void_p()
    check_error(
        LIBINFINIOP.infiniopCreateGroupedGemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            group_count,
            m_arr,
            n_arr,
            k_arr,
            lda_arr,
            ldb_arr,
            ldc_arr,
            dtype,
        )
    )

    # Get workspace size and create workspace
    workspace_size = c_size_t(0)
    check_error(
        LIBINFINIOP.infiniopGetGroupedGemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Execute infiniop grouped gemm operator (writes the result into the C tensors)
    def lib_grouped_gemm():
        check_error(
            LIBINFINIOP.infiniopGroupedGemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                c_ptr_arr,
                a_ptr_arr,
                b_ptr_arr,
                alpha,
                beta,
                None,
            )
        )

    lib_grouped_gemm()

    # Validate results group by group
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    for i in range(group_count):
        actual = c_tensors[i].actual_tensor()
        expected = ans_tensors[i].torch_tensor()
        if DEBUG:
            debug(actual, expected, atol=atol, rtol=rtol)
        assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
            f"GroupedGemm mismatch in group {i} "
            f"(M,K,N)=({m_list[i]},{k_list[i]},{n_list[i]}) dtype={InfiniDtypeNames[dtype]}"
        )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_grouped_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_grouped_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyGroupedGemmDescriptor(descriptor))


# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
