import torch
import ctypes
from ctypes import c_void_p, c_int, c_size_t, c_float, POINTER
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    test_operator,
    check_error,
    get_args,
    TestWorkspace,
    InfiniDtype,
    InfiniDeviceNames,
)

# 注册底层 C 接口
LIBINFINIOP.infiniopCreateGroupedGemmDescriptor.argtypes = [
    c_void_p, POINTER(c_void_p), c_int,
    POINTER(c_int), POINTER(c_int), POINTER(c_int),
    POINTER(c_int), POINTER(c_int), POINTER(c_int),
    c_int
]
LIBINFINIOP.infiniopGetGroupedGemmWorkspaceSize.argtypes = [c_void_p, POINTER(c_size_t)]
LIBINFINIOP.infiniopGroupedGemm.argtypes = [
    c_void_p, c_void_p, c_size_t,
    POINTER(c_void_p), POINTER(c_void_p), POINTER(c_void_p),
    c_float, c_float, c_void_p
]
LIBINFINIOP.infiniopDestroyGroupedGemmDescriptor.argtypes = [c_void_p]

# 测试用例: [(m, k), (k, n)]
_TEST_CASES = [
    (1.0, 0.0, [
        ((128, 256), (256, 64)),
        ((64, 128),  (128, 128)),
        ((256, 64),  (64, 256)),
    ])
]

_TENSOR_DTYPES = [InfiniDtype.F16]

def test(handle, device, alpha, beta, groups, dtype=InfiniDtype.F16, sync=None):
    group_count = len(groups)
    m_list, n_list, k_list = [], [], []
    lda_list, ldb_list, ldc_list = [], [], []

    a_tensors, b_tensors, c_tensors, ans_tensors = [], [], [], []
    a_ptrs, b_ptrs, c_ptrs = [], [], []

    for (a_shape, b_shape) in groups:
        m, k = a_shape
        _, n = b_shape

        a = TestTensor((m, k), None, dtype, device)
        b = TestTensor((k, n), None, dtype, device)
        c = TestTensor((m, n), None, dtype, device, mode="ones")
        ans = TestTensor((m, n), None, dtype, device, mode="zeros")

        torch.matmul(a.torch_tensor(), b.torch_tensor(), out=ans.torch_tensor())
        ans.torch_tensor().mul_(alpha).add_(c.torch_tensor(), alpha=beta)

        a_tensors.append(a); b_tensors.append(b); c_tensors.append(c); ans_tensors.append(ans)

        m_list.append(m); n_list.append(n); k_list.append(k)
        lda_list.append(k); ldb_list.append(n); ldc_list.append(n)
        a_ptrs.append(a.data()); b_ptrs.append(b.data()); c_ptrs.append(c.data())

    if sync is not None: sync()

    IntArray = c_int * group_count
    PtrArray = c_void_p * group_count
    m_arr, n_arr, k_arr = IntArray(*m_list), IntArray(*n_list), IntArray(*k_list)
    lda_arr, ldb_arr, ldc_arr = IntArray(*lda_list), IntArray(*ldb_list), IntArray(*ldc_list)
    a_ptr_arr, b_ptr_arr, c_ptr_arr = PtrArray(*a_ptrs), PtrArray(*b_ptrs), PtrArray(*c_ptrs)

    desc = c_void_p()
    check_error(LIBINFINIOP.infiniopCreateGroupedGemmDescriptor(
        handle, ctypes.byref(desc), group_count,
        m_arr, n_arr, k_arr, lda_arr, ldb_arr, ldc_arr, dtype
    ))

    ws_size = c_size_t(0)
    check_error(LIBINFINIOP.infiniopGetGroupedGemmWorkspaceSize(desc, ctypes.byref(ws_size)))
    workspace = TestWorkspace(ws_size.value, device)

    check_error(LIBINFINIOP.infiniopGroupedGemm(
        desc, workspace.data(), ws_size.value,
        c_ptr_arr, a_ptr_arr, b_ptr_arr, alpha, beta, None
    ))

    # 输出统计表格
    print(f"{'Group':<8} | {'Shape (M, K, N)':<20} | {'Max Diff':<12} | {'Status':<10}")
    print("-" * 60)
    
    for i in range(group_count):
        c_actual = c_tensors[i].actual_tensor()
        c_ans = ans_tensors[i].torch_tensor()
        diff = (c_actual - c_ans).abs().max().item()
        
        status = "OK" if diff < 1e-2 else "FAIL"
        shape_str = f"({m_list[i]},{k_list[i]},{n_list[i]})"
        print(f"{i:<8} | {shape_str:<20} | {diff:<12.6f} | {status:<10}")

    check_error(LIBINFINIOP.infiniopDestroyGroupedGemmDescriptor(desc))

if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)