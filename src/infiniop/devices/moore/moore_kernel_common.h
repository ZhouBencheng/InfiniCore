#ifndef __INFINIOP_MOORE_KERNEL_COMMON_H__
#define __INFINIOP_MOORE_KERNEL_COMMON_H__
#define INFINIOP_MOORE_KERNEL __global__ void

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_fp8.h>

// Posible maximum number of threads per block for MUSA architectures
// Used for picking correct kernel launch configuration
#define MOORE_BLOCK_SIZE_4096 4096
#define MOORE_BLOCK_SIZE_2048 2048
#define MOORE_BLOCK_SIZE_1024 1024
#define MOORE_BLOCK_SIZE_512 512

#define CHECK_MOORE(API) CHECK_INTERNAL(API, musaSuccess)

using cuda_bfloat16 = mt_bfloat16;
using cuda_bfloat162 = mt_bfloat162;
using cuda_fp8_e4m3 = __mt_fp8_e4m3;

using __nv_bfloat16 = __mt_bfloat16;

namespace device::moore {

// get the memory offset of the given element in a tensor given its flat index
__forceinline__ __device__ __host__ size_t
indexToOffset(
    size_t flat_index,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides) {
    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}
} // namespace device::moore

using device::moore::indexToOffset;

// Fallback atomic add. Mirrors the helper in nvidia_kernel_common.cuh so that
// op kernels shared via ../cuda/kernel.cuh compile on every backend. The CAS
// loop is only used on Iluvatar (which lacks a native double atomicAdd); every
// other backend keeps using the native hardware atomicAdd unchanged.
template <typename T>
__forceinline__ __device__ T atomicAddSafe(T *address, T val) {
    return atomicAdd(address, val);
}

#if defined(ENABLE_ILUVATAR_API)
template <>
__forceinline__ __device__ double atomicAddSafe<double>(double *address, double val) {
    unsigned long long int *addr_as_ull = reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__forceinline__ __device__ float
exp_(const float val) {
    return expf(val);
}

// Computes exp for long double on Moore GPU,
// casts to double to resolve ambiguous exp call,
// due to conflicting double/float definitions in MUSA math libraries.
__forceinline__ __device__ long double
exp_(const long double val) {
    return static_cast<long double>(exp(static_cast<double>(val)));
}

__forceinline__ __device__ double
exp_(const double val) {
    return exp(val);
}

// <musa_bf16.h> may not support hexp
__forceinline__ __device__ __half
exp_(const __half x) {
    float f_val = __half2float(x);
    float f_result = expf(f_val);
    return __float2half(f_result);
}

// <musa_bf16.h> may not support hexp
__forceinline__ __device__ __mt_bfloat16
exp_(const __mt_bfloat16 x) {
    float f_val = __bfloat162float(x);
    float f_result = expf(f_val);
    return __float2bfloat16(f_result);
}
#endif
