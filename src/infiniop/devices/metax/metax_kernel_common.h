#pragma once
#define INFINIOP_METAX_KERNEL __global__ void

#ifdef ENABLE_METAX_MC_API
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <maca_fp8.h>
#include <mccub/block/block_reduce.cuh>
#else
#include <hccub/block/block_reduce.cuh>
#include <hpcc_bfloat16.h>
#include <hpcc_fp16.h>
#include <hpcc_fp8.h>
#endif

// Posible maximum number of threads per block for METAX architectures
// Used for picking correct kernel launch configuration
#define METAX_BLOCK_SIZE_512 512
#define METAX_BLOCK_SIZE_1024 1024
#define METAX_BLOCK_SIZE_2048 2048
#define METAX_BLOCK_SIZE_4096 4096

#define CHECK_METAX(API) CHECK_INTERNAL(API, hcSuccess)

using cuda_bfloat16 = hpcc_bfloat16;
using cuda_bfloat162 = hpcc_bfloat162;
using cuda_fp8_e4m3 = __hpcc_fp8_e4m3;

#ifdef ENABLE_METAX_MC_API
using __nv_bfloat16 = __maca_bfloat16;
#else
using __nv_bfloat16 = __hpcc_bfloat16;
#endif

namespace device::metax {

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
} // namespace device::metax

using device::metax::indexToOffset;

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

__forceinline__ __device__ long double
exp_(const long double val) {
    return exp(val);
}

__forceinline__ __device__ double
exp_(const double val) {
    return exp(val);
}

__forceinline__ __device__ __half
exp_(const __half x) {
    return hexp(x);
}

__forceinline__ __device__ __hpcc_bfloat16
exp_(const __hpcc_bfloat16 x) {
    return hexp(x);
}
