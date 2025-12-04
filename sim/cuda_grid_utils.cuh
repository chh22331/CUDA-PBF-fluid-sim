#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace sim {

// Under NVCC (.cu host+device passes), emit device-only symbols to avoid
// host-side duplicates across multiple translation units that include this header.
#if defined(__CUDACC__)

// Device-only helpers for __global__/__device__ code.

// Check if a 3D coordinate is inside [0, dim) on each axis.
__device__ __forceinline__
bool inBounds(int3 c, int3 dim) {
    return (c.x >= 0 && c.x < dim.x &&
            c.y >= 0 && c.y < dim.y &&
            c.z >= 0 && c.z < dim.z);
}

// Row-major linear index: ((z * Y) + y) * X + x.
__device__ __forceinline__
uint32_t linIdx(int3 c, int3 dim) {
    return static_cast<uint32_t>((c.z * dim.y + c.y) * dim.x + c.x);
}

// Alias for linIdx (kept for compatibility).
__device__ __forceinline__
uint32_t lid(int3 c, int3 dim) { return linIdx(c, dim); }

// --- Compact grid lookups on unique sorted keys ---

// lower_bound on a sorted uint32 array: first index i where keys[i] >= key.
__device__ __forceinline__
int lower_bound_u32(const uint32_t* keys, int M, uint32_t key) {
    int lo = 0, hi = M;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        uint32_t v = keys[mid];
        if (v < key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// If found, outputs [beg, end) via offsets; returns true when range is non-empty.
__device__ __forceinline__
bool compact_cell_range(
    const uint32_t* uniqueKeys,
    const uint32_t* offsets,
    uint32_t M,
    uint32_t key,
    uint32_t& beg,
    uint32_t& end)
{
    if (M == 0u) { beg = end = 0u; return false; }

    int pos = lower_bound_u32(uniqueKeys, int(M), key);
    if (pos >= int(M) || uniqueKeys[pos] != key) {
        beg = end = 0u;
        return false;
    }

    beg = offsets[pos];
    end = offsets[pos + 1];
    return (beg < end);
}

#else // !__CUDACC__

// Host-only inline fallbacks (useful for CPU-side utilities/tests).

inline bool inBounds(int3 c, int3 dim) {
    return (c.x >= 0 && c.x < dim.x &&
            c.y >= 0 && c.y < dim.y &&
            c.z >= 0 && c.z < dim.z);
}

inline uint32_t linIdx(int3 c, int3 dim) {
    return static_cast<uint32_t>((c.z * dim.y + c.y) * dim.x + c.x);
}

inline uint32_t lid(int3 c, int3 dim) { return linIdx(c, dim); }

#endif // __CUDACC__

} // namespace sim