#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>

extern "C" void LaunchSortPairsQuery(size_t* tempBytes,
    const uint32_t* d_keys_in, uint32_t* d_keys_out,
    const uint32_t* d_vals_in, uint32_t* d_vals_out,
    uint32_t N, cudaStream_t s)
{
    size_t bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, bytes,
        d_keys_in, d_keys_out,
        d_vals_in, d_vals_out,
        N, 0, 32, s);
    *tempBytes = bytes;
}

extern "C" void LaunchSortPairs(void* d_temp_storage, size_t tempBytes,
    uint32_t* d_keys_in, uint32_t* d_keys_out,
    uint32_t* d_vals_in, uint32_t* d_vals_out,
    uint32_t N, cudaStream_t s)
{
    cub::DeviceRadixSort::SortPairs(d_temp_storage, tempBytes,
        d_keys_in, d_keys_out,
        d_vals_in, d_vals_out,
        N, 0, 32, s);
}