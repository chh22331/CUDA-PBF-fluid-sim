#pragma once
#include <cuda_runtime.h>

namespace sim {

    // Table holding device pointers to current and next positions.
    // Used as a single __constant__ block for fast access in kernels.
    struct SimPosDeviceTable {
        float4* curr; // current position buffer
        float4* next; // next/predicted position buffer
    };

    // Device-side constant symbol (defined in device_pos_state.cu).
    extern __device__ __constant__ SimPosDeviceTable g_simPosConst;

    // Upload host-side pointers to the device constant table.
    // This should be called whenever ping-pong buffers swap or rebind.
    void UploadSimPosTableConst(const float4* curr, const float4* next);

    // Placeholder kernel for potential synchronization/marking.
    // Currently performs no writes; kept for future extension.
    __global__ void SimPosSwapKernelConst(float4* newCurr, float4* newNext);

} // namespace sim