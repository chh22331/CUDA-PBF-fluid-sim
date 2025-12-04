#include "device_pos_state.cuh"

namespace sim {

    // Single definition of the device-side constant symbol.
    __device__ __constant__ SimPosDeviceTable g_simPosConst;

    // Upload host-side position pointers into the constant table.
    // Note: const_cast is safe here because only the pointer value is copied;
    // kernels treat these buffers as read-only or per-kernel policy.
    void UploadSimPosTableConst(const float4* curr, const float4* next) {
        SimPosDeviceTable h{ const_cast<float4*>(curr), const_cast<float4*>(next) };
        cudaMemcpyToSymbol(g_simPosConst, &h, sizeof(h));
    }

    // Device placeholder kernel (no-op).
    // Useful for future tagging/synchronization without touching data.
    __global__ void SimPosSwapKernelConst(float4* newCurr, float4* newNext) {
        (void)newCurr;
        (void)newNext;
    }

} // namespace sim