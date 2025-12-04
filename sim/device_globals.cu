#include "device_globals.cuh"
#include "device_buffers.cuh"

namespace sim {

    // Device-side symbol definitions (initialized to nullptr).
    __device__ float4* g_pos       = nullptr;
    __device__ float4* g_vel       = nullptr;
    __device__ float4* g_pos_pred  = nullptr;
    __device__ float4* g_delta     = nullptr;
    __device__ float*  g_lambda    = nullptr;

    __device__ uint32_t g_ghostCount = 0;

    // Note: cudaMemcpyToSymbol requires symbol names at compile time.
    // The helper below is only a conceptual placeholder and intentionally unused.
    static void CopyPtrToSymbol(float4* const* /*hPtr*/, float4** /*symbol*/) {
        // No-op: kept to document intent.
    }

    // Bind basic device globals by copying host-side pointers into device symbols.
    // The cost is minimal (a few pointer-sized copies).
    void BindDeviceGlobals(float4* d_pos_curr,
                           float4* d_vel_curr,
                           float4* d_pos_next,
                           float4* d_delta_ptr,
                           float*  d_lambda_ptr) {
        cudaMemcpyToSymbol(g_pos,       &d_pos_curr,  sizeof(float4*));
        cudaMemcpyToSymbol(g_vel,       &d_vel_curr,  sizeof(float4*));
        cudaMemcpyToSymbol(g_pos_pred,  &d_pos_next,  sizeof(float4*));
        cudaMemcpyToSymbol(g_delta,     &d_delta_ptr, sizeof(float4*));
        cudaMemcpyToSymbol(g_lambda,    &d_lambda_ptr,sizeof(float*));
    }

    // Preferred binding: uses aliases maintained by DeviceBuffers.
    void BindDeviceGlobalsFrom(const DeviceBuffers& bufs) {
        BindDeviceGlobals(bufs.d_pos,
                          bufs.d_vel,
                          bufs.d_pos_pred,
                          bufs.d_delta,
                          bufs.d_lambda);
    }

    // Upload ghost particle count to device symbol.
    void UploadGhostCount(uint32_t ghostCount) {
        cudaMemcpyToSymbol(g_ghostCount, &ghostCount, sizeof(uint32_t));
    }

} // namespace sim