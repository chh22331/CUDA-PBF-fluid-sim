#include "kernel_dispatcher.h"
#include "simulation_context.h"
#include "device_buffers.cuh"
#include "grid_buffers.cuh"
#include "logging.h"
#include "device_globals.cuh"
#include "parameters.h"

// 新全局接口
extern "C" void LaunchVelocityGlobals(float dtInv, uint32_t N, cudaStream_t);
extern "C" void LaunchIntegratePred(float4*, const float4*, float4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocity(float4*, const float4*, const float4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchBoundary(float4*, float4*, sim::GridBounds, float, uint32_t, cudaStream_t);
extern "C" void LaunchIntegratePred(float4*, const float4*, float4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocity(float4*, const float4*, const float4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchBoundary(float4*, float4*, sim::GridBounds, float, uint32_t, cudaStream_t);
// Phase Integrate：只调用预测积分 + 初次无弹性边界（不使用 XSPH）
namespace sim {

// Wrapper implementations for initial phases (Integrate + Velocity)
static void KernelIntegrate(DeviceBuffers& bufs, GridBuffers&, const SimParams& p, cudaStream_t s){
    LaunchIntegratePred(bufs.d_pos,bufs.d_vel,bufs.d_pos_pred,p.gravity,p.dt,p.numParticles,s);
    LaunchBoundary(bufs.d_pos_pred,bufs.d_vel,p.grid,0.0f,p.numParticles,s);
}

static void KernelVelocity(DeviceBuffers& bufs, GridBuffers&, const SimParams& p, cudaStream_t s){
    LaunchVelocity(bufs.d_vel,bufs.d_pos,bufs.d_pos_pred,1.0f/p.dt,p.numParticles,s);
}

static void RegisterBaseKernels(KernelDispatcher& kd) {
    kd.registerKernel(PhaseType::Integrate, false, false, false, KernelIntegrate);
    kd.registerKernel(PhaseType::Velocity,  false, false, false, KernelVelocity);
}

void InitializeKernelDispatcher(KernelDispatcher& kd) { RegisterBaseKernels(kd); }

} // namespace sim
