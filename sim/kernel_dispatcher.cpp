#include "kernel_dispatcher.h"
#include "precision_stage.h"
#include "simulation_context.h"
#include "device_buffers.cuh"
#include "grid_buffers.cuh"
#include "logging.h"
#include "device_globals.cuh"
#include "parameters.h"

// 新全局接口
extern "C" void LaunchIntegratePredGlobals(float3 gravity, float dt, uint32_t N, cudaStream_t);
extern "C" void LaunchVelocityGlobals(float dtInv, uint32_t N, cudaStream_t);

// Phase Integrate：只调用预测积分 + 初次无弹性边界（不使用 XSPH）
namespace sim {

// Wrapper implementations for initial phases (Integrate + Velocity)
static void KernelIntegrate(DeviceBuffers& bufs, GridBuffers&, const SimParams& p, cudaStream_t s){
    bool useMP=(UseHalfForPosition(p,Stage::Integration,bufs)&&UseHalfForVelocity(p,Stage::Integration,bufs));
    if(useMP) LaunchIntegratePredMP(bufs.d_pos,bufs.d_vel,bufs.d_pos_pred,bufs.d_pos_h4,bufs.d_vel_h4,p.gravity,p.dt,p.numParticles,s);
    else LaunchIntegratePred(bufs.d_pos,bufs.d_vel,bufs.d_pos_pred,p.gravity,p.dt,p.numParticles,s);
    LaunchBoundary(bufs.d_pos_pred,bufs.d_vel,p.grid,0.0f,p.numParticles,s);
}

static void KernelVelocity(DeviceBuffers& bufs, GridBuffers&, const SimParams& p, cudaStream_t s){
    bool useMP=UseHalfForPosition(p,Stage::VelocityUpdate,bufs);
    if(useMP) LaunchVelocityMP(bufs.d_vel,bufs.d_pos,bufs.d_pos_pred,bufs.d_pos_h4,bufs.d_pos_pred_h4,1.0f/p.dt,p.numParticles,s);
    else LaunchVelocity(bufs.d_vel,bufs.d_pos,bufs.d_pos_pred,1.0f/p.dt,p.numParticles,s);
}

static void RegisterBaseKernels(KernelDispatcher& kd) {
    kd.registerKernel(PhaseType::Integrate, false, false, false, KernelIntegrate);
    kd.registerKernel(PhaseType::Velocity,  false, false, false, KernelVelocity);
}

void InitializeKernelDispatcher(KernelDispatcher& kd) { RegisterBaseKernels(kd); }

} // namespace sim
