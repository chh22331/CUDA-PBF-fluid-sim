#include "kernel_dispatcher.h"
#include "precision_stage.h"
#include "simulation_context.h"
#include "device_buffers.cuh"
#include "grid_buffers.cuh"
#include "logging.h"

extern "C" void LaunchIntegratePred(float4*, const float4*, float4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchIntegratePredMP(const float4*, const float4*, float4*, const sim::Half4*, const sim::Half4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocity(float4*, const float4*, const float4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocityMP(float4*, const float4*, const float4*, const sim::Half4*, const sim::Half4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchBoundary(float4*, float4*, sim::GridBounds, float, uint32_t, cudaStream_t);

namespace sim {

// Wrapper implementations for initial phases (Integrate + Velocity)
static void KernelIntegrate(DeviceBuffers& bufs, GridBuffers&, const SimParams& p, cudaStream_t s){
    bool useMP=(UseHalfForPosition(p,Stage::Integration,bufs)&&UseHalfForVelocity(p,Stage::Integration,bufs));
    if(useMP) LaunchIntegratePredMP(bufs.d_pos_curr, bufs.d_vel, bufs.d_pos_next, bufs.d_pos_curr_h4, bufs.d_vel_h4, p.gravity, p.dt, p.numParticles, s);
    else LaunchIntegratePred(bufs.d_pos_curr, bufs.d_vel, bufs.d_pos_next, p.gravity, p.dt, p.numParticles, s);
    LaunchBoundary(bufs.d_pos_next, bufs.d_vel, p.grid,0.0f, p.numParticles, s);
}

static void KernelVelocity(DeviceBuffers& bufs, GridBuffers&, const SimParams& p, cudaStream_t s){
    bool useMP=UseHalfForPosition(p,Stage::VelocityUpdate,bufs);
    if(useMP) LaunchVelocityMP(bufs.d_vel, bufs.d_pos_curr, bufs.d_pos_next, bufs.d_pos_curr_h4, bufs.d_pos_next_h4,1.0f/p.dt, p.numParticles, s);
    else LaunchVelocity(bufs.d_vel, bufs.d_pos_curr, bufs.d_pos_next,1.0f/p.dt, p.numParticles, s);
}

// Registration helper
static KernelDispatcher g_dispatcherInit;

static void RegisterBaseKernels(KernelDispatcher& kd){
    kd.registerKernel(PhaseType::Integrate,false,false,false,KernelIntegrate);
    kd.registerKernel(PhaseType::Integrate,true,true,false,KernelIntegrate); // mixed path inside wrapper chooses half logic
    kd.registerKernel(PhaseType::Velocity,false,false,false,KernelVelocity);
    kd.registerKernel(PhaseType::Velocity,true,true,false,KernelVelocity);
}

// Exposed initialization entry (to be called from Simulator later when migrating)
void InitializeKernelDispatcher(KernelDispatcher& kd){ RegisterBaseKernels(kd); }

} // namespace sim
