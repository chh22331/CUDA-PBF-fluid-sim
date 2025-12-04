#include "kernel_dispatcher.h"
#include "simulation_context.h"
#include "device_buffers.cuh"
#include "grid_buffers.cuh"
#include "logging.h"
#include "device_globals.cuh"
#include "parameters.h"

// CUDA kernel entry points (C linkage, implemented in .cu files elsewhere)
extern "C" void LaunchVelocityGlobals(float dtInv, uint32_t N, cudaStream_t);
extern "C" void LaunchIntegratePred(float4* posCurr,
                                    const float4* velCurr,
                                    float4* posNext,
                                    float3 gravity,
                                    float dt,
                                    uint32_t N,
                                    cudaStream_t stream);
extern "C" void LaunchVelocity(float4* velOut,
                               const float4* posCurr,
                               const float4* posNext,
                               float dtInv,
                               uint32_t N,
                               cudaStream_t stream);
extern "C" void LaunchBoundary(float4* posNext,
                               float4* velCurr,
                               sim::GridBounds grid,
                               float restitution,
                               uint32_t N,
                               cudaStream_t stream);

// Phase Integrate: predict positions + apply boundary clamp (without XSPH).
// Phase Velocity: update velocity from predicted positions.
namespace sim {

    // Integrate phase: x_pred = x + v*dt + gravity*dt^2, then boundary handling
    static void KernelIntegrate(DeviceBuffers& bufs,
                                GridBuffers& /*gridBufs*/,
                                const SimParams& p,
                                cudaStream_t s) {
        LaunchIntegratePred(bufs.d_pos,
                            bufs.d_vel,
                            bufs.d_pos_pred,
                            p.gravity,
                            p.dt,
                            p.numParticles,
                            s);

        // Boundary kernel: restitution currently 0.0f (clamp without bounce)
        LaunchBoundary(bufs.d_pos_pred,
                       bufs.d_vel,
                       p.grid,
                       0.0f,
                       p.numParticles,
                       s);
    }

    // Velocity phase: recompute velocity using predicted positions (dtInv = 1/dt)
    static void KernelVelocity(DeviceBuffers& bufs,
                               GridBuffers& /*gridBufs*/,
                               const SimParams& p,
                               cudaStream_t s) {
        LaunchVelocity(bufs.d_vel,
                       bufs.d_pos,
                       bufs.d_pos_pred,
                       1.0f / p.dt,
                       p.numParticles,
                       s);
    }

    // Register base phases in dispatcher
    static void RegisterBaseKernels(KernelDispatcher& kd) {
        kd.registerKernel(PhaseType::Integrate, false, false, false, KernelIntegrate);
        kd.registerKernel(PhaseType::Velocity,  false, false, false, KernelVelocity);
    }

    // Public entry: initialize kernel dispatcher with default phases
    void InitializeKernelDispatcher(KernelDispatcher& kd) {
        RegisterBaseKernels(kd);
    }

} // namespace sim
