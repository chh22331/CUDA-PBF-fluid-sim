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

// 只做预测积分，不做边界
static void KernelIntegrate(DeviceBuffers&, GridBuffers&, const SimParams& p, cudaStream_t s) {
    LaunchIntegratePredGlobals(p.gravity, p.dt, p.numParticles, s);
}

// 只做速度更新，不做边界
static void KernelVelocity(DeviceBuffers&, GridBuffers&, const SimParams& p, cudaStream_t s) {
    LaunchVelocityGlobals(1.0f / p.dt, p.numParticles, s);
}

static void RegisterBaseKernels(KernelDispatcher& kd) {
    kd.registerKernel(PhaseType::Integrate, false, false, false, KernelIntegrate);
    kd.registerKernel(PhaseType::Velocity,  false, false, false, KernelVelocity);
}

void InitializeKernelDispatcher(KernelDispatcher& kd) { RegisterBaseKernels(kd); }

} // namespace sim
