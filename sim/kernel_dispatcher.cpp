#include "kernel_dispatcher.h"
#include "precision_stage.h"
#include "simulation_context.h"
#include "device_buffers.cuh"
#include "grid_buffers.cuh"
#include "logging.h"
#include "../engine/core/console.h"
#include "../engine/core/prof_nvtx.h"

extern "C" void LaunchIntegratePred(float4*, const float4*, float4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchIntegratePredMP(const float4*, const float4*, float4*, const sim::Half4*, const sim::Half4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocity(float4*, const float4*, const float4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocityMP(float4*, const float4*, const float4*, const sim::Half4*, const sim::Half4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocityHalf(float4*, sim::Half4*, const sim::Half4*, const sim::Half4*, float, uint32_t, bool, float, cudaStream_t);
extern "C" void LaunchVelocityDiag(const float4*, const float4*, uint32_t, double*, double*, int*, cudaStream_t);
extern "C" void LaunchVelocityExplodeDiag(const float4*, const float4*, const float4*, const sim::Half4*, uint32_t, float, float, int, float, double*, double*, double*, double*, double*, unsigned long long*, unsigned long long*, cudaStream_t);
extern "C" void LaunchBoundary(float4*, float4*, sim::GridBounds, float, uint32_t, uint32_t, uint8_t, cudaStream_t);
extern "C" void LaunchBoundaryHalf(float4*, float4*, const sim::Half4*, const sim::Half4*, sim::GridBounds, float, uint32_t, uint32_t, bool, uint8_t, cudaStream_t);

namespace sim {

static inline uint8_t GhostClampFlag(){ const auto& c = console::Instance(); const auto& g = c.sim.boundaryGhost; return (g.enable ? (g.place_outside ?0u :1u) :0u); }

// 边界阶段统一调度，支持半精
static inline void DispatchBoundary(DeviceBuffers& bufs, const SimParams& p, float4* posPredFp32, float4* velFp32, uint32_t count, float restitution, cudaStream_t s){
 if (count ==0) return;
 bool useHalfPos = UseHalfForPosition(p, Stage::Boundary, bufs);
 bool useHalfVel = UseHalfForVelocity(p, Stage::Boundary, bufs);
 bool canHalf = useHalfPos && useHalfVel && bufs.d_pos_pred_h4 && bufs.d_vel_h4;
 if (bufs.nativeHalfActive) canHalf = true;
 uint8_t ghostClamp = GhostClampFlag(); uint32_t ghostCount = p.ghostParticleCount;
 if (canHalf){ if (!bufs.nativeHalfActive) bufs.packAllToHalf(count,s); LaunchBoundaryHalf(posPredFp32, velFp32, bufs.d_pos_pred_h4, bufs.d_vel_h4, p.grid, restitution, count, ghostCount, p.precision.forceFp32Accumulate, ghostClamp, s); }
 else { LaunchBoundary(posPredFp32, velFp32, p.grid, restitution, count, ghostCount, ghostClamp, s); }
}

static void KernelIntegrate(DeviceBuffers& bufs, GridBuffers&, const SimParams& p, cudaStream_t s){
 bool useMP=(UseHalfForPosition(p,Stage::Integration,bufs)&&UseHalfForVelocity(p,Stage::Integration,bufs));
 if(useMP) LaunchIntegratePredMP(bufs.d_pos,bufs.d_vel,bufs.d_pos_pred,bufs.d_pos_h4,bufs.d_vel_h4,p.gravity,p.dt,p.numParticles,s);
 else LaunchIntegratePred(bufs.d_pos,bufs.d_vel,bufs.d_pos_pred,p.gravity,p.dt,p.numParticles,s);
 DispatchBoundary(bufs, p, bufs.d_pos_pred, bufs.d_vel, p.numParticles,0.0f, s);
}

static void KernelVelocity(DeviceBuffers& bufs, GridBuffers&, const SimParams& p, cudaStream_t s){
 if (p.numParticles==0) return;
 if (p.dt <=0.f){ const auto& cc = console::Instance(); if (cc.debug.printWarnings) std::fprintf(stderr,"[Velocity][Skip] dt<=0 (dt=%.6g)\n", p.dt); return; }
 const auto& cc = console::Instance();
 bool doDiag = cc.debug.logVelocityEffect; // 新独立开关
 if (doDiag && bufs.d_vel_prev){ cudaMemcpyAsync(bufs.d_vel_prev, bufs.d_vel, sizeof(float4)*p.numParticles, cudaMemcpyDeviceToDevice, s); }

 bool halfLoadPos = UseHalfForPosition(p, Stage::VelocityUpdate, bufs);
 bool halfLoadVel = UseHalfForVelocity(p, Stage::VelocityUpdate, bufs);
 bool haveHalfPos = (bufs.d_pos_h4 && bufs.d_pos_pred_h4);
 bool haveHalfVel = (bufs.d_vel_h4 != nullptr);
 bool preferHalfArithmetic = (halfLoadPos && halfLoadVel && haveHalfPos && haveHalfVel && p.precision.enableHalfIntrinsics);
 if (preferHalfArithmetic || ((halfLoadPos||halfLoadVel) && (haveHalfPos||haveHalfVel))) bufs.packAllToHalf(p.numParticles, s);
 static bool loggedHalf=false, loggedMP=false, loggedFP32=false, loggedFallback=false;
 auto logSel = [&](const char* tag){ if (cc.debug.printDiagnostics) std::fprintf(stderr,"[Velocity][Select] %s N=%u inv_dt=%.4g halfPos=%d halfVel=%d intrinsic=%d forceFp32Acc=%d\n", tag, p.numParticles,1.0/p.dt, (int)halfLoadPos,(int)halfLoadVel,(int)p.precision.enableHalfIntrinsics,(int)p.precision.forceFp32Accumulate); };

 if (preferHalfArithmetic){
 if (!loggedHalf){ logSel("HalfArithmetic"); loggedHalf=true; }
 if (cc.debug.velocityEnableNvtx) prof::Range rNvtx("Velocity.HalfArithmetic", prof::Color(0x50,0x90,0xD0));
 LaunchVelocityHalf(bufs.d_vel, bufs.d_vel_h4, bufs.d_pos_h4, bufs.d_pos_pred_h4,1.0f/p.dt, p.numParticles, p.precision.forceFp32Accumulate, cc.debug.velocityNaNGuardMax, s);
 }
 else if ((halfLoadPos||halfLoadVel) && (haveHalfPos||haveHalfVel)){
 if (!loggedMP){ logSel("MixedHalfLoad"); loggedMP=true; }
 if (cc.debug.velocityEnableNvtx) prof::Range rNvtx("Velocity.MixedHalfLoad", prof::Color(0x70,0x70,0xD0));
 LaunchVelocityMP(bufs.d_vel, bufs.d_pos, bufs.d_pos_pred, bufs.d_pos_h4, bufs.d_pos_pred_h4,1.0f/p.dt, p.numParticles, s);
 }
 else {
 if ((halfLoadPos||halfLoadVel) && !haveHalfPos && !haveHalfVel && !loggedFallback){ if (cc.debug.printWarnings) std::fprintf(stderr,"[Velocity][Fallback] half requested but buffers missing pos_h4=%p pos_pred_h4=%p vel_h4=%p -> FP32\n", (void*)bufs.d_pos_h4,(void*)bufs.d_pos_pred_h4,(void*)bufs.d_vel_h4); loggedFallback=true; }
 if (!loggedFP32){ logSel("FP32"); loggedFP32=true; }
 if (cc.debug.velocityEnableNvtx) prof::Range rNvtx("Velocity.FP32", prof::Color(0x90,0x50,0x30));
 LaunchVelocity(bufs.d_vel, bufs.d_pos, bufs.d_pos_pred,1.0f/p.dt, p.numParticles, s);
 }

 if (doDiag){
 static double* d_sumPrev=nullptr, * d_sumDv=nullptr; static int* d_hasNaN=nullptr;
 if (!d_sumPrev){ cudaMalloc(&d_sumPrev,sizeof(double)); cudaMalloc(&d_sumDv,sizeof(double)); cudaMalloc(&d_hasNaN,sizeof(int)); }
 double zD=0.0; int zI=0; cudaMemcpyAsync(d_sumPrev,&zD,sizeof(double),cudaMemcpyHostToDevice,s); cudaMemcpyAsync(d_sumDv,&zD,sizeof(double),cudaMemcpyHostToDevice,s); cudaMemcpyAsync(d_hasNaN,&zI,sizeof(int),cudaMemcpyHostToDevice,s);
 LaunchVelocityDiag(bufs.d_vel_prev, bufs.d_vel, p.numParticles, d_sumPrev, d_sumDv, d_hasNaN, s);
 cudaStreamSynchronize(s);
 double hPrev=0.0, hDv=0.0; int hNaN=0; cudaMemcpy(&hPrev,d_sumPrev,sizeof(double),cudaMemcpyDeviceToHost); cudaMemcpy(&hDv,d_sumDv,sizeof(double),cudaMemcpyDeviceToHost); cudaMemcpy(&hNaN,d_hasNaN,sizeof(int),cudaMemcpyDeviceToHost);
 double ratio = (hPrev>0.0) ? (hDv / hPrev) :0.0;
 bool rollback=false;
 if (hNaN){ rollback=true; if (cc.debug.printWarnings) std::fprintf(stderr,"[Velocity][Rollback] NaN ratio=%.4g\n", ratio); }
 else if (ratio > cc.debug.velocityRollbackRatioMax){ rollback=true; if (cc.debug.printWarnings) std::fprintf(stderr,"[Velocity][Rollback] ratio=%.4g > %.3f\n", ratio, cc.debug.velocityRollbackRatioMax); }
 if (rollback){ cudaMemcpyAsync(bufs.d_vel, bufs.d_vel_prev, sizeof(float4)*p.numParticles, cudaMemcpyDeviceToDevice, s); if (cc.debug.printWarnings) std::fprintf(stderr,"[Velocity][Rollback] restored prev velocities\n"); }
 else if (cc.debug.printDiagnostics){ std::fprintf(stderr,"[Velocity][Diag] ratio=%.4g prevL1=%.4g dvL1=%.4g NaN=%d\n", ratio, hPrev, hDv, hNaN); }
 }

 // 半精爆散专项诊断：仅在 HalfArithmetic 分支 + 开启 logVelocityEffect 时执行
 if (preferHalfArithmetic && doDiag){
 static double *d_sumSpeed=nullptr, *d_sumRelErr=nullptr, *d_sumPosQuant=nullptr, *d_maxSpeed=nullptr, *d_maxRelErr=nullptr; static unsigned long long *d_overLimit=nullptr, *d_samples=nullptr;
 if (!d_sumSpeed){ cudaMalloc(&d_sumSpeed,sizeof(double)); cudaMalloc(&d_sumRelErr,sizeof(double)); cudaMalloc(&d_sumPosQuant,sizeof(double)); cudaMalloc(&d_maxSpeed,sizeof(double)); cudaMalloc(&d_maxRelErr,sizeof(double)); cudaMalloc(&d_overLimit,sizeof(unsigned long long)); cudaMalloc(&d_samples,sizeof(unsigned long long)); }
 double z=0.0; unsigned long long z64=0ull; cudaMemcpyAsync(d_sumSpeed,&z,sizeof(double),cudaMemcpyHostToDevice,s); cudaMemcpyAsync(d_sumRelErr,&z,sizeof(double),cudaMemcpyHostToDevice,s); cudaMemcpyAsync(d_sumPosQuant,&z,sizeof(double),cudaMemcpyHostToDevice,s); cudaMemcpyAsync(d_maxSpeed,&z,sizeof(double),cudaMemcpyHostToDevice,s); cudaMemcpyAsync(d_maxRelErr,&z,sizeof(double),cudaMemcpyHostToDevice,s); cudaMemcpyAsync(d_overLimit,&z64,sizeof(unsigned long long),cudaMemcpyHostToDevice,s); cudaMemcpyAsync(d_samples,&z64,sizeof(unsigned long long),cudaMemcpyHostToDevice,s);
 int stride = (cc.debug.logSampleStride > 0)? cc.debug.logSampleStride : 64; // 采样步长
 float kLimit = 6.0f; // 可调：速度阈值 = kLimit * h/dt
 LaunchVelocityExplodeDiag(bufs.d_vel, bufs.d_pos, bufs.d_pos_pred, bufs.d_pos_h4, p.numParticles, 1.0f/p.dt, p.kernel.h, stride, kLimit,
 d_sumSpeed, d_sumRelErr, d_sumPosQuant, d_maxSpeed, d_maxRelErr, d_overLimit, d_samples, s);
 cudaStreamSynchronize(s);
 double hSumSpeed=0.0, hSumRelErr=0.0, hSumPosQuant=0.0, hMaxSpeed=0.0, hMaxRelErr=0.0; unsigned long long hOver=0ull, hSamples=0ull;
 cudaMemcpy(&hSumSpeed,d_sumSpeed,sizeof(double),cudaMemcpyDeviceToHost); cudaMemcpy(&hSumRelErr,d_sumRelErr,sizeof(double),cudaMemcpyDeviceToHost); cudaMemcpy(&hSumPosQuant,d_sumPosQuant,sizeof(double),cudaMemcpyDeviceToHost); cudaMemcpy(&hMaxSpeed,d_maxSpeed,sizeof(double),cudaMemcpyDeviceToHost); cudaMemcpy(&hMaxRelErr,d_maxRelErr,sizeof(double),cudaMemcpyDeviceToHost); cudaMemcpy(&hOver,d_overLimit,sizeof(unsigned long long),cudaMemcpyDeviceToHost); cudaMemcpy(&hSamples,d_samples,sizeof(unsigned long long),cudaMemcpyDeviceToHost);
 double invS = (hSamples>0)? (1.0/double(hSamples)) : 0.0; double meanSpeed = hSumSpeed*invS; double meanRelErr = hSumRelErr*invS; double meanPosQuant = hSumPosQuant*invS; double fracOver = (hSamples>0)? (double(hOver)/double(hSamples)) : 0.0;
 if (cc.debug.printDiagnostics){ std::fprintf(stderr,"[VelocityHalf][ExplodeDiag] samples=%llu stride=%d meanSpeed=%.4g maxSpeed=%.4g limit=%.4g fracOver=%.3f meanRelErr=%.3f maxRelErr=%.3f meanPosQuant=%.3g\n", (unsigned long long)hSamples, stride, meanSpeed, hMaxSpeed, 6.0 * p.kernel.h / p.dt, fracOver, meanRelErr, hMaxRelErr, meanPosQuant); }
 if (cc.debug.printWarnings){ if (fracOver>0.2 || hMaxSpeed > 10.0 * p.kernel.h / p.dt || meanRelErr>0.3){ std::fprintf(stderr,"[VelocityHalf][Warn] potential explosion: fracOver=%.3f maxSpeed=%.4g relErrAvg=%.3f relErrMax=%.3f\n", fracOver, hMaxSpeed, meanRelErr, hMaxRelErr); } }
 }
}

static KernelDispatcher g_dispatcherInit;

static void RegisterBaseKernels(KernelDispatcher& kd){
 kd.registerKernel(PhaseType::Integrate,false,false,false,KernelIntegrate);
 kd.registerKernel(PhaseType::Integrate,true,true,false,KernelIntegrate);
 kd.registerKernel(PhaseType::Velocity,false,false,false,KernelVelocity);
 kd.registerKernel(PhaseType::Velocity,true,true,false,KernelVelocity);
}

void InitializeKernelDispatcher(KernelDispatcher& kd){ RegisterBaseKernels(kd); }

} // namespace sim
