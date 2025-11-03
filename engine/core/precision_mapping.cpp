#include "precision_mapping.h"
#include "console.h"
#include <cstdio>

namespace console {

static inline bool ShouldAutoMapMixedInternal(const RuntimeConsole::Simulation& s) {
 const auto& pc = s.precision;
 if (!s.useMixedPrecision) return false;
 if (!pc.autoMapFromLegacyMixedFlag) return false;
 bool allDefault =
 pc.positionStore == NumericType::FP32 &&
 pc.velocityStore == NumericType::FP32 &&
 pc.predictedPosStore == NumericType::FP32 &&
 pc.lambdaStore == NumericType::FP32 &&
 pc.densityStore == NumericType::FP32 &&
 pc.auxStore == NumericType::FP32 &&
 pc.renderTransfer == NumericType::FP32 &&
 pc.coreCompute == NumericType::FP32 &&
 !pc.useStageOverrides &&
 pc.fp16StageMask ==0;
 return allDefault;
}

static inline sim::NumericType ConvertNt(console::NumericType t) {
 return static_cast<sim::NumericType>(static_cast<uint8_t>(t));
}

static inline void PromotePlainHalf(sim::NumericType& t) {
 if (t == sim::NumericType::FP16) t = sim::NumericType::FP16_Packed;
}

sim::SimPrecision MapPrecision(const RuntimeConsole::Simulation& simCfg) {
 sim::SimPrecision out{};
 const auto& src = simCfg.precision;

 if (ShouldAutoMapMixedInternal(simCfg)) {
 out.positionStore = sim::NumericType::FP16_Packed;
 out.velocityStore = sim::NumericType::FP16_Packed;
 out.predictedPosStore = sim::NumericType::FP16_Packed;
 out.lambdaStore = sim::NumericType::FP32;
 out.densityStore = sim::NumericType::FP32;
 out.auxStore = sim::NumericType::FP16_Packed;
 out.renderTransfer = sim::NumericType::FP32;
 out.coreCompute = sim::NumericType::FP32;
 out.forceFp32Accumulate = true;
 } else {
 out.positionStore = ConvertNt(src.positionStore);
 out.velocityStore = ConvertNt(src.velocityStore);
 out.predictedPosStore = ConvertNt(src.predictedPosStore);
 out.lambdaStore = ConvertNt(src.lambdaStore);
 out.densityStore = ConvertNt(src.densityStore);
 out.auxStore = ConvertNt(src.auxStore);
 out.renderTransfer = ConvertNt(src.renderTransfer);
 out.coreCompute = ConvertNt(src.coreCompute);
 out.forceFp32Accumulate = src.forceFp32Accumulate;
 out.enableHalfIntrinsics = src.enableHalfIntrinsics;
 out.useStageOverrides = src.useStageOverrides;
 out.emissionCompute = ConvertNt(src.emissionCompute);
 out.gridBuildCompute = ConvertNt(src.gridBuildCompute);
 out.neighborCompute = ConvertNt(src.neighborCompute);
 out.densityCompute = ConvertNt(src.densityCompute);
 out.lambdaCompute = ConvertNt(src.lambdaCompute);
 out.integrateCompute = ConvertNt(src.integrateCompute);
 out.velocityCompute = ConvertNt(src.velocityCompute);
 out.boundaryCompute = ConvertNt(src.boundaryCompute);
 out.xsphCompute = ConvertNt(src.xsphCompute);
 out.fp16StageMask = src.fp16StageMask;
 out.adaptivePrecision = src.adaptivePrecision;
 out.densityErrorTolerance = src.densityErrorTolerance;
 out.lambdaVarianceTolerance = src.lambdaVarianceTolerance;
 out.adaptCheckEveryN = src.adaptCheckEveryN;
 PromotePlainHalf(out.positionStore);
 PromotePlainHalf(out.velocityStore);
 PromotePlainHalf(out.predictedPosStore);
 PromotePlainHalf(out.auxStore);
 // lambda / density 保持标量 half (不强制 Packed)
 }

 // 阶段覆盖 / fp16StageMask逻辑整合
 if (!src.useStageOverrides) {
 sim::NumericType coreT = ConvertNt(src.coreCompute);
 auto wantFp16 = [&](uint32_t bit){ return (src.fp16StageMask & bit) !=0; };
 auto choose = [&](uint32_t bit)->sim::NumericType { return wantFp16(bit) ? sim::NumericType::FP16 : coreT; };
 using B = console::ComputeStageBits;
 out.emissionCompute = choose(B::Stage_Emission);
 out.gridBuildCompute = choose(B::Stage_GridBuild);
 out.neighborCompute = choose(B::Stage_NeighborGather);
 out.densityCompute = choose(B::Stage_Density);
 out.lambdaCompute = choose(B::Stage_LambdaSolve);
 out.integrateCompute = choose(B::Stage_Integration);
 out.velocityCompute = choose(B::Stage_VelocityUpdate);
 out.boundaryCompute = choose(B::Stage_Boundary);
 out.xsphCompute = choose(B::Stage_XSPH);
 }

 // 原生 half 激活判定
 if (src.nativeHalfPrefer) {
 bool posHalf = (out.positionStore == sim::NumericType::FP16_Packed);
 bool velHalf = (out.velocityStore == sim::NumericType::FP16_Packed);
 bool predHalf= (out.predictedPosStore == sim::NumericType::FP16_Packed);
 if (posHalf && velHalf && predHalf) out.nativeHalfActive = true;
 }

 //诊断（后续统一迁移到 diagnostics dispatcher）
 if (console::Instance().debug.printDiagnostics) {
 static sim::SimPrecision s_prev{};
 auto diff = [&](){
 return s_prev.positionStore != out.positionStore ||
 s_prev.velocityStore != out.velocityStore ||
 s_prev.predictedPosStore != out.predictedPosStore ||
 s_prev.nativeHalfActive != out.nativeHalfActive ||
 s_prev.forceFp32Accumulate != out.forceFp32Accumulate ||
 s_prev.useStageOverrides != out.useStageOverrides ||
 s_prev.integrateCompute != out.integrateCompute ||
 s_prev.velocityCompute != out.velocityCompute ||
 s_prev.boundaryCompute != out.boundaryCompute ||
 s_prev.xsphCompute != out.xsphCompute; };
 if (diff()) {
 std::fprintf(stderr,
 "[Precision.Map] pos=%u vel=%u pred=%u native=%d forceAcc=%d stageOv=%d integ=%u velC=%u bnd=%u xsph=%u aux=%u lambda=%u density=%u render=%u fp16Mask=0x%X\n",
 (unsigned)out.positionStore,
 (unsigned)out.velocityStore,
 (unsigned)out.predictedPosStore,
 (int)out.nativeHalfActive,
 (int)out.forceFp32Accumulate,
 (int)out.useStageOverrides,
 (unsigned)out.integrateCompute,
 (unsigned)out.velocityCompute,
 (unsigned)out.boundaryCompute,
 (unsigned)out.xsphCompute,
 (unsigned)out.auxStore,
 (unsigned)out.lambdaStore,
 (unsigned)out.densityStore,
 (unsigned)out.renderTransfer,
 out.fp16StageMask);
 s_prev = out;
 }
 }

 return out;
}

} // namespace console
