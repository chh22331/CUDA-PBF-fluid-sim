#include "phase_pipeline.h"
#include "simulation_context.h"
#include "precision_stage.h"
#include "kernel_dispatcher.h"
#include "grid_strategy.h"
#include "grid_buffers.cuh"
#include "../engine/core/console.h"

extern "C" void LaunchIntegratePred(float4*, const float4*, float4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchIntegratePredMP(const float4*, const float4*, float4*, const sim::Half4*, const sim::Half4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocity(float4*, const float4*, const float4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocityMP(float4*, const float4*, const float4*, const sim::Half4*, const sim::Half4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchBoundary(float4*, float4*, sim::GridBounds, float, uint32_t, uint32_t, uint8_t, cudaStream_t);
extern "C" void LaunchBoundaryHalf(float4*, float4*, const sim::Half4*, const sim::Half4*, sim::GridBounds, float, uint32_t, uint32_t, bool, uint8_t, cudaStream_t);
extern "C" void LaunchHashKeys(uint32_t*, uint32_t*, const float4*, sim::GridBounds, uint32_t, cudaStream_t);
extern "C" void LaunchHashKeysMP(uint32_t*, uint32_t*, const float4*, const sim::Half4*, sim::GridBounds, uint32_t, cudaStream_t);
extern "C" void LaunchSortPairs(void*, size_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchCellRanges(uint32_t*, uint32_t*, const uint32_t*, uint32_t, uint32_t, cudaStream_t);
extern "C" void LaunchCellRangesCompact(uint32_t*, uint32_t*, uint32_t*, const uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchLambda(float*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaCompact(float*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaCompactMP(float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaMP(float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApply(float4*, float4*, const float*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyCompact(float4*, float4*, const float*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyCompactMP(float4*, float4*, const float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyMP(float4*, float4*, const float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);

namespace sim {
 static inline uint8_t GhostClampFlag(){ const auto& c = console::Instance(); const auto& g = c.sim.boundaryGhost; return (g.enable ? (g.place_outside ?0u :1u) :0u); }
 static inline void DispatchBoundaryPhase(SimulationContext& ctx, const SimParams& p, float4* posPred, float4* vel, float restitution, cudaStream_t s){
 uint32_t N = p.numParticles; if (N==0) return;
 bool useHalfPos = UseHalfForPosition(p, Stage::Boundary, *ctx.bufs);
 bool useHalfVel = UseHalfForVelocity(p, Stage::Boundary, *ctx.bufs);
 bool canHalf = ((useHalfPos && useHalfVel && ctx.bufs->d_pos_pred_h4 && ctx.bufs->d_vel_h4) || ctx.bufs->nativeHalfActive);
 uint8_t ghostClamp = GhostClampFlag(); uint32_t ghostCount = p.ghostParticleCount;
 if (canHalf){ if (!ctx.bufs->nativeHalfActive) ctx.bufs->packAllToHalf(N, s); LaunchBoundaryHalf(posPred, vel, ctx.bufs->d_pos_pred_h4, ctx.bufs->d_vel_h4, p.grid, restitution, N, ghostCount, p.precision.forceFp32Accumulate, ghostClamp, s); }
 else { LaunchBoundary(posPred, vel, p.grid, restitution, N, ghostCount, ghostClamp, s); }
 }

 class PhaseIntegrate : public IPhase { public: PhaseType type() const override { return PhaseType::Integrate; } void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {
 bool useMP = (UseHalfForPosition(p, Stage::Integration, *ctx.bufs) && UseHalfForVelocity(p, Stage::Integration, *ctx.bufs));
 if (useMP) LaunchIntegratePredMP(ctx.bufs->d_pos, ctx.bufs->d_vel, ctx.bufs->d_pos_pred, ctx.bufs->d_pos_h4, ctx.bufs->d_vel_h4, p.gravity, p.dt, p.numParticles, s);
 else LaunchIntegratePred(ctx.bufs->d_pos, ctx.bufs->d_vel, ctx.bufs->d_pos_pred, p.gravity, p.dt, p.numParticles, s);
 DispatchBoundaryPhase(ctx, p, ctx.bufs->d_pos_pred, ctx.bufs->d_vel,0.0f, s);
 } };

 class PhaseGridBuildFull : public IPhase { public: PhaseType type() const override { return PhaseType::GridBuild; } void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {
 bool useMP = UseHalfForPosition(p, Stage::GridBuild, *ctx.bufs);
 if (useMP) LaunchHashKeysMP(ctx.grid->d_cellKeys, ctx.grid->d_indices, ctx.bufs->d_pos_pred, ctx.bufs->d_pos_pred_h4, p.grid, p.numParticles, s);
 else LaunchHashKeys(ctx.grid->d_cellKeys, ctx.grid->d_indices, ctx.bufs->d_pos_pred, p.grid, p.numParticles, s);
 LaunchSortPairs(ctx.grid->d_sortTemp, ctx.grid->sortTempBytes, ctx.grid->d_cellKeys, ctx.grid->d_cellKeys_sorted, ctx.grid->d_indices, ctx.grid->d_indices_sorted, p.numParticles, s);
 if (ctx.useHashedGrid) LaunchCellRangesCompact(ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount, ctx.grid->d_cellKeys_sorted, p.numParticles, s);
 else { cudaMemsetAsync(ctx.grid->d_cellStart,0xFF,sizeof(uint32_t)*ctx.grid->numCells,s); cudaMemsetAsync(ctx.grid->d_cellEnd,0xFF,sizeof(uint32_t)*ctx.grid->numCells,s); LaunchCellRanges(ctx.grid->d_cellStart, ctx.grid->d_cellEnd, ctx.grid->d_cellKeys_sorted, p.numParticles, ctx.grid->numCells, s); }
 } };

 class PhaseSolveIterations : public IPhase { public: PhaseType type() const override { return PhaseType::SolveIterations; } void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {
 for(int iter=0; iter < p.solverIters; ++iter){ DeviceParams dp = MakeDeviceParams(p); if (ctx.useHashedGrid){ bool useMP = UseHalfForPosition(p, Stage::LambdaSolve, *ctx.bufs); if (useMP){ LaunchLambdaCompactMP(ctx.bufs->d_lambda, ctx.bufs->d_pos_pred, ctx.bufs->d_pos_pred_h4, ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted, ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount, dp, p.numParticles, s); LaunchDeltaApplyCompactMP(ctx.bufs->d_pos_pred, ctx.bufs->d_delta, ctx.bufs->d_lambda, ctx.bufs->d_pos_pred, ctx.bufs->d_pos_pred_h4, ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted, ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount, dp, p.numParticles, s); } else { LaunchLambdaCompact(ctx.bufs->d_lambda, ctx.bufs->d_pos_pred, ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted, ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount, dp, p.numParticles, s); LaunchDeltaApplyCompact(ctx.bufs->d_pos_pred, ctx.bufs->d_delta, ctx.bufs->d_lambda, ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted, ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount, dp, p.numParticles, s); } }
 else { bool useMP = UseHalfForPosition(p, Stage::LambdaSolve, *ctx.bufs); if (useMP){ LaunchLambdaMP(ctx.bufs->d_lambda, ctx.bufs->d_pos_pred, ctx.bufs->d_pos_pred_h4, ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted, ctx.grid->d_cellStart, ctx.grid->d_cellEnd, dp, p.numParticles, s); LaunchDeltaApplyMP(ctx.bufs->d_pos_pred, ctx.bufs->d_delta, ctx.bufs->d_lambda, ctx.bufs->d_pos_pred, ctx.bufs->d_pos_pred_h4, ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted, ctx.grid->d_cellStart, ctx.grid->d_cellEnd, dp, p.numParticles, s); } else { LaunchLambda(ctx.bufs->d_lambda, ctx.bufs->d_pos_pred, ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted, ctx.grid->d_cellStart, ctx.grid->d_cellEnd, dp, p.numParticles, s); LaunchDeltaApply(ctx.bufs->d_pos_pred, ctx.bufs->d_delta, ctx.bufs->d_lambda, ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted, ctx.grid->d_cellStart, ctx.grid->d_cellEnd, dp, p.numParticles, s); } }
 DispatchBoundaryPhase(ctx, p, ctx.bufs->d_pos_pred, ctx.bufs->d_vel,0.0f, s); }
 } };

 class PhaseVelocity : public IPhase { public: PhaseType type() const override { return PhaseType::Velocity; } void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override { bool useMP = UseHalfForPosition(p, Stage::VelocityUpdate, *ctx.bufs); if (useMP) LaunchVelocityMP(ctx.bufs->d_vel, ctx.bufs->d_pos, ctx.bufs->d_pos_pred, ctx.bufs->d_pos_h4, ctx.bufs->d_pos_pred_h4,1.0f/p.dt, p.numParticles, s); else LaunchVelocity(ctx.bufs->d_vel, ctx.bufs->d_pos, ctx.bufs->d_pos_pred,1.0f/p.dt, p.numParticles, s); } };

 void BuildDefaultPipelines(PhasePipeline& pipeline){ pipeline.addFull<PhaseIntegrate>(); pipeline.addFull<PhaseGridBuildFull>(); pipeline.addFull<PhaseSolveIterations>(); pipeline.addFull<PhaseVelocity>(); pipeline.addCheap<PhaseIntegrate>(); pipeline.addCheap<PhaseSolveIterations>(); pipeline.addCheap<PhaseVelocity>(); }

} // namespace sim