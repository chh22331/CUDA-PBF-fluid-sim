#include "phase_pipeline.h"
#include "simulation_context.h"
#include "precision_stage.h"
#include "kernel_dispatcher.h"
#include "grid_strategy.h"
#include "grid_buffers.cuh"

extern "C" void LaunchIntegratePred(float4*, const float4*, float4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchIntegratePredMP(const float4*, const float4*, float4*, const sim::Half4*, const sim::Half4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocity(float4*, const float4*, const float4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocityMP(float4*, const float4*, const float4*, const sim::Half4*, const sim::Half4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchBoundary(float4*, float4*, sim::GridBounds, float, uint32_t, cudaStream_t);
extern "C" void LaunchHashKeys(uint32_t*, uint32_t*, const float4*, sim::GridBounds, uint32_t, cudaStream_t);
extern "C" void LaunchHashKeysMP(uint32_t*, uint32_t*, const float4*, const sim::Half4*, sim::GridBounds, uint32_t, cudaStream_t);
extern "C" void LaunchSortPairs(void*, size_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchCellRanges(uint32_t*, uint32_t*, const uint32_t*, uint32_t, uint32_t, cudaStream_t);
extern "C" void LaunchCellRangesCompact(uint32_t*, uint32_t*, uint32_t*, const uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchLambda(float*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaCompact(float*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaCompactMP(float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, uint32_t, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaMP(float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApply(float4*, float4*, const float*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyCompact(float4*, float4*, const float*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyCompactMP(float4*, float4*, const float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, uint32_t, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyMP(float4*, float4*, const float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);

namespace sim {

class PhaseIntegrate : public IPhase {
public:
    PhaseType type() const override { return PhaseType::Integrate; }
    void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {
        bool useMP = (UseHalfForPosition(p, Stage::Integration, *ctx.bufs) &&
                      UseHalfForVelocity(p, Stage::Integration, *ctx.bufs));
        if (useMP) {
            LaunchIntegratePredMP(ctx.bufs->d_pos_curr, ctx.bufs->d_vel, ctx.bufs->d_pos_next,
                                  ctx.bufs->d_pos_curr_h4, ctx.bufs->d_vel_h4,
                                  p.gravity, p.dt, p.numParticles, s);
        } else {
            LaunchIntegratePred(ctx.bufs->d_pos_curr, ctx.bufs->d_vel, ctx.bufs->d_pos_next,
                                p.gravity, p.dt, p.numParticles, s);
        }
        LaunchBoundary(ctx.bufs->d_pos_next, ctx.bufs->d_vel, p.grid, 0.0f, p.numParticles, s);
    }
};

class PhaseGridBuild : public IPhase {
public:
    PhaseType type() const override { return PhaseType::GridBuild; }
    void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {
        // 统一保留完整重建（后续可加条件跳过）
        bool useMP = UseHalfForPosition(p, Stage::GridBuild, *ctx.bufs);
        if (useMP)
            LaunchHashKeysMP(ctx.grid->d_cellKeys, ctx.grid->d_indices,
                             ctx.bufs->d_pos_next, ctx.bufs->d_pos_next_h4,
                             p.grid, p.numParticles, s);
        else
            LaunchHashKeys(ctx.grid->d_cellKeys, ctx.grid->d_indices,
                           ctx.bufs->d_pos_next, p.grid, p.numParticles, s);

        LaunchSortPairs(ctx.grid->d_sortTemp, ctx.grid->sortTempBytes,
                        ctx.grid->d_cellKeys, ctx.grid->d_cellKeys_sorted,
                        ctx.grid->d_indices, ctx.grid->d_indices_sorted,
                        p.numParticles, s);

        if (ctx.useHashedGrid) {
            LaunchCellRangesCompact(ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                                    ctx.grid->d_cellKeys_sorted, p.numParticles, s);
        } else {
            cudaMemsetAsync(ctx.grid->d_cellStart, 0xFF, sizeof(uint32_t) * ctx.grid->numCells, s);
            cudaMemsetAsync(ctx.grid->d_cellEnd, 0xFF, sizeof(uint32_t) * ctx.grid->numCells, s);
            LaunchCellRanges(ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                             ctx.grid->d_cellKeys_sorted, p.numParticles, ctx.grid->numCells, s);
        }
    }
};

class PhaseSolveIterations : public IPhase {
public:
    PhaseType type() const override { return PhaseType::SolveIterations; }
    void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {
        for (int iter = 0; iter < p.solverIters; ++iter) {
            DeviceParams dp = MakeDeviceParams(p);
            if (ctx.useHashedGrid) {
                bool useMP = UseHalfForPosition(p, Stage::LambdaSolve, *ctx.bufs);
                const uint32_t* keyMap = (p.compactBinarySearch ? nullptr : ctx.grid->d_keyToCompact);
                uint32_t numCells = ctx.grid->numCells;
                if (useMP) {
                    LaunchLambdaCompactMP(ctx.bufs->d_lambda, ctx.bufs->d_pos_next, ctx.bufs->d_pos_next_h4,
                                          ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                                          ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                                          keyMap, numCells, dp, p.numParticles, s);
                    LaunchDeltaApplyCompactMP(ctx.bufs->d_pos_next, ctx.bufs->d_delta, ctx.bufs->d_lambda,
                                              ctx.bufs->d_pos_next, ctx.bufs->d_pos_next_h4,
                                              ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                                              ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                                              keyMap, numCells, dp, p.numParticles, s);
                } else {
                    LaunchLambdaCompact(ctx.bufs->d_lambda, ctx.bufs->d_pos_next,
                                        ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                                        ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                                        dp, p.numParticles, s);
                    LaunchDeltaApplyCompact(ctx.bufs->d_pos_next, ctx.bufs->d_delta, ctx.bufs->d_lambda,
                                            ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                                            ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                                            dp, p.numParticles, s);
                }
            } else {
                bool useMP = UseHalfForPosition(p, Stage::LambdaSolve, *ctx.bufs);
                if (useMP) {
                    LaunchLambdaMP(ctx.bufs->d_lambda, ctx.bufs->d_pos_next, ctx.bufs->d_pos_next_h4,
                                   ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                                   ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                                   dp, p.numParticles, s);
                    LaunchDeltaApplyMP(ctx.bufs->d_pos_next, ctx.bufs->d_delta, ctx.bufs->d_lambda,
                                       ctx.bufs->d_pos_next, ctx.bufs->d_pos_next_h4,
                                       ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                                       ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                                       dp, p.numParticles, s);
                } else {
                    LaunchLambda(ctx.bufs->d_lambda, ctx.bufs->d_pos_next,
                                 ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                                 ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                                 dp, p.numParticles, s);
                    LaunchDeltaApply(ctx.bufs->d_pos_next, ctx.bufs->d_delta, ctx.bufs->d_lambda,
                                     ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                                     ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                                     dp, p.numParticles, s);
                }
            }
            LaunchBoundary(ctx.bufs->d_pos_next, ctx.bufs->d_vel, p.grid, 0.0f, p.numParticles, s);
        }
    }
};

class PhaseVelocity : public IPhase {
public:
    PhaseType type() const override { return PhaseType::Velocity; }
    void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {
        bool useMP = UseHalfForPosition(p, Stage::VelocityUpdate, *ctx.bufs);
        if (useMP)
            LaunchVelocityMP(ctx.bufs->d_vel, ctx.bufs->d_pos_curr, ctx.bufs->d_pos_next,
                             ctx.bufs->d_pos_curr_h4, ctx.bufs->d_pos_next_h4,
                             1.0f / p.dt, p.numParticles, s);
        else
            LaunchVelocity(ctx.bufs->d_vel, ctx.bufs->d_pos_curr, ctx.bufs->d_pos_next,
                           1.0f / p.dt, p.numParticles, s);
    }
};

void BuildDefaultPipelines(PhasePipeline& pipeline) {
    // 单一序列（删除 addFull/addCheap）
    pipeline.addPhase<PhaseIntegrate>();
    pipeline.addPhase<PhaseGridBuild>();
    pipeline.addPhase<PhaseSolveIterations>();
    pipeline.addPhase<PhaseVelocity>();
}

} // namespace sim