#include "phase_pipeline.h"
#include "simulation_context.h"
#include "kernel_dispatcher.h"
#include "grid_strategy.h"
#include "grid_buffers.cuh"
#include "device_globals.cuh"

extern "C" void LaunchVelocityGlobals(float dtInv, uint32_t N, cudaStream_t);
extern "C" void LaunchHashKeys(uint32_t*, uint32_t*, const float4*, sim::GridBounds, uint32_t, cudaStream_t);
extern "C" void LaunchSortPairs(void*, size_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchCellRanges(uint32_t*, uint32_t*, const uint32_t*, uint32_t, uint32_t, cudaStream_t);
extern "C" void LaunchCellRangesCompact(uint32_t*, uint32_t*, uint32_t*, const uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchLambda(float*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);              // 新增
extern "C" void LaunchLambdaCompact(float*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApply(float4*, float4*, const float*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);    // 新增
extern "C" void LaunchDeltaApplyCompact(float4*, float4*, const float*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaDenseGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaCompactGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyDenseGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyCompactGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXsphDenseGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXsphCompactGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchIntegratePred(float4*, const float4*, float4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocity(float4*, const float4*, const float4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchBoundary(float4*, float4*, sim::GridBounds, float, uint32_t, cudaStream_t);

namespace sim {

// Integrate + clamp
class PhaseIntegrate : public IPhase {
public:
    PhaseType type() const override { return PhaseType::Integrate; }
    void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {
        LaunchIntegratePred(ctx.bufs->d_pos, ctx.bufs->d_vel, ctx.bufs->d_pos_pred,
                            p.gravity, p.dt, p.numParticles, s);
        LaunchBoundary(ctx.bufs->d_pos_pred, ctx.bufs->d_vel, p.grid, 0.0f, p.numParticles, s);
    }
};

// Full grid build phase
class PhaseGridBuildFull : public IPhase {
public:
    PhaseType type() const override { return PhaseType::GridBuild; }
    void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {
         
        LaunchHashKeys(ctx.grid->d_cellKeys, ctx.grid->d_indices,
                       ctx.bufs->d_pos_pred, p.grid, p.numParticles, s);

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

// Solve iterations
class PhaseSolveIterations : public IPhase {
public:
    PhaseType type() const override { return PhaseType::SolveIterations; }
    void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {
        for (int iter = 0; iter < p.solverIters; ++iter) {
            DeviceParams dp = MakeDeviceParams(p);
            if (ctx.useHashedGrid) {                
                LaunchLambdaCompact(ctx.bufs->d_lambda, ctx.bufs->d_pos_pred,
                                    ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                                    ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                                    dp, p.numParticles, s);
                LaunchDeltaApplyCompact(ctx.bufs->d_pos_pred, ctx.bufs->d_delta, ctx.bufs->d_lambda,
                                        ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                                        ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                                        dp, p.numParticles, s);              
            } else {
                 
                LaunchLambda(ctx.bufs->d_lambda, ctx.bufs->d_pos_pred,
                             ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                             ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                             dp, p.numParticles, s);
                LaunchDeltaApply(ctx.bufs->d_pos_pred, ctx.bufs->d_delta, ctx.bufs->d_lambda,
                                 ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                                 ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                                 dp, p.numParticles, s);
                }
            }
            LaunchBoundary(ctx.bufs->d_pos_pred, ctx.bufs->d_vel, p.grid, 0.0f, p.numParticles, s);
        }
    };

// Velocity
class PhaseVelocity : public IPhase {
public:
    PhaseType type() const override { return PhaseType::Velocity; }
    void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {         
        LaunchVelocity(ctx.bufs->d_vel, ctx.bufs->d_pos, ctx.bufs->d_pos_pred,
                       1.0f / p.dt, p.numParticles, s);
    }
};

void BuildDefaultPipelines(PhasePipeline& pipeline) {
    pipeline.addFull<PhaseIntegrate>();
    pipeline.addFull<PhaseGridBuildFull>();
    pipeline.addFull<PhaseSolveIterations>();
    pipeline.addFull<PhaseVelocity>();

    pipeline.addCheap<PhaseIntegrate>();
    pipeline.addCheap<PhaseSolveIterations>();
    pipeline.addCheap<PhaseVelocity>();
}

} // namespace sim
