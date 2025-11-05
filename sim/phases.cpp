#include "phase_pipeline.h"
#include "simulation_context.h"
#include "precision_stage.h"
#include "kernel_dispatcher.h"
#include "grid_strategy.h"
#include "grid_buffers.cuh"
#include "device_globals.cuh"        // 新增

// 旧外部核声明保留（其它阶段尚未迁移）
extern "C" void LaunchIntegratePredGlobals(float3 gravity, float dt, uint32_t N, cudaStream_t);
extern "C" void LaunchVelocityGlobals(float dtInv, uint32_t N, cudaStream_t);
extern "C" void LaunchHashKeys(uint32_t*, uint32_t*, const float4*, sim::GridBounds, uint32_t, cudaStream_t);
extern "C" void LaunchHashKeysMP(uint32_t*, uint32_t*, const float4*, const sim::Half4*, sim::GridBounds, uint32_t, cudaStream_t);
extern "C" void LaunchSortPairs(void*, size_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchCellRanges(uint32_t*, uint32_t*, const uint32_t*, uint32_t, uint32_t, cudaStream_t);
extern "C" void LaunchCellRangesCompact(uint32_t*, uint32_t*, uint32_t*, const uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaCompact(float*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaCompactMP(float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, uint32_t, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaMP(float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyCompact(float4*, float4*, const float*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyCompactMP(float4*, float4*, const float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, uint32_t, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyMP(float4*, float4*, const float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaDenseGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaCompactGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyDenseGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyCompactGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXsphDenseGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXsphCompactGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchBoundaryGlobals(sim::GridBounds, float, uint32_t, bool, cudaStream_t);
namespace sim {

class PhaseIntegrate : public IPhase {
public:
    PhaseType type() const override { return PhaseType::Integrate; }
    void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {
        LaunchIntegratePredGlobals(p.gravity, p.dt, p.numParticles, s);
        LaunchBoundaryGlobals(p.grid, 0.0f, p.numParticles, false, s);
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
                LaunchLambdaCompactGlobals(ctx.grid->d_indices_sorted,
                    ctx.grid->d_cellKeys_sorted,
                    ctx.grid->d_cellUniqueKeys,
                    ctx.grid->d_cellOffsets,
                    ctx.grid->d_compactCount,
                    dp, p.numParticles, s);
                LaunchDeltaApplyCompactGlobals(ctx.grid->d_indices_sorted,
                    ctx.grid->d_cellKeys_sorted,
                    ctx.grid->d_cellUniqueKeys,
                    ctx.grid->d_cellOffsets,
                    ctx.grid->d_compactCount,
                    dp, p.numParticles, s);
            }
            else {
                LaunchLambdaDenseGlobals(ctx.grid->d_indices_sorted,
                    ctx.grid->d_cellKeys_sorted,
                    ctx.grid->d_cellStart,
                    ctx.grid->d_cellEnd,
                    dp, p.numParticles, s);
                LaunchDeltaApplyDenseGlobals(ctx.grid->d_indices_sorted,
                    ctx.grid->d_cellKeys_sorted,
                    ctx.grid->d_cellStart,
                    ctx.grid->d_cellEnd,
                    dp, p.numParticles, s);
            }
            // 每迭代后边界（约束后，无 XSPH）
            LaunchBoundaryGlobals(p.grid, 0.0f, p.numParticles, false, s);
        }
        // 迭代完成后可选择性执行 XSPH 平滑（一次）
        if (p.xsph_c > 0.f && p.numParticles > 0) {
            DeviceParams dp = MakeDeviceParams(p);
            if (ctx.useHashedGrid) {
                LaunchXsphCompactGlobals(ctx.grid->d_indices_sorted,
                    ctx.grid->d_cellKeys_sorted,
                    ctx.grid->d_cellUniqueKeys,
                    ctx.grid->d_cellOffsets,
                    ctx.grid->d_compactCount,
                    dp, p.numParticles, s);
            }
            else {
                LaunchXsphDenseGlobals(ctx.grid->d_indices_sorted,
                    ctx.grid->d_cellKeys_sorted,
                    ctx.grid->d_cellStart,
                    ctx.grid->d_cellEnd,
                    dp, p.numParticles, s);
            }
            // XSPH 后再做一次带 restitution 的边界修正
            LaunchBoundaryGlobals(p.grid, p.boundaryRestitution, p.numParticles, true, s);
        }
    }
};

class PhaseVelocity : public IPhase {
public:
    PhaseType type() const override { return PhaseType::Velocity; }
    void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {
        LaunchVelocityGlobals(1.0f / p.dt, p.numParticles, s);
    }
};

// 其它 PhaseGridBuild / PhaseSolveIterations 保持原样（未展示部分不改）

void BuildDefaultPipelines(PhasePipeline& pipeline) {
    pipeline.addPhase<PhaseIntegrate>();
    pipeline.addPhase<PhaseGridBuild>();
    pipeline.addPhase<PhaseSolveIterations>();
    pipeline.addPhase<PhaseVelocity>();
}

} // namespace sim