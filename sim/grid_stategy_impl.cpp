#include "grid_strategy_dense.h"
#include "grid_strategy_hashed.h"
#include "simulation_context.h"
#include "precision_stage.h"
#include "parameters.h"
#include "device_buffers.cuh"
#include "grid_buffers.cuh"

extern "C" {
    void LaunchLambda(float*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
    void LaunchDeltaApply(float4*, float4*, const float*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
    void LaunchLambdaMP(float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
    void LaunchDeltaApplyMP(float4*, float4*, const float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);

    void LaunchLambdaCompact(float*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
    void LaunchDeltaApplyCompact(float4*, float4*, const float*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
    void LaunchLambdaCompactMP(float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
    void LaunchDeltaApplyCompactMP(float4*, float4*, const float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);

    void LaunchBoundary(float4*, float4*, sim::GridBounds, float, uint32_t, cudaStream_t);
}

namespace sim {

    static inline DeviceParams MakeDP(const SimParams& p) { return MakeDeviceParams(p); }

    void DenseGridStrategy::solveIter(SimulationContext& ctx, const SimParams& p, cudaStream_t s, int /*iterIndex*/, KernelDispatcher& /*kd*/) {
        DeviceParams dp = MakeDP(p);
        bool useMP = UseHalfForPosition(p, Stage::LambdaSolve, *ctx.bufs);
        if (useMP) {
            LaunchLambdaMP(ctx.bufs->d_lambda, ctx.bufs->d_pos_pred, ctx.bufs->d_pos_pred_h4,
                ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                dp, p.numParticles, s);
            LaunchDeltaApplyMP(ctx.bufs->d_pos_pred, ctx.bufs->d_delta, ctx.bufs->d_lambda,
                ctx.bufs->d_pos_pred, ctx.bufs->d_pos_pred_h4,
                ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                dp, p.numParticles, s);
        }
        else {
            LaunchLambda(ctx.bufs->d_lambda, ctx.bufs->d_pos_pred,
                ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                dp, p.numParticles, s);
            LaunchDeltaApply(ctx.bufs->d_pos_pred, ctx.bufs->d_delta, ctx.bufs->d_lambda,
                ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                dp, p.numParticles, s);
        }
        // 与原版一致：迭代内对预测位置做一次无弹性边界钳制
        LaunchBoundary(ctx.bufs->d_pos_pred, ctx.bufs->d_vel, p.grid, 0.0f, p.numParticles, s);
    }

    void HashedGridStrategy::solveIter(SimulationContext& ctx, const SimParams& p, cudaStream_t s, int /*iterIndex*/, KernelDispatcher& /*kd*/) {
        DeviceParams dp = MakeDP(p);
        bool useMP = UseHalfForPosition(p, Stage::LambdaSolve, *ctx.bufs);
        if (useMP) {
            LaunchLambdaCompactMP(ctx.bufs->d_lambda, ctx.bufs->d_pos_pred, ctx.bufs->d_pos_pred_h4,
                ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                dp, p.numParticles, s);
            LaunchDeltaApplyCompactMP(ctx.bufs->d_pos_pred, ctx.bufs->d_delta, ctx.bufs->d_lambda,
                ctx.bufs->d_pos_pred, ctx.bufs->d_pos_pred_h4,
                ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                dp, p.numParticles, s);
        }
        else {
            LaunchLambdaCompact(ctx.bufs->d_lambda, ctx.bufs->d_pos_pred,
                ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                dp, p.numParticles, s);
            LaunchDeltaApplyCompact(ctx.bufs->d_pos_pred, ctx.bufs->d_delta, ctx.bufs->d_lambda,
                ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                dp, p.numParticles, s);
        }
        LaunchBoundary(ctx.bufs->d_pos_pred, ctx.bufs->d_vel, p.grid, 0.0f, p.numParticles, s);
    }

} // namespace sim