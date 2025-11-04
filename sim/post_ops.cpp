#include "post_ops.h"
#include "simulation_context.h"
#include "precision_stage.h"
#include "logging.h"
#include "stats.h"
#include <cstdio>
#include "../engine/core/prof_nvtx.h"

extern "C" void LaunchBoundary(float4*, float4*, sim::GridBounds, float, uint32_t, cudaStream_t);
extern "C" void LaunchRecycleToNozzleConst(float4*, float4*, float4*, sim::GridBounds, float, uint32_t, int, cudaStream_t);
extern "C" void LaunchXSPH(float4*, const float4*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHCompact(float4*, const float4*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHMP(float4*, const float4*, const sim::Half4*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHCompactMP(float4*, const float4*, const sim::Half4*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, uint32_t, sim::DeviceParams, uint32_t, cudaStream_t);

namespace sim {

static DeviceParams MakeDP(const SimParams& p){ return MakeDeviceParams(p); }

void BoundaryOp::run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
    if(!ctx.effectiveVel) ctx.effectiveVel = ctx.bufs->d_vel;
    prof::Range r("PostOp.Boundary", prof::Color(0x50,0x90,0x60));
    LaunchBoundary(ctx.bufs->d_pos_next, ctx.effectiveVel, p.grid, p.boundaryRestitution, p.numParticles, s);
}

void RecycleOp::run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
    if (!ctx.effectiveVel) ctx.effectiveVel = ctx.bufs->d_vel;
    prof::Range r("PostOp.Recycle", prof::Color(0x90,0x40,0xA0));
    LaunchRecycleToNozzleConst(ctx.bufs->d_pos_curr, ctx.bufs->d_pos_next, ctx.effectiveVel,
                               p.grid, p.dt, p.numParticles, 0, s);

    static uint64_t s_lastCopyFrame = UINT64_MAX;
    // 移除不存在的 posPredExternal：仅保留 pingpong 与 externalPingPong 判定
    bool needCopy = (p.numParticles > 0) && !ctx.pingPongPos && !ctx.bufs->externalPingPong;
    if (needCopy) {
        if (s_lastCopyFrame == g_simFrameIndex) return;
        s_lastCopyFrame = g_simFrameIndex;
        size_t bytes = sizeof(float4) * p.numParticles;
        std::fprintf(stderr,
            "[RecycleFallback][Frame=%llu] bytes=%zu (%.3f MB)\n",
            (unsigned long long)g_simFrameIndex,
            bytes, bytes / (1024.0 * 1024.0));
        cudaMemcpyAsync(ctx.bufs->d_pos_curr, ctx.bufs->d_pos_curr,
                        bytes, cudaMemcpyDeviceToDevice, s);
    }
}

void XsphOp::run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
    if(p.xsph_c<=0.f || p.numParticles==0) return;
    DeviceParams dp = MakeDP(p);
    prof::Range r("PostOp.XSPH", prof::Color(0x30,0x70,0xC0));
    bool useMP = (UseHalfForPosition(p, Stage::XSPH, *ctx.bufs) && UseHalfForVelocity(p, Stage::XSPH, *ctx.bufs));

    if(ctx.useHashedGrid){
        const uint32_t* keyMap = (p.compactBinarySearch ? nullptr : ctx.grid->d_keyToCompact);
        uint32_t numCells = ctx.grid->numCells;
        if(useMP)
            LaunchXSPHCompactMP(ctx.bufs->d_vel, ctx.bufs->d_vel, ctx.bufs->d_vel_h4,
                                ctx.bufs->d_pos_next, ctx.bufs->d_pos_next_h4,
                                ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                                ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                                keyMap, numCells, dp, p.numParticles, s);
        else
            LaunchXSPHCompact(ctx.bufs->d_vel, ctx.bufs->d_vel, ctx.bufs->d_pos_next,
                              ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                              ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                              dp, p.numParticles, s);
    } else {
        if(useMP)
            LaunchXSPHMP(ctx.bufs->d_vel, ctx.bufs->d_vel, ctx.bufs->d_vel_h4,
                         ctx.bufs->d_pos_next, ctx.bufs->d_pos_next_h4,
                         ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                         ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                         dp, p.numParticles, s);
        else
            LaunchXSPH(ctx.bufs->d_vel, ctx.bufs->d_vel, ctx.bufs->d_pos_next,
                       ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                       ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                       dp, p.numParticles, s);
    }
    ctx.effectiveVel = ctx.bufs->d_vel;
}

void PostOpsPipeline::configure(const PostOpsConfig& cfg, bool /*useHashedGrid*/, bool hasXsph) {
    m_ops.clear();
    if(cfg.enableXsph && hasXsph) m_ops.push_back(std::make_unique<XsphOp>());
    if(cfg.enableBoundary)        m_ops.push_back(std::make_unique<BoundaryOp>());
    if(cfg.enableRecycle)         m_ops.push_back(std::make_unique<RecycleOp>());
}

void PostOpsPipeline::runAll(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
    ctx.effectiveVel = ctx.bufs->d_vel;
    ctx.xsphApplied = false;
    prof::Range r("PostOps.RunAll", prof::Color(0x40,0x40,0xD0));
    for(auto& op : m_ops) op->run(ctx,p,s);
}

} // namespace sim
