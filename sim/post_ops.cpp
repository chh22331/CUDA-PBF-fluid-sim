#include "post_ops.h"
#include "simulation_context.h"
#include "precision_stage.h"
#include "logging.h"
#include "stats.h"
#include <cstdio>
#include "../engine/core/prof_nvtx.h"
#include "device_globals.cuh"
#include "parameters.h"

// 新接口 extern
extern "C" void LaunchBoundaryGlobals(sim::GridBounds, float restitution, uint32_t N, bool xsphApplied, cudaStream_t);
extern "C" void LaunchXsphDenseGlobals(const uint32_t* indicesSorted,
                                       const uint32_t* keysSorted,
                                       const uint32_t* cellStart,
                                       const uint32_t* cellEnd,
                                       sim::DeviceParams dp,
                                       uint32_t N,
                                       cudaStream_t s);
extern "C" void LaunchXsphCompactGlobals(const uint32_t* indicesSorted,
                                         const uint32_t* keysSorted,
                                         const uint32_t* uniqueKeys,
                                         const uint32_t* offsets,
                                         const uint32_t* compactCount,
                                         sim::DeviceParams dp,
                                         uint32_t N,
                                         cudaStream_t s);
extern "C" void LaunchRecycleToNozzleConst(float4*, float4*, float4*, sim::GridBounds, float, uint32_t, int, cudaStream_t);

namespace sim {

static DeviceParams MakeDP(const SimParams& p) { return MakeDeviceParams(p); }

void BoundaryOp::run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
    prof::Range r("PostOp.Boundary", prof::Color(0x50,0x90,0x60));
    // 根据是否已执行 XSPH 选择使用 delta(平滑后速度) 或 vel
    LaunchBoundaryGlobals(p.grid, p.boundaryRestitution, p.numParticles, ctx.xsphApplied, s);
    ctx.effectiveVel = ctx.xsphApplied ? ctx.bufs->d_delta : ctx.bufs->d_vel;
}

void RecycleOp::run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
    prof::Range r("PostOp.Recycle", prof::Color(0x90,0x40,0xA0));
    float4* effectiveVel = ctx.xsphApplied ? ctx.bufs->d_delta : ctx.bufs->d_vel;
    LaunchRecycleToNozzleConst(ctx.bufs->d_pos_curr,
                               ctx.bufs->d_pos_next,
                               effectiveVel,
                               p.grid, p.dt, p.numParticles, 0, s);

    static uint64_t s_lastCopyFrame = UINT64_MAX;
    bool needCopy = (p.numParticles > 0) && !ctx.pingPongPos && !ctx.bufs->externalPingPong;
    if (needCopy) {
        if (s_lastCopyFrame == g_simFrameIndex) return;
        s_lastCopyFrame = g_simFrameIndex;
        size_t bytes = sizeof(float4) * p.numParticles;
        std::fprintf(stderr, "[RecycleFallback][Frame=%llu] bytes=%zu (%.3f MB)\n",
                     (unsigned long long)g_simFrameIndex, bytes, bytes / (1024.0 * 1024.0));
        cudaMemcpyAsync(ctx.bufs->d_pos_curr,
                        ctx.bufs->d_pos_curr,
                        bytes,
                        cudaMemcpyDeviceToDevice,
                        s);
    }
    ctx.effectiveVel = effectiveVel;
}

void XsphOp::run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
    if (p.xsph_c <= 0.f || p.numParticles == 0) return;
    prof::Range r("PostOp.XSPH", prof::Color(0x30,0x70,0xC0));
    DeviceParams dp = MakeDP(p);

    if (ctx.useHashedGrid) {
        LaunchXsphCompactGlobals(ctx.grid->d_indices_sorted,
                                 ctx.grid->d_cellKeys_sorted,
                                 ctx.grid->d_cellUniqueKeys,
                                 ctx.grid->d_cellOffsets,
                                 ctx.grid->d_compactCount,
                                 dp, p.numParticles, s);
    } else {
        LaunchXsphDenseGlobals(ctx.grid->d_indices_sorted,
                               ctx.grid->d_cellKeys_sorted,
                               ctx.grid->d_cellStart,
                               ctx.grid->d_cellEnd,
                               dp, p.numParticles, s);
    }
    ctx.xsphApplied = true;
    ctx.effectiveVel = ctx.bufs->d_delta;
}

void PostOpsPipeline::configure(const PostOpsConfig& cfg, bool useHashedGrid, bool hasXsph) {
    m_ops.clear();
    if (cfg.enableXsph && hasXsph) m_ops.push_back(std::make_unique<XsphOp>());
    // 去除第二次边界反弹
    // if (cfg.enableBoundary) m_ops.push_back(std::make_unique<BoundaryOp>());
    if (cfg.enableRecycle)  m_ops.push_back(std::make_unique<RecycleOp>());
}

void PostOpsPipeline::runAll(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
    ctx.effectiveVel = ctx.bufs->d_vel;
    ctx.xsphApplied = false;
    prof::Range r("PostOps.RunAll", prof::Color(0x40,0x40,0xD0));
    for (auto& op : m_ops) op->run(ctx, p, s);
}

} // namespace sim
