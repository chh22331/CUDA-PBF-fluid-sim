#include "post_ops.h"
#include "simulation_context.h"
#include "precision_stage.h"
#include "logging.h"
#include "stats.h"
#include <cstdio>

// extern kernels
extern "C" void LaunchBoundary(float4*, float4*, sim::GridBounds, float, uint32_t, cudaStream_t);
extern "C" void LaunchRecycleToNozzleConst(float4*, float4*, float4*, sim::GridBounds, float, uint32_t, int, cudaStream_t);
extern "C" void LaunchXSPH(float4*, const float4*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHCompact(float4*, const float4*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHMP(float4*, const float4*, const sim::Half4*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHCompactMP(float4*, const float4*, const sim::Half4*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);

namespace sim {

static inline const char* ClassifyRecycleFallback(const SimulationContext& ctx) {
    bool ext = ctx.bufs->posPredExternal;
    bool pp = ctx.pingPongPos;
    if (ext && !pp) return "Both(External+PingPongOff)";
    if (ext && pp)  return "ExternalPredOnly";
    if (!ext && !pp) return "PingPongOffOnly";
    return "Unexpected(NoCondition)"; // ��Ӧ��������ʱ����
}

static DeviceParams MakeDP(const SimParams& p){ return MakeDeviceParams(p); }

void BoundaryOp::run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
    if(!ctx.effectiveVel) ctx.effectiveVel = ctx.bufs->d_vel;
    LaunchBoundary(ctx.bufs->d_pos_pred, ctx.effectiveVel, p.grid, p.boundaryRestitution, p.numParticles, s);
}

void RecycleOp::run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
    if(!ctx.effectiveVel) ctx.effectiveVel = ctx.bufs->d_vel;
    LaunchRecycleToNozzleConst(ctx.bufs->d_pos, ctx.bufs->d_pos_pred, ctx.effectiveVel, p.grid, p.dt, p.numParticles, 0, s);
    if(p.numParticles>0){
        cudaMemcpyAsync(ctx.bufs->d_pos, ctx.bufs->d_pos_pred, sizeof(float4)*p.numParticles, cudaMemcpyDeviceToDevice, s);
        if(ctx.xsphApplied){
            cudaMemcpyAsync(ctx.bufs->d_vel, ctx.bufs->d_delta, sizeof(float4)*p.numParticles, cudaMemcpyDeviceToDevice, s);
        }
    }
}

void XsphOp::run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
    if(p.xsph_c<=0.f || p.numParticles==0) return;
    DeviceParams dp = MakeDP(p);
    bool useMP = (UseHalfForPosition(p, Stage::XSPH, *ctx.bufs) && UseHalfForVelocity(p, Stage::XSPH, *ctx.bufs));
    if(ctx.useHashedGrid){
        if(useMP) LaunchXSPHCompactMP(ctx.bufs->d_delta, ctx.bufs->d_vel, ctx.bufs->d_vel_h4, ctx.bufs->d_pos_pred, ctx.bufs->d_pos_pred_h4, ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted, ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount, dp, p.numParticles, s);
        else      LaunchXSPHCompact(ctx.bufs->d_delta, ctx.bufs->d_vel, ctx.bufs->d_pos_pred, ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted, ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount, dp, p.numParticles, s);
    } else {
        if(useMP) LaunchXSPHMP(ctx.bufs->d_delta, ctx.bufs->d_vel, ctx.bufs->d_vel_h4, ctx.bufs->d_pos_pred, ctx.bufs->d_pos_pred_h4, ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted, ctx.grid->d_cellStart, ctx.grid->d_cellEnd, dp, p.numParticles, s);
        else      LaunchXSPH(ctx.bufs->d_delta, ctx.bufs->d_vel, ctx.bufs->d_pos_pred, ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted, ctx.grid->d_cellStart, ctx.grid->d_cellEnd, dp, p.numParticles, s);
    }
    ctx.effectiveVel = ctx.bufs->d_delta;
    ctx.xsphApplied = true;
}

void PostOpsPipeline::configure(const PostOpsConfig& cfg, bool /*useHashedGrid*/, bool hasXsph) {
    m_ops.clear();
    if(cfg.enableXsph && hasXsph){ m_ops.push_back(std::make_unique<XsphOp>()); }
    if(cfg.enableBoundary){ m_ops.push_back(std::make_unique<BoundaryOp>()); }
    if(cfg.enableRecycle){ m_ops.push_back(std::make_unique<RecycleOp>()); }
}

void PostOpsPipeline::runAll(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
    ctx.effectiveVel = ctx.bufs->d_vel;
    ctx.xsphApplied = false;
    for(auto& op : m_ops) op->run(ctx,p,s);
}

} // namespace sim
