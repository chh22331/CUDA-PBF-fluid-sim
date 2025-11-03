#include "post_ops.h"
#include "simulation_context.h"
#include "precision_stage.h"
#include "logging.h"
#include "stats.h"
#include "../engine/core/console.h"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include "../engine/core/prof_nvtx.h"

// 修正：Boundary / BoundaryHalf 正确函数签名（含幽灵粒子参数）
// 旧签名缺少 ghostCount / ghostClamp 导致调用栈参数错位，引发 nvcuda64.dll 访问冲突
extern "C" void LaunchBoundary(
    float4* pos_out,
    float4* vel_io,
    sim::GridBounds grid,
    float restitution,
    uint32_t N,
    uint32_t ghostCount,
    uint8_t ghostClamp,
    cudaStream_t s);

extern "C" void LaunchBoundaryHalf(
    float4* pos_out,
    float4* vel_io,
    const sim::Half4* pos_pred_h4,
    const sim::Half4* vel_h4,
    sim::GridBounds grid,
    float restitution,
    uint32_t N,
    uint32_t ghostCount,
    bool forceFp32Accumulate,
    uint8_t ghostClamp,
    cudaStream_t s);

extern "C" void LaunchRecycleToNozzleConst(float4*, float4*, float4*, sim::GridBounds, float, uint32_t, int, cudaStream_t);
extern "C" void LaunchXSPH(float4*, const float4*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHCompact(float4*, const float4*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHMP(float4*, const float4*, const sim::Half4*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHCompactMP(float4*, const float4*, const sim::Half4*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHHalfDense(float4*, const sim::Half4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, bool, cudaStream_t);
extern "C" void LaunchXSPHHalfCompact(float4*, const sim::Half4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, bool, cudaStream_t);
extern "C" void LaunchXSphDiag(const float4* velPrev, const float4* velNew, uint32_t N, double* sumPrev, double* sumDv, int* hasNaN, cudaStream_t s);

namespace sim {

    static inline const char* ClassifyRecycleFallback(const SimulationContext& ctx) {
        bool ext = ctx.bufs->posPredExternal; bool pp = ctx.pingPongPos;
        if (ext && !pp) return "Both(External+PingPongOff)";
        if (ext && pp)  return "ExternalPredOnly";
        if (!ext && !pp) return "PingPongOffOnly";
        return "Unexpected(NoCondition)";
    }
    static DeviceParams MakeDP(const SimParams& p) { return MakeDeviceParams(p); }

    // 修复 BoundaryOp::run
    void BoundaryOp::run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
        prof::Range r("PostOp.Boundary", prof::Color(0x50, 0x90, 0x60));
        const auto& cc = console::Instance();
        uint32_t ghostCount = p.ghostParticleCount;
        uint8_t ghostClamp = cc.sim.boundaryGhost.enable
            ? (cc.sim.boundaryGhost.place_outside ? 0u : 1u)
            : 0u;

        bool useHalfPos = UseHalfForPosition(p, Stage::Boundary, *ctx.bufs);
        bool useHalfVel = UseHalfForVelocity(p, Stage::Boundary, *ctx.bufs);
        bool haveHalf = (ctx.bufs->d_pos_pred_h4 && ctx.bufs->d_vel_h4);
        bool nativeHalf = ctx.bufs->nativeHalfActive;
        bool canHalf = nativeHalf || (useHalfPos && useHalfVel && haveHalf);

        if (cc.debug.printDiagnostics) {
            NumericType ct = StageComputeType(p.precision, Stage::Boundary);
            std::fprintf(stderr,
                "[PredHalf.Stage] stage=Boundary halfPos=%d halfVel=%d haveHalf=%d native=%d canHalf=%d "
                "pred_h4=%p vel_h4=%p pos_pred=%p N=%u ghost=%u computeT=%u reason=%s\n",
                (int)useHalfPos, (int)useHalfVel, (int)haveHalf, (int)nativeHalf, (int)canHalf,
                (void*)ctx.bufs->d_pos_pred_h4, (void*)ctx.bufs->d_vel_h4, (void*)ctx.bufs->d_pos_pred,
                p.numParticles, ghostCount, (unsigned)ct,
                (!useHalfPos || !useHalfVel) ? "StageCompute!=FP16? or mirrors missing" : "OK");
        }

        if (canHalf) {
            if (!nativeHalf) ctx.bufs->packAllToHalf(p.numParticles, s);
            LaunchBoundaryHalf(
                ctx.bufs->d_pos_pred,
                ctx.bufs->d_vel,
                ctx.bufs->d_pos_pred_h4,
                ctx.bufs->d_vel_h4,
                p.grid,
                p.boundaryRestitution,
                p.numParticles,
                ghostCount,
                p.precision.forceFp32Accumulate,
                ghostClamp,
                s);
        }
        else {
            LaunchBoundary(
                ctx.bufs->d_pos_pred,
                ctx.bufs->d_vel,
                p.grid,
                p.boundaryRestitution,
                p.numParticles,
                ghostCount,
                ghostClamp,
                s);
        }
    }

    void RecycleOp::run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
        prof::Range r("PostOp.Recycle", prof::Color(0x90, 0x40, 0xA0));
        LaunchRecycleToNozzleConst(ctx.bufs->d_pos, ctx.bufs->d_pos_pred, ctx.bufs->d_vel, p.grid, p.dt, p.numParticles, 0, s);
        static uint64_t s_lastCopyFrame = UINT64_MAX;
        bool needCopy = (p.numParticles > 0) && !ctx.pingPongPos && !ctx.bufs->posPredExternal && !ctx.bufs->externalPingPong;
        if (needCopy) {
            if (s_lastCopyFrame == g_simFrameIndex) return;
            s_lastCopyFrame = g_simFrameIndex;
            size_t bytes = sizeof(float4) * p.numParticles;
            const char* reason = ClassifyRecycleFallback(ctx);
            std::fprintf(stderr, "[RecycleFallback][Frame=%llu] reason=%s bytes=%.3fMB\n",
                (unsigned long long)g_simFrameIndex, reason, bytes / 1048576.0);
            prof::Range rCopy("D2D.pos_pred->pos", prof::Color(0xE0, 0x30, 0x30));
            cudaMemcpyAsync(ctx.bufs->d_pos, ctx.bufs->d_pos_pred, bytes, cudaMemcpyDeviceToDevice, s);
        }
    }

    void XsphOp::run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
        if (p.xsph_c <= 0.f || p.numParticles == 0) return;
        DeviceParams dp = MakeDP(p);
        prof::Range r("PostOp.XSPH", prof::Color(0x30, 0x70, 0xC0));
        bool halfLoadPos = UseHalfForPosition(p, Stage::XSPH, *ctx.bufs);
        bool halfLoadVel = UseHalfForVelocity(p, Stage::XSPH, *ctx.bufs);
        bool haveHalfBufs = (ctx.bufs->d_vel_h4 && ctx.bufs->d_pos_pred_h4);
        bool preferHalfArithmetic = (halfLoadPos && halfLoadVel && haveHalfBufs && p.precision.enableHalfIntrinsics);
        const auto& cc = console::Instance();
        if (cc.debug.printDiagnostics) {
            std::fprintf(stderr, "[PredHalf.Stage] stage=XSPH halfPos=%d halfVel=%d haveHalf=%d preferHalfArith=%d pos_pred_h4=%p vel_h4=%p pos_pred=%p N=%u\n",
                (int)halfLoadPos, (int)halfLoadVel, (int)haveHalfBufs, (int)preferHalfArithmetic, (void*)ctx.bufs->d_pos_pred_h4, (void*)ctx.bufs->d_vel_h4, (void*)ctx.bufs->d_pos_pred, p.numParticles);
        }

        // snapshot previous velocity
        cudaMemcpyAsync(ctx.bufs->d_vel_prev, ctx.bufs->d_vel, sizeof(float4) * p.numParticles, cudaMemcpyDeviceToDevice, s);

        // pack mirrors if needed
        if (preferHalfArithmetic) ctx.bufs->packAllToHalf(p.numParticles, s);
        else if ((halfLoadPos || halfLoadVel) && haveHalfBufs) ctx.bufs->packAllToHalf(p.numParticles, s);

        static bool loggedHalf = false, loggedMP = false, loggedFP32 = false;
        auto logSel = [&](const char* tag) {
            if (cc.debug.printDiagnostics)
                std::fprintf(stderr, "[XSPH][Select] %s N=%u compact=%d forceFp32Acc=%d c=%.3f\n",
                    tag, p.numParticles, (int)ctx.useHashedGrid, (int)p.precision.forceFp32Accumulate, p.xsph_c);
            };
        if (preferHalfArithmetic) { if (!loggedHalf) { logSel("HalfArithmetic"); loggedHalf = true; } }
        else if ((halfLoadPos || halfLoadVel) && haveHalfBufs) { if (!loggedMP) { logSel("MixedHalfLoad"); loggedMP = true; } }
        else { if (!loggedFP32) { logSel("FP32"); loggedFP32 = true; } }

        if (!haveHalfBufs && (halfLoadPos || halfLoadVel) && cc.debug.printWarnings) {
            std::fprintf(stderr, "[XSPH][Fallback] half store requested but buffers missing (vel_h4=%p pos_pred_h4=%p) -> FP32\n",
                (void*)ctx.bufs->d_vel_h4, (void*)ctx.bufs->d_pos_pred_h4);
        }

        if (ctx.useHashedGrid) {
            if (preferHalfArithmetic) {
                LaunchXSPHHalfCompact(ctx.bufs->d_vel, ctx.bufs->d_vel_h4, ctx.bufs->d_pos_pred_h4,
                    ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                    ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                    dp, p.numParticles, p.precision.forceFp32Accumulate, s);
            }
            else if ((halfLoadPos || halfLoadVel) && haveHalfBufs) {
                LaunchXSPHCompactMP(ctx.bufs->d_vel, ctx.bufs->d_vel, ctx.bufs->d_vel_h4,
                    ctx.bufs->d_pos_pred, ctx.bufs->d_pos_pred_h4,
                    ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                    ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                    dp, p.numParticles, s);
            }
            else {
                LaunchXSPHCompact(ctx.bufs->d_vel, ctx.bufs->d_vel, ctx.bufs->d_pos_pred,
                    ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                    ctx.grid->d_cellUniqueKeys, ctx.grid->d_cellOffsets, ctx.grid->d_compactCount,
                    dp, p.numParticles, s);
            }
        }
        else {
            if (preferHalfArithmetic) {
                LaunchXSPHHalfDense(ctx.bufs->d_vel, ctx.bufs->d_vel_h4, ctx.bufs->d_pos_pred_h4,
                    ctx.grid->d_indices_sorted, ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                    dp, p.numParticles, p.precision.forceFp32Accumulate, s);
            }
            else if ((halfLoadPos || halfLoadVel) && haveHalfBufs) {
                LaunchXSPHMP(ctx.bufs->d_vel, ctx.bufs->d_vel, ctx.bufs->d_vel_h4,
                    ctx.bufs->d_pos_pred, ctx.bufs->d_pos_pred_h4,
                    ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                    ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                    dp, p.numParticles, s);
            }
            else {
                LaunchXSPH(ctx.bufs->d_vel, ctx.bufs->d_vel, ctx.bufs->d_pos_pred,
                    ctx.grid->d_indices_sorted, ctx.grid->d_cellKeys_sorted,
                    ctx.grid->d_cellStart, ctx.grid->d_cellEnd,
                    dp, p.numParticles, s);
            }
        }

        // Diagnostics rollback logic unchanged
        if (cc.debug.logXSphEffect) {
            static double* d_sumPrev = nullptr, * d_sumDv = nullptr;
            static int* d_hasNaN = nullptr;
            if (!d_sumPrev) {
                cudaMalloc(&d_sumPrev, sizeof(double));
                cudaMalloc(&d_sumDv, sizeof(double));
                cudaMalloc(&d_hasNaN, sizeof(int));
            }
            double zeroD = 0.0; int zeroI = 0;
            cudaMemcpyAsync(d_sumPrev, &zeroD, sizeof(double), cudaMemcpyHostToDevice, s);
            cudaMemcpyAsync(d_sumDv, &zeroD, sizeof(double), cudaMemcpyHostToDevice, s);
            cudaMemcpyAsync(d_hasNaN, &zeroI, sizeof(int), cudaMemcpyHostToDevice, s);
            LaunchXSphDiag(ctx.bufs->d_vel_prev, ctx.bufs->d_vel, p.numParticles, d_sumPrev, d_sumDv, d_hasNaN, s);
            cudaStreamSynchronize(s);
            double hPrev = 0.0, hDv = 0.0; int hNaN = 0;
            cudaMemcpy(&hPrev, d_sumPrev, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&hDv, d_sumDv, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&hNaN, d_hasNaN, sizeof(int), cudaMemcpyDeviceToHost);
            double ratio = (hPrev > 0.0) ? (hDv / hPrev) : 0.0;
            bool rollback = false;
            if (hNaN) {
                rollback = true;
                if (cc.debug.printWarnings) std::fprintf(stderr, "[XSPH][Rollback] NaN ratio=%.4g\n", ratio);
            }
            else if (ratio > 0.6) {
                rollback = true;
                if (cc.debug.printWarnings) std::fprintf(stderr, "[XSPH][Rollback] ratio=%.4g too large\n", ratio);
            }
            if (rollback) {
                cudaMemcpyAsync(ctx.bufs->d_vel, ctx.bufs->d_vel_prev, sizeof(float4) * p.numParticles, cudaMemcpyDeviceToDevice, s);
                if (cc.debug.printWarnings) std::fprintf(stderr, "[XSPH][Rollback] restored prev velocities\n");
            }
            else if (cc.debug.printDiagnostics) {
                std::fprintf(stderr, "[XSPH][Diag] ratio=%.4g prevL1=%.4g dvL1=%.4g NaN=%d\n", ratio, hPrev, hDv, hNaN);
            }
        }
    }

    void PostOpsPipeline::configure(const PostOpsConfig& cfg, bool /*useHashedGrid*/, bool hasXsph) {
        m_ops.clear();
        if (cfg.enableXsph && hasXsph) m_ops.push_back(std::make_unique<XsphOp>());
        if (cfg.enableBoundary) m_ops.push_back(std::make_unique<BoundaryOp>());
        if (cfg.enableRecycle)  m_ops.push_back(std::make_unique<RecycleOp>());
    }
    void PostOpsPipeline::runAll(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
        prof::Range r("PostOps.RunAll", prof::Color(0x40,0x40,0xD0));
        for (auto& op : m_ops) op->run(ctx, p, s);
    }

} // namespace sim