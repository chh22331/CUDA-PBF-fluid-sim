#include "graph_builder.h"
#include "simulator.h"
#include "grid_system.h"
#include "phase_pipeline.h"
#include "post_ops.h"
#include "simulation_context.h"
#include "grid_strategy.h"
#include "grid_buffers.cuh"
#include "device_buffers.cuh"
#include "kernel_dispatcher.h"
#include "../engine/core/console.h"
#include "logging.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) do { cudaError_t _e=(expr); if(_e!=cudaSuccess){ sim::Log(sim::LogChannel::Error,"CUDA %s (%d)", cudaGetErrorString(_e), (int)_e);} } while(0)
#endif

extern "C" bool EnsureCellCompactScratch(uint32_t, uint32_t);
extern "C" void LaunchSortPairsQuery(size_t*, const uint32_t*, uint32_t*, const uint32_t*, uint32_t*, uint32_t, cudaStream_t);

namespace sim {

bool GraphBuilder::recordSequencePipeline(Simulator& sim, const SimParams& p, cudaGraph_t& outGraph) {
    outGraph = nullptr;

    if (p.numParticles > 0) {
        size_t tempBytes = 0;
        LaunchSortPairsQuery(&tempBytes,
                             sim.m_grid.d_cellKeys,
                             sim.m_grid.d_cellKeys_sorted,
                             sim.m_grid.d_indices,
                             sim.m_grid.d_indices_sorted,
                             p.numParticles,
                             sim.m_stream);
        sim.ensureSortTemp(tempBytes);
    }
    if (sim.m_useHashedGrid) {
        if (!EnsureCellCompactScratch(p.numParticles, 256)) {
            if (console::Instance().debug.printErrors)
                std::fprintf(stderr, "[GraphBuilder][Error] EnsureCellCompactScratch failed.\n");
            return false;
        }
    }

    if (sim.m_pipeline.full().empty()) {
        BuildDefaultPipelines(sim.m_pipeline);
        PostOpsConfig cfg{};
        cfg.enableXsph     = (p.xsph_c > 0.f);
        cfg.enableBoundary = true;
        cfg.enableRecycle  = true;
        sim.m_pipeline.post().configure(cfg, sim.m_useHashedGrid, cfg.enableXsph);
    }

    sim.m_ctx.bufs         = &sim.m_bufs;
    sim.m_ctx.grid         = &sim.m_grid;
    sim.m_ctx.useHashedGrid= sim.m_useHashedGrid;
    sim.m_ctx.gridStrategy = sim.m_gridStrategy.get();
    sim.m_ctx.dispatcher   = &sim.m_kernelDispatcher;

    CUDA_CHECK(cudaStreamBeginCapture(sim.m_stream, cudaStreamCaptureModeGlobal));
    sim.m_pipeline.runFull(sim.m_ctx, p, sim.m_stream);
    CUDA_CHECK(cudaStreamEndCapture(sim.m_stream, &outGraph));

    BindDeviceGlobalsFrom(sim.m_bufs);
    return outGraph != nullptr;
}

void GraphBuilder::destroyGraph(Simulator& sim) {
    if (sim.m_graphExec) { cudaGraphExecDestroy(sim.m_graphExec); sim.m_graphExec = nullptr; }
    if (sim.m_graph)     { cudaGraphDestroy(sim.m_graph);         sim.m_graph = nullptr; }
    sim.m_cachedNodesReady = false;
}

void GraphBuilder::updateCapturedSignature(Simulator& sim, const SimParams& p, uint32_t numCells) {
    sim.m_captured.numParticles  = p.numParticles;
    sim.m_captured.numCells      = numCells;
    sim.m_captured.solverIters   = p.solverIters;
    sim.m_captured.maxNeighbors  = p.maxNeighbors;
    sim.m_captured.sortEveryN    = p.sortEveryN;
    sim.m_captured.grid          = p.grid;
    sim.m_captured.kernel        = p.kernel;
    sim.m_captured.dt            = p.dt;
    sim.m_captured.gravity       = p.gravity;
    sim.m_captured.restDensity   = p.restDensity;
    sim.m_paramTracker.capture(p, numCells);
}

bool GraphBuilder::structuralChanged(const Simulator& sim, const SimParams& p) const {
    return sim.structuralGraphChange(p);
}

bool GraphBuilder::dynamicChanged(const Simulator& sim, const SimParams& p) const {
    if (structuralChanged(sim, p)) return false;
    return sim.paramOnlyGraphChange(p);
}

GraphBuildResult GraphBuilder::BuildStructural(Simulator& sim, const SimParams& p) {
    GraphBuildResult res{};
    if (!sim.m_bufs.d_pos || !sim.m_bufs.d_vel || !sim.m_bufs.d_pos_pred) {
        std::fprintf(stderr, "[GraphBuilder][Error] Required buffers null before structural capture.\n");
    }

    destroyGraph(sim);

    cudaGraph_t newGraph = nullptr;
    if (!recordSequencePipeline(sim, p, newGraph)) {
        std::fprintf(stderr, "[GraphBuilder][Error] recordSequence failed.\n");
        return res;
    }
    CUDA_CHECK(cudaGraphInstantiate(&sim.m_graphExec, newGraph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphUpload(sim.m_graphExec, sim.m_stream));

    sim.m_graph = newGraph;
    sim.cacheGraphNodes();

    int3 dim       = GridSystem::ComputeDims(p.grid);
    uint32_t cells = GridSystem::NumCells(dim);
    updateCapturedSignature(sim, p, cells);

    sim.m_graphDirty = false;
    sim.m_paramDirty = false;

    res.structuralRebuilt = true;
    res.reuseSucceeded    = true;
    return res;
}

GraphBuildResult GraphBuilder::UpdateDynamic(Simulator& sim,
                                             const SimParams& p,
                                             int minIntervalFrames,
                                             int frameIndex,
                                             int lastUpdateFrame) {
    GraphBuildResult res{};
    if (!sim.m_paramDirty) return res;

    if (minIntervalFrames > 0 && lastUpdateFrame >= 0 &&
        (frameIndex - lastUpdateFrame) < minIntervalFrames) {
        sim.m_paramDirty = false;
        return res;
    }

    if (structuralChanged(sim, p)) {
        sim.m_graphDirty = true;
        return res;
    }

    cudaGraph_t newGraph = nullptr;
    if (!recordSequencePipeline(sim, p, newGraph)) {
        sim.m_graphDirty = true;
        return res;
    }

    cudaGraphExecUpdateResultInfo info{};
    cudaError_t e = cudaGraphExecUpdate(sim.m_graphExec, newGraph, &info);
    if (newGraph) cudaGraphDestroy(newGraph);

    if (!(e == cudaSuccess && info.result == cudaGraphExecUpdateSuccess)) {
        sim.m_graphDirty = true;
        return res;
    }

    sim.cacheGraphNodes();
    sim.m_captured.dt          = p.dt;
    sim.m_captured.gravity     = p.gravity;
    sim.m_captured.restDensity = p.restDensity;
    sim.m_captured.kernel      = p.kernel;
    sim.m_paramTracker.capture(p, sim.m_captured.numCells);

    sim.m_paramDirty   = false;
    res.dynamicUpdated = true;
    res.reuseSucceeded = true;
    return res;
}

} // namespace sim