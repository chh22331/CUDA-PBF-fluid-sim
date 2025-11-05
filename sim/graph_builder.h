#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "parameters.h"

namespace sim {
    class Simulator;

    struct GraphBuildResult {
        bool structuralRebuilt = false;
        bool dynamicUpdated = false;
        bool reuseSucceeded = false;
    };

    class GraphBuilder {
    public:
        GraphBuildResult BuildStructural(Simulator& sim, const SimParams& p);
        GraphBuildResult UpdateDynamic(Simulator& sim, const SimParams& p,
            int /*minIntervalFrames*/,
            int frameIndex,
            int lastUpdateFrame);
    private:
        bool recordSequencePipeline(Simulator& sim, const SimParams& p, cudaGraph_t& outGraph);
        void destroyGraphs(Simulator& sim);
        void updateCapturedSignature(Simulator& sim, const SimParams& p, uint32_t numCells);
        bool structuralChanged(const Simulator& sim, const SimParams& p) const;
        bool dynamicChanged(const Simulator& sim, const SimParams& p) const;
    };

} // namespace sim