#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "parameters.h"

namespace sim {
    class Simulator;

    // Graph 构建结果便于扩展（当前直接操作 Simulator 内部成员，结构体预留）
    struct GraphBuildResult {
        bool structuralRebuilt = false;
        bool dynamicUpdated = false;
        bool reuseSucceeded = false;
    };

    class GraphBuilder {
    public:
        // 结构重建（粒子数量 / 网格尺寸 / solverIters 等变化）
        GraphBuildResult BuildStructural(Simulator& sim, const SimParams& p);

        // 仅动态参数变化（dt / gravity / restDensity / kernel 等）
        GraphBuildResult UpdateDynamic(Simulator& sim, const SimParams& p,
            int minIntervalFrames,
            int frameIndex,
            int lastUpdateFrame);

    private:
        // 录制指定序列到新 graph
        bool recordSequencePipeline(Simulator& sim, const SimParams& p, bool full,
            cudaGraph_t& outGraph);

        // 释放旧资源
        void destroyGraphs(Simulator& sim);

        // 写入捕获参数签名
        void updateCapturedSignature(Simulator& sim, const SimParams& p, uint32_t numCells);

        // 判定结构变化（使用 Simulator 内已有对比逻辑）
        bool structuralChanged(const Simulator& sim, const SimParams& p) const;

        // 判定动态变化
        bool dynamicChanged(const Simulator& sim, const SimParams& p) const;
    };

} // namespace sim