#pragma once
#include <vector>
#include <memory>
#include "parameters.h"
#include "device_buffers.cuh"
#include "grid_buffers.cuh"
#include "simulation_context.h"
#include "post_ops.h"

namespace sim {

enum class PhaseType { Integrate, GridBuild, SortPairs, CellRanges, SolveIterations, Velocity, Post }; 

class IPhase {
public:
    virtual ~IPhase() = default;
    virtual PhaseType type() const = 0;
    virtual void run(SimulationContext& ctx, const SimParams& params, cudaStream_t stream) = 0;
};

class PhasePipeline {
public:
    // 统一：仅保留单一阶段序列（原 full + cheap 合并）
    const std::vector<std::unique_ptr<IPhase>>& phases() const { return m_phases; }

    template<typename P, typename...Args>
    void addPhase(Args&&...args){ m_phases.push_back(std::make_unique<P>(std::forward<Args>(args)...)); }

    // 原 runFull/runCheap 合并
    void runAll(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
        for(auto& ph: m_phases) ph->run(ctx,p,s);
        m_post.runAll(ctx,p,s);
    }

    PostOpsPipeline& post(){ return m_post; }
private:
    std::vector<std::unique_ptr<IPhase>> m_phases; // [CHEAP_REMOVED] 统一序列
    PostOpsPipeline m_post;
};

// Populate default phases
void BuildDefaultPipelines(PhasePipeline& pipeline);

} // namespace sim
