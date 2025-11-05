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
        const std::vector<std::unique_ptr<IPhase>>& full() const { return m_full; }

        template<typename P, typename...Args>
        void addFull(Args&&...args) { m_full.push_back(std::make_unique<P>(std::forward<Args>(args)...)); }

        void runFull(SimulationContext& ctx, const SimParams& p, cudaStream_t s) {
            for (auto& ph : m_full) ph->run(ctx, p, s);
            m_post.runAll(ctx, p, s);
        }

        PostOpsPipeline& post() { return m_post; }
    private:
        std::vector<std::unique_ptr<IPhase>> m_full;
        PostOpsPipeline m_post;
    };

    void BuildDefaultPipelines(PhasePipeline& pipeline);

} // namespace sim
