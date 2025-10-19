#pragma once
#include "grid_strategy.h"
#include "simulation_context.h"
#include "kernel_dispatcher.h"

namespace sim {
class DenseGridStrategy : public IGridStrategy {
public:
    void build(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override {
        // placeholder: will call hash, sort, cell ranges in future migrated code
        (void)ctx; (void)p; (void)s;
    }
    void solveIter(SimulationContext& ctx, const SimParams& p, cudaStream_t s, int iterIndex, KernelDispatcher& kd) override {
        (void)iterIndex; (void)kd; (void)ctx; (void)p; (void)s; }
    void xsph(SimulationContext& ctx, const SimParams& p, cudaStream_t s, KernelDispatcher& kd, bool useHalfPosVel) override {
        (void)useHalfPosVel; (void)kd; (void)ctx; (void)p; (void)s; }
};
} // namespace sim
