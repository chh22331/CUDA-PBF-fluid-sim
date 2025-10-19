#pragma once
#include "simulation_context.h"
#include "kernel_dispatcher.h"
#include "precision_stage.h"

namespace sim {

class IGridStrategy {
public:
    virtual ~IGridStrategy() = default;
    virtual void build(SimulationContext&, const SimParams&, cudaStream_t) = 0; // hash + sort + ranges
    virtual void solveIter(SimulationContext&, const SimParams&, cudaStream_t, int iterIndex, KernelDispatcher&) = 0;
    virtual void xsph(SimulationContext&, const SimParams&, cudaStream_t, KernelDispatcher&, bool useHalfPosVel) = 0;
};

} // namespace sim
