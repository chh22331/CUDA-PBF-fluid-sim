#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "simulation_context.h"
#include "parameters.h"

namespace sim {
class IPostOp {
public:
    virtual ~IPostOp() = default;
    virtual const char* name() const = 0;
    virtual void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) = 0;
};

class BoundaryOp : public IPostOp {
public:
    const char* name() const override { return "Boundary"; }
    void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override;
};

class RecycleOp : public IPostOp {
public:
    const char* name() const override { return "Recycle"; }
    void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override;
};

class XsphOp : public IPostOp {
public:
    const char* name() const override { return "XSPH"; }
    void run(SimulationContext& ctx, const SimParams& p, cudaStream_t s) override;
};

struct PostOpsConfig {
    bool enableXsph   = true;
    bool enableBoundary = true;
    bool enableRecycle  = true;
};

class PostOpsPipeline {
public:
    void configure(const PostOpsConfig& cfg, bool useHashedGrid, bool hasXsph);
    void runAll(SimulationContext& ctx, const SimParams& p, cudaStream_t s);
private:
    std::vector<std::unique_ptr<IPostOp>> m_ops;
};
}
