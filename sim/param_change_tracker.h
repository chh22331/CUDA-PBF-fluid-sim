#pragma once
#include "parameters.h"
#include "numeric_utils.h"
#include "grid_system.h"

namespace sim {

struct StructuralSignature {
    uint32_t numParticles = 0;
    int      solverIters = 0;
    int      maxNeighbors = 0;
    uint32_t numCells = 0;
    GridBounds grid{}; // includes mins/maxs/cellSize/dim
};

struct DynamicSignature {
    float    dt = 0.0f;
    float3   gravity = make_float3(0.f,0.f,0.f);
    float    restDensity = 0.0f;
    KernelCoeffs kernel{};
};

class ParamChangeTracker {
public:
    void capture(const SimParams& p, uint32_t numCells) {
        m_struct.numParticles = p.numParticles;
        m_struct.solverIters  = p.solverIters;
        m_struct.maxNeighbors = p.maxNeighbors;
        m_struct.numCells     = numCells;
        m_struct.grid         = p.grid;
        m_dynamic.dt          = p.dt;
        m_dynamic.gravity     = p.gravity;
        m_dynamic.restDensity = p.restDensity;
        m_dynamic.kernel      = p.kernel;
    }

    bool structuralChanged(const SimParams& p, uint32_t numCells) const {
        if(p.solverIters != m_struct.solverIters) return true;
        if(p.maxNeighbors!= m_struct.maxNeighbors) return true;
        if(p.numParticles != m_struct.numParticles) return true;
        if(numCells       != m_struct.numCells) return true;
        if(!gridEqual(p.grid, m_struct.grid)) return true;
        return false;
    }

    bool dynamicChanged(const SimParams& p) const {
        // dt relative tolerance: replicate existing heuristic
        float dtRel = fabsf(p.dt - m_dynamic.dt) / fmaxf(1e-9f, fmaxf(p.dt, m_dynamic.dt));
        if(dtRel < 0.002f && approxEq3(p.gravity, m_dynamic.gravity) && approxEq(p.restDensity, m_dynamic.restDensity) && kernelEqualRelaxed(p.kernel, m_dynamic.kernel))
            return false; // all within relaxed region
        if(!approxEq(p.dt, m_dynamic.dt)) return true;
        if(!approxEq3(p.gravity, m_dynamic.gravity)) return true;
        if(!approxEq(p.restDensity, m_dynamic.restDensity)) return true;
        if(!kernelEqualRelaxed(p.kernel, m_dynamic.kernel)) return true;
        return false;
    }

private:
    StructuralSignature m_struct{};
    DynamicSignature    m_dynamic{};
};

} // namespace sim
