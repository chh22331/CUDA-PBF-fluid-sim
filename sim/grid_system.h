#pragma once
#include <cuda_runtime.h>
#include "parameters.h"
#include "grid_buffers.cuh"

namespace sim {
// GridSystem: shared helpers for grid dimension & allocation (Stage A).
struct GridSystem {
    static inline int3 ComputeDims(const GridBounds& g) {
        const float cs = (g.cellSize > 0.f) ? g.cellSize : 1.f;
        int3 d;
        d.x = int(ceilf((g.maxs.x - g.mins.x) / cs));
        d.y = int(ceilf((g.maxs.y - g.mins.y) / cs));
        d.z = int(ceilf((g.maxs.z - g.mins.z) / cs));
        if (d.x < 0) d.x = 0; if (d.y < 0) d.y = 0; if (d.z < 0) d.z = 0;
        return d;
    }
    static inline uint32_t NumCells(int3 dim) {
        return uint32_t(dim.x) * uint32_t(dim.y) * uint32_t(dim.z);
    }
    static inline uint32_t NumCellsFromBounds(const GridBounds& g) { return NumCells(ComputeDims(g)); }
    static inline bool EnsureGridAllocated(GridBuffers& gb, GridBounds& bounds, uint32_t& numCellsCurrent) {
        int3 newDim = ComputeDims(bounds);
        uint32_t newNum = NumCells(newDim);
        bool changed = (newNum != numCellsCurrent) ||
            (newDim.x != bounds.dim.x || newDim.y != bounds.dim.y || newDim.z != bounds.dim.z);
        bounds.dim = newDim;
        if (changed) {
            gb.resizeGridRanges(newNum);
            numCellsCurrent = newNum;
        }
        return changed;
    }
};
} // namespace sim
