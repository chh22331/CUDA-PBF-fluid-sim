#pragma once
#include "device_buffers.cuh"
#include "grid_buffers.cuh"
#include "parameters.h"

namespace sim {
    class IGridStrategy;
    class KernelDispatcher;
    struct SimulationContext {
        DeviceBuffers* bufs = nullptr;
        GridBuffers* grid = nullptr;
        bool               useHashedGrid = false;
        IGridStrategy* gridStrategy = nullptr;
        KernelDispatcher* dispatcher = nullptr;
        // Post-ops mutable state
        float4* effectiveVel = nullptr;
        bool               xsphApplied = false;
        bool               pingPongPos = true; 
        // Graph / pipeline capture helpers
        bool               sortTempPreEnsured = false; 
    };
    extern uint64_t g_simFrameIndex;
} // namespace sim
