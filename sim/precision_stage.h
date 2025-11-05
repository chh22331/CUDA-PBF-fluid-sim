#pragma once
#include "parameters.h"

namespace sim {

    enum class Stage {
        Emission,
        GridBuild,
        NeighborGather,
        Density,
        LambdaSolve,
        Integration,
        VelocityUpdate,
        Boundary,
        XSPH
    };

    inline NumericType StageComputeType(const SimPrecision& pr, Stage s) {
        switch (s) {
        case Stage::Emission:      return pr.emissionCompute;
        case Stage::GridBuild:     return pr.gridBuildCompute;
        case Stage::NeighborGather:return pr.neighborCompute;
        case Stage::Density:       return pr.densityCompute;
        case Stage::LambdaSolve:   return pr.lambdaCompute;
        case Stage::Integration:   return pr.integrateCompute;
        case Stage::VelocityUpdate:return pr.velocityCompute;
        case Stage::Boundary:      return pr.boundaryCompute;
        case Stage::XSPH:          return pr.xsphCompute;
        }
        return pr.coreCompute;
    }

    inline bool StageWantsHalfLoad(const SimParams& p, Stage s) {
        const auto& pr = p.precision;
        NumericType ty = StageComputeType(pr, s);
        if (ty != NumericType::FP16 && ty != NumericType::FP16_Packed) return false;
        switch (s) {
        case Stage::Integration:
        case Stage::VelocityUpdate:
        case Stage::Boundary:
        case Stage::Emission:
        case Stage::GridBuild:
        case Stage::LambdaSolve:
        case Stage::XSPH:
            return true;
        default:
            return true;
        }
    }

    // 修改 forward 声明为 struct 匹配实际定义，消除 C4099 警告
    inline bool UseHalfForPosition(const SimParams& p, Stage s, const struct DeviceBuffers& bufs) {
        if (!StageWantsHalfLoad(p, s)) return false;
        if (p.precision.positionStore != NumericType::FP16_Packed &&
            p.precision.positionStore != NumericType::FP16 &&
            p.precision.predictedPosStore != NumericType::FP16_Packed &&
            p.precision.predictedPosStore != NumericType::FP16) return false;
        if (!bufs.d_pos_h4 && !bufs.d_pos_pred_h4) return false;
        return true;
    }

    inline bool UseHalfForVelocity(const SimParams& p, Stage s, const struct DeviceBuffers& bufs) {
        if (!StageWantsHalfLoad(p, s)) return false;
        if (p.precision.velocityStore != NumericType::FP16_Packed &&
            p.precision.velocityStore != NumericType::FP16) return false;
        if (!bufs.d_vel_h4) return false;
        return true;
    }

} // namespace sim