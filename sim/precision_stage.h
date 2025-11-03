#pragma once
#include "parameters.h"

namespace sim {

    struct HalfLoadPolicy {
        // 临时开关：允许在 compute=FP32 但存储=FP16_Packed 时仍走 half 只读加载
        static bool allowFp16LoadOnFp32() { return true; }
    };

    enum class Stage {
        Emission, GridBuild, NeighborGather, Density, LambdaSolve,
        Integration, VelocityUpdate, Boundary, XSPH
    };

    inline NumericType StageComputeType(const SimPrecision& pr, Stage s) {
        switch (s) {
        case Stage::Emission:       return pr.emissionCompute;
        case Stage::GridBuild:      return pr.gridBuildCompute;
        case Stage::NeighborGather: return pr.neighborCompute;
        case Stage::Density:        return pr.densityCompute;
        case Stage::LambdaSolve:    return pr.lambdaCompute;
        case Stage::Integration:    return pr.integrateCompute;
        case Stage::VelocityUpdate: return pr.velocityCompute;
        case Stage::Boundary:       return pr.boundaryCompute;
        case Stage::XSPH:           return pr.xsphCompute;
        }
        return pr.coreCompute;
    }

    inline bool StageWantsHalfLoad(const SimParams& p, Stage s) {
        const auto& pr = p.precision;
        NumericType ct = StageComputeType(pr, s);
        if (ct == NumericType::FP16 || ct == NumericType::FP16_Packed)
            return true;
        // 带宽优化：允许 FP32 计算读取 half 存储
        if (HalfLoadPolicy::allowFp16LoadOnFp32()) {
            switch (s) {
            case Stage::Integration:
            case Stage::VelocityUpdate:
            case Stage::Boundary:
            case Stage::XSPH:
            case Stage::LambdaSolve:
                return true; // 这些阶段典型存在大量只读邻域访问
            default: break;
            }
        }
        return false;
    }

    inline bool UseHalfForPosition(const SimParams& p, Stage s, const class DeviceBuffers& bufs) {
        if (!StageWantsHalfLoad(p, s)) return false;
        bool storeHalf =
            (p.precision.positionStore == NumericType::FP16_Packed ||
             p.precision.positionStore == NumericType::FP16) ||
            (p.precision.predictedPosStore == NumericType::FP16_Packed ||
             p.precision.predictedPosStore == NumericType::FP16);
        if (!storeHalf) return false;
        if (!bufs.d_pos_h4 && !bufs.d_pos_pred_h4) return false;
        return true;
    }

    inline bool UseHalfForVelocity(const SimParams& p, Stage s, const class DeviceBuffers& bufs) {
        if (!StageWantsHalfLoad(p, s)) return false;
        if (p.precision.velocityStore != NumericType::FP16_Packed &&
            p.precision.velocityStore != NumericType::FP16) return false;
        return bufs.d_vel_h4 != nullptr;
    }

} // namespace sim