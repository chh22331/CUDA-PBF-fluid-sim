#pragma once
#include "parameters.h"

namespace sim {

    // 枚举内部使用（与 console::ComputeStageBits 概念对应）
    enum class Stage {
        Emission,
        GridBuild,
        NeighborGather, // (当前未单独有内核，预留)
        Density,        // (PBF 中等价于 rho 部分，归入 LambdaSolve 前半，可与 LambdaSolve 合并)
        LambdaSolve,    // Lambda + DeltaApply
        Integration,
        VelocityUpdate,
        Boundary,
        XSPH
    };

    // 将 SimPrecision 中对应阶段 compute 取出（已在 BuildSimParams 映射）
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

    // 判定：该阶段是否“允许”用半精镜像做只读加载（算术仍 FP32）
    inline bool StageWantsHalfLoad(const SimParams& p, Stage s) {
        const auto& pr = p.precision;
        NumericType ty = StageComputeType(pr, s);
        if (ty != NumericType::FP16 && ty != NumericType::FP16_Packed) return false;
        // 若 coreCompute=FP16 但阶段 override 为 FP32，则不使用
        // 若阶段 override=FP16 则优先
        // 仅支持 Packed 模式下的高效半精读取
        switch (s) {
        case Stage::Integration:
        case Stage::VelocityUpdate:
        case Stage::Boundary:
        case Stage::Emission:
        case Stage::GridBuild:
        case Stage::LambdaSolve:
        case Stage::XSPH:
            // 必须存在对应 half 镜像（位置 / 速度）
            return true;
        default:
            return true;
        }
    }

    // 汇总：是否真的使用半精（需要存储类型、镜像指针与阶段意愿均满足）
    inline bool UseHalfForPosition(const SimParams& p, Stage s, const class DeviceBuffers& bufs) {
        if (!StageWantsHalfLoad(p, s)) return false;
        if (p.precision.positionStore != NumericType::FP16_Packed &&
            p.precision.positionStore != NumericType::FP16 &&
            p.precision.predictedPosStore != NumericType::FP16_Packed &&
            p.precision.predictedPosStore != NumericType::FP16) return false;
        if (!bufs.d_pos_h4 && !bufs.d_pos_pred_h4) return false;
        return true;
    }

    inline bool UseHalfForVelocity(const SimParams& p, Stage s, const class DeviceBuffers& bufs) {
        if (!StageWantsHalfLoad(p, s)) return false;
        if (p.precision.velocityStore != NumericType::FP16_Packed &&
            p.precision.velocityStore != NumericType::FP16) return false;
        if (!bufs.d_vel_h4) return false;
        return true;
    }

} // namespace sim