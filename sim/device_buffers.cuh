#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <array>
#include "parameters.h"
#include "numeric_utils.h"
#include "logging.h" // 新增：日志（用于 allocate 失败等）

namespace sim {

    struct Half4 { __half x, y, z, w; };

    // 前置包装函数声明（实现于 device_buffers_kernels.cu）
    void PackFloat4ToHalf4(const float4* src, Half4* dst, uint32_t N, cudaStream_t s);
    void UnpackHalf4ToFloat4(const Half4* src, float4* dst, uint32_t N, cudaStream_t s);

    // 专注粒子主/镜像数据；网格/排序缓冲已迁移至 GridBuffers
    struct DeviceBuffers {
        // FP32 主数据
        float4* d_pos = nullptr;
        float4* d_vel = nullptr;
        float4* d_pos_pred = nullptr;
        float*  d_lambda = nullptr;
        float4* d_delta = nullptr; // XSPH / 临时速度 (ping-pong)

        // 半精镜像
        Half4* d_pos_h4 = nullptr;
        Half4* d_vel_h4 = nullptr;
        Half4* d_pos_pred_h4 = nullptr;
        Half4* d_delta_h4 = nullptr;     // 新增：与 d_delta 配套，用于 XSPH 后 ping-pong
        Half4* d_prev_pos_h4 = nullptr;  // 位置快照（用于诊断/回溯）

        uint32_t  capacity = 0;
        bool      posPredExternal = false; // 预测位置由外部共享缓冲提供

        // 精度标志（将被后续 PrecisionState 替换）
        bool usePosHalf = false;
        bool useVelHalf = false;
        bool usePosPredHalf = false;

        bool anyHalf() const { return usePosHalf || useVelHalf || usePosPredHalf; }
        bool hasPrevSnapshot() const { return d_prev_pos_h4 != nullptr; }

        // 分配接口
        void allocate(uint32_t cap) {
            allocateInternal(cap, false, false, false);
            ensurePrevSnapshot();
        }
        void allocateWithPrecision(const sim::SimPrecision& prec, uint32_t cap) {
            bool posH  = (prec.positionStore     == sim::NumericType::FP16_Packed || prec.positionStore     == sim::NumericType::FP16);
            bool velH  = (prec.velocityStore     == sim::NumericType::FP16_Packed || prec.velocityStore     == sim::NumericType::FP16);
            bool predH = (prec.predictedPosStore == sim::NumericType::FP16_Packed || prec.predictedPosStore == sim::NumericType::FP16);
            allocateInternal(cap, posH, velH, predH);
            ensurePrevSnapshot();
        }

        void ensurePrevSnapshot() {
            if (d_prev_pos_h4) { cudaFree(d_prev_pos_h4); d_prev_pos_h4 = nullptr; }
            if (capacity > 0) cudaMalloc((void**)&d_prev_pos_h4, sizeof(sim::Half4) * capacity);
        }
        void freePrevPosSnapshot() { if (d_prev_pos_h4) { cudaFree(d_prev_pos_h4); d_prev_pos_h4 = nullptr; } }

        void bindExternalPosPred(float4* ptr) {
            if (d_pos_pred && !posPredExternal) cudaFree(d_pos_pred);
            d_pos_pred = ptr; posPredExternal = true;
        }
        void detachExternalPosPred() { d_pos_pred = nullptr; posPredExternal = false; }

        // 半精打包/解包
        void packAllToHalf(uint32_t N, cudaStream_t s) {
            if (!anyHalf() || N == 0) return;
            struct PackItem { const float4* src; Half4* dst; bool enabled; };
            PackItem items[] = {
                { d_pos,       d_pos_h4,       usePosHalf     && d_pos       && d_pos_h4 },
                { d_vel,       d_vel_h4,       useVelHalf     && d_vel       && d_vel_h4 },
                { d_pos_pred,  d_pos_pred_h4,  usePosPredHalf && d_pos_pred  && d_pos_pred_h4 },
                { d_delta,     d_delta_h4,     useVelHalf     && d_delta     && d_delta_h4 } // 新增：delta
            };
            for (auto& it : items) if (it.enabled) PackFloat4ToHalf4(it.src, it.dst, N, s);
        }
        void unpackAllFromHalf(uint32_t N, cudaStream_t s) {
            if (!anyHalf() || N == 0) return;
            struct UnpackItem { const Half4* src; float4* dst; bool enabled; };
            UnpackItem items[] = {
                { d_pos_h4,      d_pos,       usePosHalf     && d_pos       && d_pos_h4 },
                { d_vel_h4,      d_vel,       useVelHalf     && d_vel       && d_vel_h4 },
                { d_pos_pred_h4, d_pos_pred,  usePosPredHalf && d_pos_pred  && d_pos_pred_h4 },
                { d_delta_h4,    d_delta,     useVelHalf     && d_delta     && d_delta_h4 } // 新增：delta
            };
            for (auto& it : items) if (it.enabled) UnpackHalf4ToFloat4(it.src, it.dst, N, s);
        }

        void release();
        ~DeviceBuffers() { release(); }

    private:
        void allocateInternal(uint32_t cap, bool posH, bool velH, bool predH) {
            if (cap == 0) cap = 1;
            if (cap == capacity && posH == usePosHalf && velH == useVelHalf && predH == usePosPredHalf) return;
            release();
            capacity = cap;
            usePosHalf = posH; useVelHalf = velH; usePosPredHalf = predH;
            auto alloc = [&](void** p, size_t elemSize) { cudaMalloc(p, elemSize * capacity); };
            alloc((void**)&d_pos, sizeof(float4));
            alloc((void**)&d_vel, sizeof(float4));
            if (!posPredExternal) alloc((void**)&d_pos_pred, sizeof(float4));
            alloc((void**)&d_lambda, sizeof(float));
            alloc((void**)&d_delta, sizeof(float4));
            if (usePosHalf)      alloc((void**)&d_pos_h4,       sizeof(Half4));
            if (useVelHalf)      alloc((void**)&d_vel_h4,       sizeof(Half4));
            if (usePosPredHalf)  alloc((void**)&d_pos_pred_h4,  sizeof(Half4));
            if (useVelHalf)      alloc((void**)&d_delta_h4,     sizeof(Half4)); // 新增：与 velHalf 同条件
        }
    };

    inline void DeviceBuffers::release() {
        auto fre = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fre(d_pos); fre(d_vel); fre(d_lambda); fre(d_delta);
        fre(d_pos_h4); fre(d_vel_h4); fre(d_pos_pred_h4); fre(d_delta_h4); fre(d_prev_pos_h4);
        if (!posPredExternal) fre(d_pos_pred);
        capacity = 0; posPredExternal = false;
        usePosHalf = useVelHalf = usePosPredHalf = false;
    }

} // namespace sim