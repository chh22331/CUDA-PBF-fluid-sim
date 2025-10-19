#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <array>
#include "parameters.h"
#include "numeric_utils.h"
#include "logging.h" // 新增：日志（用于 allocate 失败等）
#include "device_globals.cuh"

namespace sim {

    struct Half4 { __half x, y, z, w; };

    void PackFloat4ToHalf4(const float4* src, Half4* dst, uint32_t N, cudaStream_t s);
    void UnpackHalf4ToFloat4(const Half4* src, float4* dst, uint32_t N, cudaStream_t s);

    struct DeviceBuffers {
        // Legacy base pointers (retain for external code expecting names)
        float4* d_pos = nullptr;        // will alias d_pos_curr
        float4* d_vel = nullptr;        // will alias d_vel_curr
        float4* d_pos_pred = nullptr;   // will alias d_pos_next
        float*  d_lambda = nullptr;
        float4* d_delta = nullptr;      // XSPH / 临时速度 (unused for final velocity in scheme A)

        // Ping-pong state (new)
        float4* d_pos_curr = nullptr;   // current (formal) positions
        float4* d_pos_next = nullptr;   // predicted / next positions
        float4* d_vel_curr = nullptr;   // current velocities
        float4* d_vel_prev = nullptr;   // optional previous velocities (future use)

        // 半精镜像
        Half4* d_pos_h4 = nullptr;
        Half4* d_vel_h4 = nullptr;
        Half4* d_pos_pred_h4 = nullptr; // alias of d_pos_next mirror
        Half4* d_delta_h4 = nullptr;
        Half4* d_prev_pos_h4 = nullptr;

        uint32_t  capacity = 0;
        bool      posPredExternal = false; // 预测位置由外部共享缓冲提供

        bool usePosHalf = false;
        bool useVelHalf = false;
        bool usePosPredHalf = false;

        bool anyHalf() const { return usePosHalf || useVelHalf || usePosPredHalf; }
        bool hasPrevSnapshot() const { return d_prev_pos_h4 != nullptr; }
       
        void ensurePrevSnapshot() { if (d_prev_pos_h4) { cudaFree(d_prev_pos_h4); d_prev_pos_h4 = nullptr; } if (capacity > 0) cudaMalloc((void**)&d_prev_pos_h4, sizeof(sim::Half4) * capacity); }
        void freePrevPosSnapshot() { if (d_prev_pos_h4) { cudaFree(d_prev_pos_h4); d_prev_pos_h4 = nullptr; } }
  
        // 半精打包/解包
        void packAllToHalf(uint32_t N, cudaStream_t s) {
            if (!anyHalf() || N == 0) return;
            struct PackItem { const float4* src; Half4* dst; bool enabled; };
            PackItem items[] = {
                { d_pos_curr,  d_pos_h4,       usePosHalf     && d_pos_curr  && d_pos_h4 },
                { d_vel_curr,  d_vel_h4,       useVelHalf     && d_vel_curr  && d_vel_h4 },
                { d_pos_next,  d_pos_pred_h4,  usePosPredHalf && d_pos_next  && d_pos_pred_h4 },
                { d_delta,     d_delta_h4,     useVelHalf     && d_delta     && d_delta_h4 }
            };
            for (auto& it : items) if (it.enabled) PackFloat4ToHalf4(it.src, it.dst, N, s);
        }
        void unpackAllFromHalf(uint32_t N, cudaStream_t s) {
            if (!anyHalf() || N == 0) return;
            struct UnpackItem { const Half4* src; float4* dst; bool enabled; };
            UnpackItem items[] = {
                { d_pos_h4,      d_pos_curr,  usePosHalf     && d_pos_curr  && d_pos_h4 },
                { d_vel_h4,      d_vel_curr,  useVelHalf     && d_vel_curr  && d_vel_h4 },
                { d_pos_pred_h4, d_pos_next,  usePosPredHalf && d_pos_next  && d_pos_pred_h4 },
                { d_delta_h4,    d_delta,     useVelHalf     && d_delta     && d_delta_h4 }
            };
            for (auto& it : items) if (it.enabled) UnpackHalf4ToFloat4(it.src, it.dst, N, s);
        }

        // Swap position ping-pong (after frame)
        // 新增：在 allocate 与 swap 后调用指针绑定
        void allocate(uint32_t cap) {
            allocateInternal(cap, false, false, false);
            ensurePrevSnapshot();
            BindDeviceGlobalsFrom(*this); // 新增
        }
        void allocateWithPrecision(const sim::SimPrecision& prec, uint32_t cap) {
            bool posH = (prec.positionStore == sim::NumericType::FP16_Packed || prec.positionStore == sim::NumericType::FP16);
            bool velH = (prec.velocityStore == sim::NumericType::FP16_Packed || prec.velocityStore == sim::NumericType::FP16);
            bool predH = (prec.predictedPosStore == sim::NumericType::FP16_Packed || prec.predictedPosStore == sim::NumericType::FP16);
            allocateInternal(cap, posH, velH, predH);
            ensurePrevSnapshot();
            BindDeviceGlobalsFrom(*this); // 新增
        }

        void swapPositionPingPong() {
            std::swap(d_pos_curr, d_pos_next);
            d_pos = d_pos_curr;
            d_pos_pred = d_pos_next;
            // 绑定更新：位置指针交换后需要刷新符号
            BindDeviceGlobalsFrom(*this); // 新增
        }

        void bindExternalPosPred(float4* ptr) {
            if (d_pos_pred && !posPredExternal) cudaFree(d_pos_pred);
            d_pos_pred = ptr;
            d_pos_next = ptr;
            posPredExternal = true;
            BindDeviceGlobalsFrom(*this); // 新增：外部预测缓冲绑定后刷新
        }
        void detachExternalPosPred() {
            d_pos_pred = nullptr;
            d_pos_next = nullptr;
            posPredExternal = false;
            BindDeviceGlobalsFrom(*this); // 新增：解除后符号指针置空
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
            // Allocate two position buffers for ping-pong unless external predicted supplied later
            alloc((void**)&d_pos_curr, sizeof(float4));
            alloc((void**)&d_pos_next, sizeof(float4));
            d_pos = d_pos_curr; d_pos_pred = d_pos_next; // legacy alias
            // Velocity single buffer
            alloc((void**)&d_vel_curr, sizeof(float4));
            d_vel = d_vel_curr;
            alloc((void**)&d_lambda, sizeof(float));
            alloc((void**)&d_delta, sizeof(float4));
            if (usePosHalf)      alloc((void**)&d_pos_h4,       sizeof(Half4));
            if (useVelHalf)      alloc((void**)&d_vel_h4,       sizeof(Half4));
            if (usePosPredHalf)  alloc((void**)&d_pos_pred_h4,  sizeof(Half4));
            if (useVelHalf)      alloc((void**)&d_delta_h4,     sizeof(Half4));
        }
    };

    inline void DeviceBuffers::release() {
        auto fre = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fre(d_pos_curr); fre(d_pos_next); fre(d_vel_curr); fre(d_vel_prev);
        fre(d_pos); fre(d_vel); fre(d_pos_pred); fre(d_lambda); fre(d_delta);
        fre(d_pos_h4); fre(d_vel_h4); fre(d_pos_pred_h4); fre(d_delta_h4); fre(d_prev_pos_h4);
        capacity = 0; posPredExternal = false;
        usePosHalf = useVelHalf = usePosPredHalf = false;
    }

} // namespace sim