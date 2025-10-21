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
        bool      posPredExternal = false; // 单外部预测或镜像模式
        bool      externalPingPong = false; // 真正双外部 ping-pong 模式
        bool      ownsPosBuffers = true;    // 是否由内部分配并负责释放位置缓冲

        bool usePosHalf = false;
        bool useVelHalf = false;
        bool usePosPredHalf = false;

        uint32_t guardA = 0xA11CE5u;
        uint32_t guardB = 0xBEEFBEEFu;

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

        // 内部分配
        void allocate(uint32_t cap) {
            allocateInternal(cap, false, false, false);
            ensurePrevSnapshot();
            BindDeviceGlobalsFrom(*this);
        }
        void allocateWithPrecision(const sim::SimPrecision& prec, uint32_t cap) {
            bool posH = (prec.positionStore == sim::NumericType::FP16_Packed || prec.positionStore == sim::NumericType::FP16);
            bool velH = (prec.velocityStore == sim::NumericType::FP16_Packed || prec.velocityStore == sim::NumericType::FP16);
            bool predH = (prec.predictedPosStore == sim::NumericType::FP16_Packed || prec.predictedPosStore == sim::NumericType::FP16);
            allocateInternal(cap, posH, velH, predH);
            ensurePrevSnapshot();
            BindDeviceGlobalsFrom(*this);
        }

        // 真正双外部 ping-pong 绑定（零拷贝）
        void bindExternalPosPingPong(float4* ptrA, float4* ptrB, uint32_t cap) {
            checkGuards("bindExternalPingPong.before");
            std::fprintf(stderr,
                "[ExternalPosPP][Bind.Enter] A=%p B=%p oldCurr=%p oldNext=%p externalPingPong(old)=%d posPredExternal(old)=%d owns=%d\n",
                (void*)ptrA, (void*)ptrB, (void*)d_pos_curr, (void*)d_pos_next, (int)externalPingPong, (int)posPredExternal, (int)ownsPosBuffers);
            if (ownsPosBuffers) {
                if (d_pos_curr) cudaFree(d_pos_curr);
                if (d_pos_next) cudaFree(d_pos_next);
                d_pos_curr = d_pos_next = nullptr;
                d_pos = d_pos_pred = nullptr;
                ownsPosBuffers = false;
            }
            capacity = cap;
            d_pos_curr = ptrA;
            d_pos_next = ptrB;
            d_pos = d_pos_curr;
            d_pos_pred = d_pos_next;
            // 关键：双外部语义不再使用“外部预测镜像”标志，避免进入镜像/复制分支
            posPredExternal = false;         // 修正：改为 false
            externalPingPong = true;
            usePosHalf = usePosPredHalf = false;
            d_pos_h4 = d_pos_pred_h4 = nullptr;
            BindDeviceGlobalsFrom(*this);
            std::fprintf(stderr,
                "[ExternalPosPP][Bind.Applied] curr=%p next=%p pred=%p externalPingPong=1 posPredExternal=0 capacity=%u\n",
                (void*)d_pos_curr, (void*)d_pos_next, (void*)d_pos_pred, capacity);
            checkGuards("bindExternalPingPong.after");
        }

        void swapPositionPingPong() {
            checkGuards("swapPositionPingPong.before");
            if (externalPingPong) {
                // 纯外部双缓冲：直接交换指针，无任何复制
                std::swap(d_pos_curr, d_pos_next);
                d_pos = d_pos_curr;
                d_pos_pred = d_pos_next; // 保持 predicted 语义 = next
                std::fprintf(stderr,
                    "[SwapPP][External] curr=%p next=%p pred=%p\n",
                    (void*)d_pos_curr, (void*)d_pos_next, (void*)d_pos_pred);
                BindDeviceGlobalsFrom(*this);
                checkGuards("swapPositionPingPong.after");
                return;
            }
            // 镜像/内部路径
            float4* oldCurr = d_pos_curr;
            float4* oldNext = d_pos_next;
            if (posPredExternal) {
                std::swap(d_pos_curr, d_pos_next);
                d_pos = d_pos_curr; // 不触碰外部 d_pos_pred（镜像模式）
                std::fprintf(stderr,
                    "[SwapPP][SkipPred] curr=%p next=%p pred(external)=%p cap=%u\n",
                    (void*)d_pos_curr, (void*)d_pos_next, (void*)d_pos_pred, capacity);
            } else {
                std::swap(d_pos_curr, d_pos_next);
                d_pos = d_pos_curr;
                d_pos_pred = d_pos_next;
                std::fprintf(stderr,
                    "[SwapPP][Internal] curr=%p next=%p pred=%p cap=%u\n",
                    (void*)d_pos_curr, (void*)d_pos_next, (void*)d_pos_pred, capacity);
            }
            BindDeviceGlobalsFrom(*this);
            checkGuards("swapPositionPingPong.after");
        }

        // 单外部预测（镜像）
        void bindExternalPosPred(float4* ptr) {
            checkGuards("bindExternal.before");
            std::fprintf(stderr,
                "[ExternalPred][Bind.Enter] newPtr=%p oldPred=%p posPredExternal(old)=%d next(internal)=%p\n",
                (void*)ptr, (void*)d_pos_pred, (int)posPredExternal, (void*)d_pos_next);
            if (d_pos_pred && !posPredExternal && ownsPosBuffers) {
                cudaFree(d_pos_pred);
            }
            d_pos_pred = ptr;
            posPredExternal = true;
            externalPingPong = false; // 只是镜像
            BindDeviceGlobalsFrom(*this);
            std::fprintf(stderr,
                "[ExternalPred][Bind.Applied] pred=%p next(internal)=%p mode=MirrorOnly external=1\n",
                (void*)d_pos_pred, (void*)d_pos_next);
            checkGuards("bindExternal.after");
        }
        void detachExternalPosPred() {
            checkGuards("detachExternal.before");
            std::fprintf(stderr,
                "[PingPong][DetachExternal][Enter] predPtr=%p nextPtr=%p external(old)=%d pingPongExt=%d cap=%u\n",
                (void*)d_pos_pred, (void*)d_pos_next, (int)posPredExternal, (int)externalPingPong, capacity);
            d_pos_pred = nullptr;
            posPredExternal = false;
            externalPingPong = false;
            BindDeviceGlobalsFrom(*this);
            std::fprintf(stderr,
                "[PingPong][DetachExternal][Applied] pred=%p external=%d pingPongExt=%d\n",
                (void*)d_pos_pred, (int)posPredExternal, (int)externalPingPong);
            checkGuards("detachExternal.after");
        }

        void checkGuards(const char* tag) const {
            if (guardA != 0xA11CE5u || guardB != 0xBEEFBEEFu) {
                std::fprintf(stderr,
                    "[GuardCorrupt][%s] guardA=%08X guardB=%08X posPredExternal=%d externalPingPong=%d capacity=%u\n",
                    tag, guardA, guardB, (int)posPredExternal, (int)externalPingPong, capacity);
            }
        }

        void release();
        ~DeviceBuffers() { release(); }

    private:
        void allocateInternal(uint32_t cap, bool posH, bool velH, bool predH) {
            checkGuards("allocateInternal.before");
            std::fprintf(stderr,
                "[Origin][AllocateInternal] cap=%u posH=%d velH=%d predH=%d (prevCap=%u)\n",
                cap, (int)posH, (int)velH, (int)predH, capacity);

            if (cap == 0) cap = 1;
            if (cap == capacity && posH == usePosHalf && velH == useVelHalf && predH == usePosPredHalf) return;
            release();
            capacity = cap;
            usePosHalf = posH; useVelHalf = velH; usePosPredHalf = predH;
            ownsPosBuffers = true;
            externalPingPong = false; posPredExternal = false;
            auto alloc = [&](void** p, size_t elemSize) { cudaMalloc(p, elemSize * capacity); };
            // Allocate two position buffers for ping-pong unless later replaced externally
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
            std::fprintf(stderr,
                "[Origin][AllocateInternal][Done] cap=%u d_pos_curr=%p d_pos_next=%p posPredExternal=%d externalPingPong=%d\n",
                capacity, (void*)d_pos_curr, (void*)d_pos_next, (int)posPredExternal, (int)externalPingPong);
            checkGuards("allocateInternal.after");

        }
    };

    inline void DeviceBuffers::release() {
        checkGuards("release.before");
        std::fprintf(stderr,
            "[Origin][Release] capacity=%u posPredExternal=%d externalPingPong=%d ownsPos=%d\n",
            capacity, (int)posPredExternal, (int)externalPingPong, (int)ownsPosBuffers);
        auto fre = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        // 仅释放内部拥有的 position 缓冲
        if (ownsPosBuffers) { fre(d_pos_curr); fre(d_pos_next); }
        fre(d_vel_curr); fre(d_vel_prev);
        // 基础别名指针避免误释放外部（d_pos/d_pos_pred 与 curr/next 同步）
        if (ownsPosBuffers) { fre(d_pos); fre(d_pos_pred); } else { d_pos = nullptr; d_pos_pred = nullptr; }
        fre(d_lambda); fre(d_delta);
        fre(d_pos_h4); fre(d_vel_h4); fre(d_pos_pred_h4); fre(d_delta_h4); fre(d_prev_pos_h4);
        capacity = 0; posPredExternal = false; externalPingPong = false; ownsPosBuffers = true;
        checkGuards("release.after");
        usePosHalf = useVelHalf = usePosPredHalf = false;
    }

} // namespace sim