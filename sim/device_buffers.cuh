#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <array>
#include "parameters.h"
#include "numeric_utils.h"
#include "logging.h" // 新增：日志（用于 allocate失败等）
#include "device_globals.cuh"

namespace sim {

    struct Half4 { __half x, y, z, w; };

    void PackFloat4ToHalf4(const float4* src, Half4* dst, uint32_t N, cudaStream_t s);
    void UnpackHalf4ToFloat4(const Half4* src, float4* dst, uint32_t N, cudaStream_t s);
    // 新增：标量数组 pack/unpack（lambda / density / aux）
    void PackFloatToHalf(const float* src, __half* dst, uint32_t N, cudaStream_t s);
    void UnpackHalfToFloat(const __half* src, float* dst, uint32_t N, cudaStream_t s);

    struct DeviceBuffers {
        // Legacy base pointers (retain for external code expecting names)
        float4* d_pos = nullptr;        // will alias d_pos_curr
        float4* d_vel = nullptr;        // will alias d_vel_curr
        float4* d_pos_pred = nullptr;   // will alias d_pos_next
        float*  d_lambda = nullptr;
        float4* d_delta = nullptr;      // XSPH / 临时速度 (unused for final velocity in scheme A)
        float*  d_density = nullptr;    // 新增：密度（若后续需要主机侧统计）
        float*  d_aux = nullptr;        // 新增：辅助（梯度等临时累加器）

        // Ping-pong state (new)
        float4* d_pos_curr = nullptr;   // current (formal) positions
        float4* d_pos_next = nullptr;   // predicted / next positions
        float4* d_vel_curr = nullptr;   // current velocities
        float4* d_vel_prev = nullptr;   // optional previous velocities (future use)

        // 半精镜像（向量）
        Half4* d_pos_h4 = nullptr;
        Half4* d_vel_h4 = nullptr;
        Half4* d_pos_pred_h4 = nullptr; // alias of d_pos_next mirror
        Half4* d_delta_h4 = nullptr;
        Half4* d_prev_pos_h4 = nullptr;
        // 新增：渲染半精镜像（若 positionStore不是 half但 renderTransfer 要求 half）
        Half4* d_render_pos_h4 = nullptr;

        // 半精镜像（标量）
        __half* d_lambda_h = nullptr;
        __half* d_density_h = nullptr;
        __half* d_aux_h = nullptr;

        uint32_t  capacity = 0;
        bool      posPredExternal = false; // 单外部预测或镜像模式
        bool      externalPingPong = false; // 真正双外部 ping-pong 模式
        bool      ownsPosBuffers = true;    // 是否由内部分配并负责释放位置缓冲

        bool usePosHalf = false;
        bool useVelHalf = false;
        bool usePosPredHalf = false;
        bool useLambdaHalf = false;
        bool useDensityHalf = false;
        bool useAuxHalf = false;
        bool useRenderHalf = false; // 新增

        uint32_t guardA = 0xA11CE5u;
        uint32_t guardB = 0xBEEFBEEFu;

        bool anyHalf() const { return usePosHalf || useVelHalf || usePosPredHalf || useLambdaHalf || useDensityHalf || useAuxHalf || useRenderHalf; }
        bool hasPrevSnapshot() const { return d_prev_pos_h4 != nullptr; }
       
        void ensurePrevSnapshot() { if (d_prev_pos_h4) { cudaFree(d_prev_pos_h4); d_prev_pos_h4 = nullptr; } if (capacity > 0) cudaMalloc((void**)&d_prev_pos_h4, sizeof(sim::Half4) * capacity); }
        void freePrevPosSnapshot() { if (d_prev_pos_h4) { cudaFree(d_prev_pos_h4); d_prev_pos_h4 = nullptr; } }
  
        // 半精打包/解包
        void packAllToHalf(uint32_t N, cudaStream_t s) {
            if (!anyHalf() || N == 0) return;
            struct PackItemVec { const float4* src; Half4* dst; bool enabled; };
            PackItemVec vitems[] = {
                { d_pos_curr,  d_pos_h4,       usePosHalf     && d_pos_curr  && d_pos_h4 },
                { d_vel_curr,  d_vel_h4,       useVelHalf     && d_vel_curr  && d_vel_h4 },
                { d_pos_next,  d_pos_pred_h4,  usePosPredHalf && d_pos_next  && d_pos_pred_h4 },
                { d_delta,     d_delta_h4,     useVelHalf     && d_delta     && d_delta_h4 }
            };
            for (auto& it : vitems) if (it.enabled) PackFloat4ToHalf4(it.src, it.dst, N, s);
            // 标量
            struct PackItemSca { const float* src; __half* dst; bool enabled; };
            PackItemSca sitems[] = {
                { d_lambda,   d_lambda_h,     useLambdaHalf   && d_lambda   && d_lambda_h },
                { d_density,  d_density_h,    useDensityHalf  && d_density  && d_density_h },
                { d_aux,      d_aux_h,        useAuxHalf      && d_aux      && d_aux_h }
            };
            for (auto& it : sitems) if (it.enabled) PackFloatToHalf(it.src, it.dst, N, s);
            // 渲染位置（独立镜像，仅当 positionStore不是 half 且 renderTransfer half）
            if (useRenderHalf && d_render_pos_h4 && d_pos_curr) PackFloat4ToHalf4(d_pos_curr, d_render_pos_h4, N, s);
        }
        void unpackAllFromHalf(uint32_t N, cudaStream_t s) {
            if (!anyHalf() || N == 0) return;
            struct UnpackItemVec { const Half4* src; float4* dst; bool enabled; };
            UnpackItemVec vitems[] = {
                { d_pos_h4,      d_pos_curr,  usePosHalf     && d_pos_curr  && d_pos_h4 },
                { d_vel_h4,      d_vel_curr,  useVelHalf     && d_vel_curr  && d_vel_h4 },
                { d_pos_pred_h4, d_pos_next,  usePosPredHalf && d_pos_next  && d_pos_pred_h4 },
                { d_delta_h4,    d_delta,     useVelHalf     && d_delta     && d_delta_h4 }
            };
            for (auto& it : vitems) if (it.enabled) UnpackHalf4ToFloat4(it.src, it.dst, N, s);
            struct UnpackItemSca { const __half* src; float* dst; bool enabled; };
            UnpackItemSca sitems[] = {
                { d_lambda_h,    d_lambda,     useLambdaHalf   && d_lambda   && d_lambda_h },
                { d_density_h,   d_density,    useDensityHalf  && d_density  && d_density_h },
                { d_aux_h,       d_aux,        useAuxHalf      && d_aux      && d_aux_h }
            };
            for (auto& it : sitems) if (it.enabled) UnpackHalfToFloat(it.src, it.dst, N, s);
        }

        // 单独渲染打包（避免与通用 packAll 混淆）
        void packRenderToHalf(uint32_t N, cudaStream_t s) {
            if (!useRenderHalf || !d_render_pos_h4 || !d_pos_curr) return; PackFloat4ToHalf4(d_pos_curr, d_render_pos_h4, N, s); }

        // 内部分配
        void allocate(uint32_t cap) {
            allocateInternal(cap, false, false, false, false, false, false);
            ensurePrevSnapshot();
            BindDeviceGlobalsFrom(*this);
        }
        void allocateWithPrecision(const sim::SimPrecision& prec, uint32_t cap) {
            bool posH = (prec.positionStore == sim::NumericType::FP16_Packed || prec.positionStore == sim::NumericType::FP16);
            bool velH = (prec.velocityStore == sim::NumericType::FP16_Packed || prec.velocityStore == sim::NumericType::FP16);
            bool predH = (prec.predictedPosStore == sim::NumericType::FP16_Packed || prec.predictedPosStore == sim::NumericType::FP16);
            bool lambdaH = (prec.lambdaStore == sim::NumericType::FP16_Packed || prec.lambdaStore == sim::NumericType::FP16);
            bool densityH = (prec.densityStore == sim::NumericType::FP16_Packed || prec.densityStore == sim::NumericType::FP16);
            bool auxH = (prec.auxStore == sim::NumericType::FP16_Packed || prec.auxStore == sim::NumericType::FP16);
            // 渲染半精：若 renderTransfer 是 half 且 positionStore不是 half，单独分配
            bool renderH = ((prec.renderTransfer == sim::NumericType::FP16_Packed || prec.renderTransfer == sim::NumericType::FP16) && !posH);
            allocateInternal(cap, posH, velH, predH, lambdaH, densityH, auxH);
            useRenderHalf = renderH;
            if (useRenderHalf) {
                cudaMalloc((void**)&d_render_pos_h4, sizeof(Half4) * capacity);
            }
            ensurePrevSnapshot();
            BindDeviceGlobalsFrom(*this);
        }

        // 真正双外部 ping-pong 绑定（零拷贝）
        void bindExternalPosPingPong(float4* ptrA, float4* ptrB, uint32_t cap) {
            checkGuards("bindExternalPingPong.before");
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
            usePosHalf = usePosPredHalf = false; // 外部缓冲不做 half 镜像
            d_pos_h4 = d_pos_pred_h4 = nullptr;
            BindDeviceGlobalsFrom(*this);
            checkGuards("bindExternalPingPong.after");
        }

        void swapPositionPingPong() {
            checkGuards("swapPositionPingPong.before");
            if (externalPingPong) {
                // 纯外部双缓冲：直接交换指针，无任何复制
                std::swap(d_pos_curr, d_pos_next);
                d_pos = d_pos_curr;
                d_pos_pred = d_pos_next; // 保持 predicted 语义 = next
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
            d_pos_pred = nullptr;
            posPredExternal = false;
            externalPingPong = false;
            BindDeviceGlobalsFrom(*this);
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
        void allocateInternal(uint32_t cap, bool posH, bool velH, bool predH, bool lambdaH, bool densityH, bool auxH) {
            checkGuards("allocateInternal.before");
            std::fprintf(stderr,
                "[Origin][AllocateInternal] cap=%u posH=%d velH=%d predH=%d lambdaH=%d densityH=%d auxH=%d (prevCap=%u)\n",
                cap, (int)posH, (int)velH, (int)predH, (int)lambdaH, (int)densityH, (int)auxH, capacity);

            if (cap == 0) cap = 1;
            if (cap == capacity && posH == usePosHalf && velH == useVelHalf && predH == usePosPredHalf && lambdaH == useLambdaHalf && densityH == useDensityHalf && auxH == useAuxHalf) return;
            release();
            capacity = cap;
            usePosHalf = posH; useVelHalf = velH; usePosPredHalf = predH; useLambdaHalf = lambdaH; useDensityHalf = densityH; useAuxHalf = auxH;
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
            alloc((void**)&d_density, sizeof(float));
            alloc((void**)&d_aux, sizeof(float));
            if (usePosHalf)      alloc((void**)&d_pos_h4,       sizeof(Half4));
            if (useVelHalf)      alloc((void**)&d_vel_h4,       sizeof(Half4));
            if (usePosPredHalf)  alloc((void**)&d_pos_pred_h4,  sizeof(Half4));
            if (useVelHalf)      alloc((void**)&d_delta_h4,     sizeof(Half4));
            if (useLambdaHalf)   alloc((void**)&d_lambda_h,     sizeof(__half));
            if (useDensityHalf)  alloc((void**)&d_density_h,    sizeof(__half));
            if (useAuxHalf)      alloc((void**)&d_aux_h,       sizeof(__half));
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
        fre(d_lambda); fre(d_delta); fre(d_density); fre(d_aux);
        fre(d_pos_h4); fre(d_vel_h4); fre(d_pos_pred_h4); fre(d_delta_h4); fre(d_prev_pos_h4);
        fre(d_lambda_h); fre(d_density_h); fre(d_aux_h);
        fre(d_render_pos_h4);
        capacity = 0; posPredExternal = false; externalPingPong = false; ownsPosBuffers = true;
        checkGuards("release.after");
        usePosHalf = useVelHalf = usePosPredHalf = useLambdaHalf = useDensityHalf = useAuxHalf = false;
    }

} // namespace sim