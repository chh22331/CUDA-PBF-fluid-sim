#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <array>
#include "parameters.h"
#include "numeric_utils.h"
#include "logging.h"
#include "device_globals.cuh"

namespace sim {

struct Half4 { __half x, y, z, w; };

void PackFloat4ToHalf4(const float4* src, Half4* dst, uint32_t N, cudaStream_t s);
void UnpackHalf4ToFloat4(const Half4* src, float4* dst, uint32_t N, cudaStream_t s);
void PackFloatToHalf(const float* src, __half* dst, uint32_t N, cudaStream_t s);
void UnpackHalfToFloat(const __half* src, float* dst, uint32_t N, cudaStream_t s);

#ifndef CUDA_CHECK_ALLOC
#define CUDA_CHECK_ALLOC(expr) do { \
    cudaError_t _e = (expr); \
    if(_e != cudaSuccess){ \
        std::fprintf(stderr,"[Alloc][Error] %s (%d)\n", cudaGetErrorString(_e), (int)_e); \
    } \
} while(0)
#endif

struct DeviceBuffers {
    // ===== FP32 primary =====
    float4* d_pos = nullptr;       // alias -> d_pos_curr
    float4* d_vel = nullptr;       // alias -> d_vel_curr
    float4* d_pos_pred = nullptr;  // alias -> d_pos_next
    float*  d_lambda = nullptr;
    float4* d_delta = nullptr;
    float*  d_density = nullptr;
    float*  d_aux = nullptr;

    // Ping-pong (双缓冲)
    float4* d_pos_curr = nullptr;
    float4* d_pos_next = nullptr;
    float4* d_vel_curr = nullptr;
    float4* d_vel_prev = nullptr;

    // ===== Half mirrors =====
    Half4* d_pos_h4 = nullptr;
    Half4* d_vel_h4 = nullptr;
    Half4* d_pos_pred_h4 = nullptr;
    Half4* d_prev_pos_h4 = nullptr;
    Half4* d_render_pos_h4 = nullptr;

    __half* d_lambda_h = nullptr;
    __half* d_density_h = nullptr;
    __half* d_aux_h = nullptr;

    // ===== 原生 half 主存储 =====
    bool   nativeHalfActive = false;
    Half4* d_pos_curr_native = nullptr;
    Half4* d_pos_next_native = nullptr;
    Half4* d_vel_curr_native = nullptr;

    // ===== NEW: 单块连续分配指针（位置 / half） =====
    float4* d_pos_block = nullptr;       // [curr | next]
    Half4*  d_pos_half_block = nullptr;  // [curr_half | next_half]

    uint32_t capacity = 0;
    bool posPredExternal = false;
    bool externalPingPong = false;
    bool ownsPosBuffers = true;

    bool usePosHalf = false;
    bool useVelHalf = false;
    bool usePosPredHalf = false;
    bool useLambdaHalf = false;
    bool useDensityHalf = false;
    bool useAuxHalf = false;
    bool useRenderHalf = false;

    bool externalPosPredAlias = false;

    uint32_t guardA = 0xA11CE5u;
    uint32_t guardB = 0xBEEFBEEFu;

    bool anyHalf() const {
        return (nativeHalfActive) || usePosHalf || useVelHalf || usePosPredHalf ||
               useLambdaHalf || useDensityHalf || useAuxHalf || useRenderHalf;
    }
    bool hasPrevSnapshot() const { return d_prev_pos_h4 != nullptr; }

    void ensurePrevSnapshot() {
        if (d_prev_pos_h4) { cudaFree(d_prev_pos_h4); d_prev_pos_h4 = nullptr; }
        if (capacity > 0) CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_prev_pos_h4, sizeof(Half4) * capacity));
    }
    void freePrevPosSnapshot() { if (d_prev_pos_h4) { cudaFree(d_prev_pos_h4); d_prev_pos_h4 = nullptr; } }

    // ===== Debug 验证 =====
    void debugValidate(const char* tag) const {
        auto chk = [&](const void* p, const char* name){
            if(!p){
                std::fprintf(stderr,"[BufChk][%s] %s=NULL\n", tag, name);
            }
        };
        chk(d_pos_curr,"pos_curr");
        chk(d_pos_next,"pos_next");
        if(usePosHalf)      chk(d_pos_h4,"pos_h4");
        if(usePosPredHalf)  chk(d_pos_pred_h4,"pos_pred_h4");
        if(useVelHalf)      chk(d_vel_h4,"vel_h4");
    }

    void sanityPrint(const char* tag) const {
        /*
        std::fprintf(stderr,
            "[BufLayout][%s] cap=%u pos_block=%p curr=%p next=%p posHalfBlock=%p pos_h4=%p pos_pred_h4=%p vel=%p vel_h4=%p\n",
            tag, capacity,
            (void*)d_pos_block, (void*)d_pos_curr, (void*)d_pos_next,
            (void*)d_pos_half_block, (void*)d_pos_h4, (void*)d_pos_pred_h4,
            (void*)d_vel_curr, (void*)d_vel_h4);
            */
    }

    // ===== Pack/Unpack mirrors（加强保护） =====
    void packAllToHalf(uint32_t N, cudaStream_t s) {
        if (nativeHalfActive || !anyHalf() || N ==0) return;
        if (N > capacity) {
            std::fprintf(stderr,"[Pack][Warn] N(%u)>capacity(%u) skip\n", N, capacity);
            return;
        }
        const auto& cc = console::Instance();
        if (cc.debug.printDiagnostics) {
            std::fprintf(stderr,"[PredHalf.Pack] N=%u curr=%p next=%p pred_h4=%p pos_h4=%p vel_h4=%p usePos=%d usePred=%d useVel=%d native=%d\n",
                N,(void*)d_pos_curr,(void*)d_pos_next,(void*)d_pos_pred_h4,(void*)d_pos_h4,(void*)d_vel_h4,
                (int)usePosHalf,(int)usePosPredHalf,(int)useVelHalf,(int)nativeHalfActive);
        }
        struct VItem { const float4* src; Half4* dst; bool enabled; const char* name; };
        VItem vitems[] = {
            { d_pos_curr, d_pos_h4,       usePosHalf     && d_pos_curr     && d_pos_h4,       "pos_curr" },
            { d_vel_curr, d_vel_h4,       useVelHalf     && d_vel_curr     && d_vel_h4,       "vel_curr" },
            { d_pos_next, d_pos_pred_h4,  usePosPredHalf && d_pos_next     && d_pos_pred_h4,  "pos_next(pred)" }
        };
        for (auto& it : vitems) {
            if (!it.enabled) continue;
            PackFloat4ToHalf4(it.src, it.dst, N, s);
        }

        struct SItem { const float* src; __half* dst; bool enabled; const char* name; };
        SItem sitems[] = {
            { d_lambda,  d_lambda_h,  useLambdaHalf  && d_lambda  && d_lambda_h,  "lambda" },
            { d_density, d_density_h, useDensityHalf && d_density && d_density_h, "density" },
            { d_aux,     d_aux_h,     useAuxHalf     && d_aux     && d_aux_h,     "aux" }
        };
        for (auto& it : sitems) {
            if (!it.enabled) continue;
            PackFloatToHalf(it.src, it.dst, N, s);
        }

        if (useRenderHalf && d_render_pos_h4 && d_pos_curr)
            PackFloat4ToHalf4(d_pos_curr, d_render_pos_h4, N, s);
    }

    void unpackAllFromHalf(uint32_t N, cudaStream_t s) {
        if (nativeHalfActive || !anyHalf() || N == 0) return;
        if (N > capacity) {
            std::fprintf(stderr,"[Unpack][Warn] N(%u)>capacity(%u) skip\n", N, capacity);
            return;
        }
        struct VItem { const Half4* src; float4* dst; bool enabled; };
        VItem vitems[] = {
            { d_pos_h4,      d_pos_curr, usePosHalf     && d_pos_curr    && d_pos_h4 },
            { d_vel_h4,      d_vel_curr, useVelHalf     && d_vel_curr    && d_vel_h4 },
            { d_pos_pred_h4, d_pos_next, usePosPredHalf && d_pos_next    && d_pos_pred_h4 }
        };
        for (auto& it : vitems) if (it.enabled) UnpackHalf4ToFloat4(it.src, it.dst, N, s);

        struct SItem { const __half* src; float* dst; bool enabled; };
        SItem sitems[] = {
            { d_lambda_h,  d_lambda,  useLambdaHalf  && d_lambda  && d_lambda_h },
            { d_density_h, d_density, useDensityHalf && d_density && d_density_h },
            { d_aux_h,     d_aux,     useAuxHalf     && d_aux     && d_aux_h }
        };
        for (auto& it : sitems) if (it.enabled) UnpackHalfToFloat(it.src, it.dst, N, s);
    }

    void packRenderToHalf(uint32_t N, cudaStream_t s) {
        if (nativeHalfActive) return;
        if (!useRenderHalf || !d_render_pos_h4 || !d_pos_curr || N == 0 || N > capacity) return;
        PackFloat4ToHalf4(d_pos_curr, d_render_pos_h4, N, s);
    }

    // ===== 公共接口 =====
    void allocate(uint32_t cap) { allocateInternal(cap, false,false,false,false,false,false); ensurePrevSnapshot(); BindDeviceGlobalsFrom(*this); }
    void allocateWithPrecision(const sim::SimPrecision& prec, uint32_t cap) {
        if (prec.nativeHalfActive) { allocateNativeHalfPrimary(prec, cap); return; }
        bool posH = (prec.positionStore == sim::NumericType::FP16_Packed || prec.positionStore == sim::NumericType::FP16);
        bool velH = (prec.velocityStore == sim::NumericType::FP16_Packed || prec.velocityStore == sim::NumericType::FP16);
        bool predH= (prec.predictedPosStore == sim::NumericType::FP16_Packed || prec.predictedPosStore == sim::NumericType::FP16);
        bool lambdaH  = (prec.lambdaStore  == sim::NumericType::FP16);
        bool densityH = (prec.densityStore == sim::NumericType::FP16);
        bool auxH     = (prec.auxStore     == sim::NumericType::FP16_Packed || prec.auxStore == sim::NumericType::FP16);
        bool renderH  = ((prec.renderTransfer == sim::NumericType::FP16_Packed || prec.renderTransfer == sim::NumericType::FP16) && !posH);
        allocateInternal(cap, posH, velH, predH, lambdaH, densityH, auxH);
        useRenderHalf = renderH;
        if (useRenderHalf) CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_render_pos_h4, sizeof(Half4)*capacity));
        ensurePrevSnapshot();
        BindDeviceGlobalsFrom(*this);
        debugValidate("allocateWithPrecision");
        sanityPrint("allocateWithPrecision");
    }

    void allocateNativeHalfPrimary(const sim::SimPrecision& prec, uint32_t cap) {
        release();
        if (cap == 0) cap = 1;
        capacity = cap;
        nativeHalfActive = true;
        ownsPosBuffers = true;
        externalPingPong = false;
        posPredExternal = false;

        auto allocH4 = [&](Half4** p){ CUDA_CHECK_ALLOC(cudaMalloc((void**)p, sizeof(Half4)*capacity)); };
        allocH4(&d_pos_curr_native);
        allocH4(&d_pos_next_native);
        allocH4(&d_vel_curr_native);

        d_pos_h4       = d_pos_curr_native;
        d_pos_pred_h4  = d_pos_next_native;
        d_vel_h4       = d_vel_curr_native;

        d_pos_curr = d_pos_next = nullptr;
        d_pos = d_pos_curr;
        d_pos_pred = d_pos_next;

        CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_vel, sizeof(float4)*capacity));
        d_vel_curr = d_vel;
        CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_lambda, sizeof(float)*capacity));
        CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_delta,  sizeof(float4)*capacity));
        CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_density,sizeof(float)*capacity));
        CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_aux,    sizeof(float)*capacity));

        usePosHalf = useVelHalf = usePosPredHalf = true;
        useLambdaHalf = useDensityHalf = useAuxHalf = false;
        useRenderHalf = (prec.renderTransfer == sim::NumericType::FP16_Packed || prec.renderTransfer == sim::NumericType::FP16);
        if (useRenderHalf) CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_render_pos_h4, sizeof(Half4)*capacity));

        ensurePrevSnapshot();
        BindDeviceGlobalsFrom(*this);
        sanityPrint("allocateNativeHalfPrimary");
    }

    void fallbackToFp32(uint32_t cap) {
        if (!nativeHalfActive) return;
        auto fre = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fre(d_pos_curr_native);
        fre(d_pos_next_native);
        fre(d_vel_curr_native);
        nativeHalfActive = false;
        // 清除别名，但不 double-free
        d_pos_h4 = nullptr;
        d_pos_pred_h4 = nullptr;
        allocateInternal(cap == 0 ? capacity : cap, false, false, false, false, false, false);
    }

    void allocateExternalHalfMirrors(uint32_t cap) {
        if (!externalPingPong) return;
        if (cap == 0) cap = capacity;
        if (cap == 0) return;
        if (d_pos_half_block) return; // 已分配
        size_t bytesPosHalf = sizeof(Half4) * size_t(cap) * 2;
        CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_pos_half_block, bytesPosHalf));
        d_pos_h4 = d_pos_half_block;
        d_pos_pred_h4 = d_pos_half_block + cap;
        usePosHalf = usePosPredHalf = true;
        std::fprintf(stderr, "[ExternalHalf][Alloc] half_block=%p curr_h4=%p next_h4=%p cap=%u\n",
            (void*)d_pos_half_block, (void*)d_pos_h4, (void*)d_pos_pred_h4, cap);
    }

    // 修复：支持半精镜像的外部双缓冲绑定
    void bindExternalPosPingPong(float4* ptrA, float4* ptrB, uint32_t cap, bool wantHalfMirror) {
        checkGuards("bindExternalPingPong.before");
        if (nativeHalfActive) {
            std::fprintf(stderr, "[NativeHalf][Warn] external ping-pong not supported, fallback FP32\n");
            fallbackToFp32(cap);
        }

        // 释放内部自有连续块（不要 cudaFree d_pos_curr/d_pos_next 这些切片）
        if (ownsPosBuffers) {
            if (d_pos_block) { cudaFree(d_pos_block); d_pos_block = nullptr; }
            if (d_pos_half_block) {
                // 位置 half 块是连续块，别名指针不再单独释放
                cudaFree(d_pos_half_block);
                d_pos_half_block = nullptr;
                d_pos_h4 = nullptr;
                d_pos_pred_h4 = nullptr;
            }
            ownsPosBuffers = false;
        }

        capacity = cap;
        d_pos_curr = ptrA;
        d_pos_next = ptrB;
        d_pos = d_pos_curr;
        d_pos_pred = d_pos_next;
        posPredExternal = false;
        externalPingPong = true;

        if (wantHalfMirror) {
            // 为外部位置建立半精镜像连续块
            if (!d_pos_half_block) {
                size_t bytesPosHalf = sizeof(Half4) * size_t(capacity) * 2;
                if (cudaMalloc((void**)&d_pos_half_block, bytesPosHalf) == cudaSuccess) {
                    d_pos_h4 = d_pos_half_block;
                    d_pos_pred_h4 = d_pos_half_block + capacity;
                    usePosHalf = usePosPredHalf = true;
                    std::fprintf(stderr,
                        "[ExternalHalf][Alloc] half_block=%p curr_h4=%p next_h4=%p cap=%u\n",
                        (void*)d_pos_half_block, (void*)d_pos_h4, (void*)d_pos_pred_h4, capacity);
                }
                else {
                    std::fprintf(stderr, "[ExternalHalf][Error] cudaMalloc failed, disable half mirrors.\n");
                    d_pos_half_block = nullptr;
                    d_pos_h4 = d_pos_pred_h4 = nullptr;
                    usePosHalf = usePosPredHalf = false;
                }
            }
        }
        else {
            d_pos_h4 = d_pos_pred_h4 = nullptr;
            usePosHalf = usePosPredHalf = false;
        }

        BindDeviceGlobalsFrom(*this);
        checkGuards("bindExternalPingPong.after");
        sanityPrint("bindExternalPosPingPong");
    }

    void swapPositionPingPong() {
        checkGuards("swapPositionPingPong.before");

        // 外部双缓冲：需要同时交换 half 镜像（若存在）
        if (externalPingPong) {
            float4* oldCurr = d_pos_curr;
            float4* oldNext = d_pos_next;
            std::swap(d_pos_curr, d_pos_next);
            d_pos = d_pos_curr;
            d_pos_pred = d_pos_next;

            // Half 镜像保持语义：d_pos_h4 = 当前 curr 的 half；d_pos_pred_h4 = 当前 next 的 half
            // 由于 half 块是 [curr_half | next_half] 固定布局，这里需要“指针交换”才能保持一致
            if ((usePosHalf || usePosPredHalf) && d_pos_h4 && d_pos_pred_h4) {
                std::swap(d_pos_h4, d_pos_pred_h4);
            }

            BindDeviceGlobalsFrom(*this);
            checkGuards("swapPositionPingPong.after.external");
            sanityPrint("swapPositionPingPong.external");
            return;
        }

        // 原生 half 主存储：也要交换 half 别名
        if (nativeHalfActive) {
            sim::Half4* oldCurrH = d_pos_curr_native;
            sim::Half4* oldNextH = d_pos_next_native;
            std::swap(d_pos_curr_native, d_pos_next_native);
            // 同步别名
            std::swap(d_pos_h4, d_pos_pred_h4);
            // FP32 pos 不存在（或无效），但保持 d_pos/d_pos_pred 语义为 nullptr 或可选逻辑
            BindDeviceGlobalsFrom(*this);
            checkGuards("swapPositionPingPong.after.nativeHalf");
            sanityPrint("swapPositionPingPong.nativeHalf");
            return;
        }

        // 内部双缓冲（标准块）
        if (posPredExternal) {
            // 外部预测别名模式：不改变 d_pos_pred，但仍需交换 curr/next 以及 half 匹配关系
            std::swap(d_pos_curr, d_pos_next);
            d_pos = d_pos_curr;
            // predicted 仍指向外部 alias，不 swap
            if ((usePosHalf || usePosPredHalf) && d_pos_h4 && d_pos_pred_h4) {
                std::swap(d_pos_h4, d_pos_pred_h4);
            }
            std::fprintf(stderr, "[SwapPP][SkipPred] curr=%p next=%p pred(ext)=%p cap=%u\n",
                (void*)d_pos_curr, (void*)d_pos_next, (void*)d_pos_pred, capacity);
        }
        else {
            std::swap(d_pos_curr, d_pos_next);
            d_pos = d_pos_curr;
            d_pos_pred = d_pos_next;
            if ((usePosHalf || usePosPredHalf) && d_pos_h4 && d_pos_pred_h4) {
                std::swap(d_pos_h4, d_pos_pred_h4);
            }
            std::fprintf(stderr, "[SwapPP][Internal] curr=%p next=%p pred=%p cap=%u\n",
                (void*)d_pos_curr, (void*)d_pos_next, (void*)d_pos_pred, capacity);
        }
        BindDeviceGlobalsFrom(*this);
        checkGuards("swapPositionPingPong.after");
    }

    void bindExternalPosPred(float4* ptr) {
        checkGuards("bindExternalPred.before");

        // 原生 half 模式不支持单独外部预测
        if (nativeHalfActive) {
            std::fprintf(stderr, "[ExternalPred][Warn] nativeHalfActive -> ignore external predicted\n");
            return;
        }

        // 如果当前采用统一连续块 (d_pos_block != nullptr)，说明 d_pos_pred 是内部切片，不能 cudaFree
        if (d_pos_block) {
            // 仅建立别名，内部 next 仍用于写；外部 ptr 只读用途（如渲染或调试）
            d_pos_pred = ptr;
            posPredExternal = true;
            externalPosPredAlias = true;
            externalPingPong = false;
            std::fprintf(stderr,
                "[ExternalPred][Alias] ptr=%p (keep internal next for simulation) block=%p cap=%u\n",
                (void*)ptr, (void*)d_pos_block, capacity);
        }
        else {
            // 旧模式（分离分配）才允许释放旧 predicted 缓冲
            if (d_pos_pred && !posPredExternal && ownsPosBuffers) {
                cudaError_t e = cudaFree(d_pos_pred);
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "[ExternalPred][Warn] cudaFree(oldPred=%p) %s (%d)\n",
                        (void*)d_pos_pred, cudaGetErrorString(e), (int)e);
                }
            }
            d_pos_pred = ptr;
            posPredExternal = true;
            externalPosPredAlias = false;
            externalPingPong = false;
            std::fprintf(stderr, "[ExternalPred][Bind] ptr=%p (legacy separate predicted)\n", (void*)ptr);
        }

        BindDeviceGlobalsFrom(*this);
        checkGuards("bindExternalPred.after");
    }

    void detachExternalPosPred() {
        checkGuards("detachExternalPred.before");
        if (nativeHalfActive) return;
        if (externalPosPredAlias) {
            // 只是别名，直接置空
            d_pos_pred = d_pos_next; // 恢复内部 predicted (next)
        }
        else {
            // 旧模式下 d_pos_pred 指向独立 cudaMalloc，保持原逻辑（这里只是置空，不释放）
            // 若需要真正释放可在 release() 统一处理
        }
        posPredExternal = false;
        externalPosPredAlias = false;
        externalPingPong = false;
        BindDeviceGlobalsFrom(*this);
        checkGuards("detachExternalPred.after");
    }

    void checkGuards(const char* tag) const {
        if (guardA!=0xA11CE5u || guardB!=0xBEEFBEEFu){
            std::fprintf(stderr,"[GuardCorrupt][%s] A=%08X B=%08X\n",tag,guardA,guardB);
        }
    }

    void release();
    ~DeviceBuffers(){ release(); }
private:
    void allocateInternal(uint32_t cap,
                          bool posH,bool velH,bool predH,bool lambdaH,bool densityH,bool auxH){
        checkGuards("allocateInternal.before");
        if (nativeHalfActive){
            std::fprintf(stderr,"[AllocateInternal][Warn] native active\n");
            return;
        }
        if (cap == 0) cap = 1;
        release();
        capacity = cap;
        usePosHalf      = posH;
        useVelHalf      = velH;
        usePosPredHalf  = predH;
        useLambdaHalf   = lambdaH;
        useDensityHalf  = densityH;
        useAuxHalf      = auxH;
        ownsPosBuffers  = true;
        externalPingPong= false;
        posPredExternal = false;
        nativeHalfActive= false;

        // ===== NEW: 单块双缓冲分配 (位置) =====
        size_t bytesPos = sizeof(float4) * size_t(capacity) * 2;
        CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_pos_block, bytesPos));
        d_pos_curr = d_pos_block;
        d_pos_next = d_pos_block + capacity;
        d_pos      = d_pos_curr;
        d_pos_pred = d_pos_next;

        // 速度 + prev
        CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_vel_curr, sizeof(float4)*capacity));
        d_vel = d_vel_curr;
        CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_vel_prev, sizeof(float4)*capacity));

        // 标量 & delta
        CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_lambda,  sizeof(float)*capacity));
        CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_delta,   sizeof(float4)*capacity));
        CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_density, sizeof(float)*capacity));
        CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_aux,     sizeof(float)*capacity));

        // Half 单块（位置双缓冲）
        if (usePosHalf || usePosPredHalf) {
            size_t bytesPosHalf = sizeof(Half4) * size_t(capacity) * 2;
            CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_pos_half_block, bytesPosHalf));
            d_pos_h4      = d_pos_half_block;
            d_pos_pred_h4 = d_pos_half_block + capacity;
        }

        if (useVelHalf)      CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_vel_h4,      sizeof(Half4)*capacity));
        if (useLambdaHalf)   CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_lambda_h,    sizeof(__half)*capacity));
        if (useDensityHalf)  CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_density_h,   sizeof(__half)*capacity));
        if (useAuxHalf)      CUDA_CHECK_ALLOC(cudaMalloc((void**)&d_aux_h,       sizeof(__half)*capacity));

        std::fprintf(stderr,"[AllocateInternal] cap=%u pos_block=%p curr=%p next=%p velPrev=%p usePredHalf=%d native=%d\n",
                     capacity,(void*)d_pos_block,(void*)d_pos_curr,(void*)d_pos_next,(void*)d_vel_prev,(int)usePosPredHalf,(int)nativeHalfActive);
        checkGuards("allocateInternal.after");
    }
};

inline void DeviceBuffers::release() {
    checkGuards("release.before");
    auto fre = [](auto*& p) {
        if (p) {
            cudaError_t e = cudaFree(p);
            if (e != cudaSuccess) {
                std::fprintf(stderr, "[Release][Warn] cudaFree(%p) %s (%d)\n",
                    (void*)p, cudaGetErrorString(e), (int)e);
            }
            p = nullptr;
        }
        };

    if (nativeHalfActive) {
        fre(d_pos_curr_native);
        fre(d_pos_next_native);
        fre(d_vel_curr_native);
    }

    // 内部双缓冲块
    fre(d_pos_block);
    fre(d_pos_half_block);

    // 指针还原，外部别名不释放
    d_pos_curr = d_pos_next = nullptr;
    d_pos = nullptr;
    if (!externalPosPredAlias) {
        d_pos_pred = nullptr;
    }
    else {
        // 外部别名模式：外部 ptr 由外部管理，不做释放
        d_pos_pred = nullptr;
    }

    fre(d_vel_curr);
    fre(d_vel_prev);
    fre(d_lambda);
    fre(d_delta);
    fre(d_density);
    fre(d_aux);

    // Half mirrors （位置 half 已经通过 half block 释放，避免 double free）
    d_pos_h4 = nullptr;
    d_pos_pred_h4 = nullptr;
    fre(d_vel_h4);
    fre(d_prev_pos_h4);
    fre(d_render_pos_h4);

    fre(d_lambda_h);
    fre(d_density_h);
    fre(d_aux_h);

    capacity = 0;
    posPredExternal = false;
    externalPosPredAlias = false;
    externalPingPong = false;
    ownsPosBuffers = true;
    nativeHalfActive = false;
    usePosHalf = useVelHalf = usePosPredHalf =
        useLambdaHalf = useDensityHalf = useAuxHalf = false;
    useRenderHalf = false;

    checkGuards("release.after");
}

} // namespace sim