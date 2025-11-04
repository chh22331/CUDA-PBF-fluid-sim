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

    // ================= 精简后的设备缓冲结构 =================
    //仅保留双位置缓冲 (curr/next) 与可选 half 镜像；去除所有 "pred" / 单预测 / 原生 half 主存储方案。
    struct DeviceBuffers {
        // ---- 主存储（始终为 FP32） ----
        float4* d_pos_curr = nullptr;   // 当前帧位置 A
        float4* d_pos_next = nullptr;   // 下一帧位置 B（积分/约束写入目标）
        //兼容旧引用别名（读/写均指向 curr/next）
        //float4* d_pos = nullptr; // alias -> d_pos_curr
        //float4* d_pos_pred = nullptr; // alias -> d_pos_next

        float4* d_vel = nullptr;        //速度
        float4* d_delta = nullptr;      // XSPH 或临时速度
        float*  d_lambda = nullptr;     // PBF λ
        float*  d_density = nullptr;    // 密度（统计/误差）
        float*  d_aux = nullptr;        // 辅助/梯度/累加器

        // ---- Half 镜像（非原生，仅作为带宽/渲染压缩） ----
        Half4* d_pos_curr_h4 = nullptr; // curr 对应 half4 镜像
        Half4* d_pos_next_h4 = nullptr; // next 对应 half4 镜像
        Half4* d_vel_h4 = nullptr;
        Half4* d_delta_h4 = nullptr;
        Half4* d_render_pos_h4 = nullptr; // 渲染发布镜像（可与 curr 相同，也可分离）
        Half4* d_prev_pos_h4 = nullptr; //位置快照（自适应/诊断）
        //兼容旧字段别名
        //Half4* d_pos_h4 = nullptr; // alias -> d_pos_curr_h4
        //Half4* d_pos_pred_h4 = nullptr; // alias -> d_pos_next_h4

        __half* d_lambda_h = nullptr;
        __half* d_density_h = nullptr;
        __half* d_aux_h = nullptr;

        // ---- 状态与配置 ----
        uint32_t capacity =0;
        bool externalPingPong = false; // 是否使用外部双缓冲（零拷贝共享）
        bool usePosHalf = false; // 是否启用位置 half 镜像（对两个 pingpong 都分配）
        bool useVelHalf = false;
        bool useLambdaHalf = false;
        bool useDensityHalf = false;
        bool useAuxHalf = false;
        bool useRenderHalf = false; // 独立渲染 half 打包（若 positionStore 已 half 则可复用 curr_h4）

        uint32_t guardA =0xA11CE5u;
        uint32_t guardB =0xBEEFBEEFu;

        // ---- 查询 & 工具 ----
        bool anyHalf() const { return usePosHalf || useVelHalf || useLambdaHalf || useDensityHalf || useAuxHalf || useRenderHalf; }
        bool hasPrevSnapshot() const { return d_prev_pos_h4 != nullptr; }

        void ensurePrevSnapshot() {
            if (d_prev_pos_h4) { cudaFree(d_prev_pos_h4); d_prev_pos_h4 = nullptr; }
            if (capacity >0) cudaMalloc((void**)&d_prev_pos_h4, sizeof(Half4) * capacity);
        }
        void freePrevPosSnapshot() { if (d_prev_pos_h4) { cudaFree(d_prev_pos_h4); d_prev_pos_h4 = nullptr; } }

        // ---- Pack/Unpack ----
        void packAllToHalf(uint32_t N, cudaStream_t s) {
            if (!anyHalf() || N ==0) return;
            if (usePosHalf && d_pos_curr && d_pos_curr_h4) PackFloat4ToHalf4(d_pos_curr, d_pos_curr_h4, N, s);
            if (usePosHalf && d_pos_next && d_pos_next_h4) PackFloat4ToHalf4(d_pos_next, d_pos_next_h4, N, s);
            if (useVelHalf && d_vel && d_vel_h4) PackFloat4ToHalf4(d_vel, d_vel_h4, N, s);
            if (useVelHalf && d_delta && d_delta_h4) PackFloat4ToHalf4(d_delta, d_delta_h4, N, s);
            if (useLambdaHalf && d_lambda && d_lambda_h) PackFloatToHalf(d_lambda, d_lambda_h, N, s);
            if (useDensityHalf && d_density && d_density_h) PackFloatToHalf(d_density, d_density_h, N, s);
            if (useAuxHalf && d_aux && d_aux_h) PackFloatToHalf(d_aux, d_aux_h, N, s);
            if (useRenderHalf && d_render_pos_h4 && d_pos_curr) PackFloat4ToHalf4(d_pos_curr, d_render_pos_h4, N, s);
        }
        void unpackAllFromHalf(uint32_t N, cudaStream_t s) {
            if (!anyHalf() || N ==0) return;
            if (usePosHalf && d_pos_curr_h4 && d_pos_curr) UnpackHalf4ToFloat4(d_pos_curr_h4, d_pos_curr, N, s);
            if (usePosHalf && d_pos_next_h4 && d_pos_next) UnpackHalf4ToFloat4(d_pos_next_h4, d_pos_next, N, s);
            if (useVelHalf && d_vel_h4 && d_vel) UnpackHalf4ToFloat4(d_vel_h4, d_vel, N, s);
            if (useVelHalf && d_delta_h4 && d_delta) UnpackHalf4ToFloat4(d_delta_h4, d_delta, N, s);
            if (useLambdaHalf && d_lambda_h && d_lambda) UnpackHalfToFloat(d_lambda_h, d_lambda, N, s);
            if (useDensityHalf && d_density_h && d_density) UnpackHalfToFloat(d_density_h, d_density, N, s);
            if (useAuxHalf && d_aux_h && d_aux) UnpackHalfToFloat(d_aux_h, d_aux, N, s);
        }
        void packRenderToHalf(uint32_t N, cudaStream_t s) {
            if (!useRenderHalf || !d_render_pos_h4 || !d_pos_curr) return;
            PackFloat4ToHalf4(d_pos_curr, d_render_pos_h4, N, s);
        }

        // ---- 分配 API ----
        void allocate(uint32_t cap) {
            allocateInternal(cap, false, false, false, false, false);
            ensurePrevSnapshot();
            BindDeviceGlobalsFrom(*this);
        }
        void allocateWithPrecision(const sim::SimPrecision& prec, uint32_t cap) {
            bool posH = (prec.positionStore == sim::NumericType::FP16_Packed || prec.positionStore == sim::NumericType::FP16);
            bool velH = (prec.velocityStore == sim::NumericType::FP16_Packed || prec.velocityStore == sim::NumericType::FP16);
            bool lambdaH = (prec.lambdaStore == sim::NumericType::FP16 || prec.lambdaStore == sim::NumericType::FP16_Packed);
            bool densityH = (prec.densityStore == sim::NumericType::FP16 || prec.densityStore == sim::NumericType::FP16_Packed);
            bool auxH = (prec.auxStore == sim::NumericType::FP16 || prec.auxStore == sim::NumericType::FP16_Packed);
            bool renderH = ((prec.renderTransfer == sim::NumericType::FP16 || prec.renderTransfer == sim::NumericType::FP16_Packed) && !posH);
            allocateInternal(cap, posH, velH, lambdaH, densityH, auxH);
            useRenderHalf = renderH;
            if (useRenderHalf) cudaMalloc((void**)&d_render_pos_h4, sizeof(Half4) * capacity);
            ensurePrevSnapshot();
            BindDeviceGlobalsFrom(*this);
        }
 
        // ---- Ping-pong交换 ----
        void swapPositionPingPong() {
            checkGuards("swapPositionPingPong.before");
            std::swap(d_pos_curr, d_pos_next);
            BindDeviceGlobalsFrom(*this);
            checkGuards("swapPositionPingPong.after");
        }

        void checkGuards(const char* tag) const {
            if (guardA !=0xA11CE5u || guardB !=0xBEEFBEEFu) {
                std::fprintf(stderr,
                    "[GuardCorrupt][%s] guardA=%08X guardB=%08X externalPingPong=%d capacity=%u\n",
                    tag, guardA, guardB, (int)externalPingPong, capacity);
            }
        }

        void release();
        ~DeviceBuffers() { release(); }
    public:
        inline float4* posCurr() const { return d_pos_curr; }
        inline float4* posNext() const { return d_pos_next; }
        inline Half4* posCurrHalf() const { return d_pos_curr_h4; }
        inline Half4* posNextHalf() const { return d_pos_next_h4; }
        inline bool    isExternalPingPong() const { return externalPingPong; }
        inline uint32_t posCapacity() const { return capacity; }

        // 采用外部双缓冲（统一入口）
        void adoptExternalPingPong(float4* a, float4* b, uint32_t cap) {
            checkGuards("adoptExternalPingPong.before");
            // 释放内部位置（仅非外部模式时）
            if (d_pos_curr && !externalPingPong) cudaFree(d_pos_curr);
            if (d_pos_next && !externalPingPong) cudaFree(d_pos_next);
            d_pos_curr = a;
            d_pos_next = b;
            capacity = cap;
            externalPingPong = true;
            // half 镜像不再自动分配，需外部重新配置
            d_pos_curr_h4 = d_pos_next_h4 = nullptr;
            BindDeviceGlobalsFrom(*this);
            checkGuards("adoptExternalPingPong.after");
        }

        // 统一的 swap（内部 / 外部均可）
        void swapPingPongPositions() {
            checkGuards("swapPingPongPositions.before");
            std::swap(d_pos_curr, d_pos_next);
            // 若存在 half 镜像，也同步交换
            std::swap(d_pos_curr_h4, d_pos_next_h4);
            BindDeviceGlobalsFrom(*this);
            checkGuards("swapPingPongPositions.after");
        }
 
    private:
        void allocateInternal(uint32_t cap, bool posH, bool velH, bool lambdaH, bool densityH, bool auxH) {
            checkGuards("allocateInternal.before");
            std::fprintf(stderr,
                "[Allocate] cap=%u posH=%d velH=%d lambdaH=%d densityH=%d auxH=%d (prevCap=%u)\n",
                cap, (int)posH, (int)velH, (int)lambdaH, (int)densityH, (int)auxH, capacity);
            if (cap ==0) cap =1;
            if (cap == capacity && posH == usePosHalf && velH == useVelHalf && lambdaH == useLambdaHalf && densityH == useDensityHalf && auxH == useAuxHalf)
                return; // 配置未变化
            release();
            capacity = cap;
            externalPingPong = false;
            usePosHalf = posH; useVelHalf = velH; useLambdaHalf = lambdaH; useDensityHalf = densityH; useAuxHalf = auxH; useRenderHalf = false;
            auto alloc = [&](void** p, size_t elemSize) { cudaMalloc(p, elemSize * capacity); };
            alloc((void**)&d_pos_curr, sizeof(float4));
            alloc((void**)&d_pos_next, sizeof(float4));
            alloc((void**)&d_vel, sizeof(float4));
            alloc((void**)&d_delta, sizeof(float4));
            alloc((void**)&d_lambda, sizeof(float));
            alloc((void**)&d_density, sizeof(float));
            alloc((void**)&d_aux, sizeof(float));
            if (usePosHalf) { cudaMalloc((void**)&d_pos_curr_h4, sizeof(Half4) * capacity); cudaMalloc((void**)&d_pos_next_h4, sizeof(Half4) * capacity); }
            if (useVelHalf) { cudaMalloc((void**)&d_vel_h4, sizeof(Half4) * capacity); cudaMalloc((void**)&d_delta_h4, sizeof(Half4) * capacity); }
            if (useLambdaHalf) cudaMalloc((void**)&d_lambda_h, sizeof(__half) * capacity);
            if (useDensityHalf) cudaMalloc((void**)&d_density_h, sizeof(__half) * capacity);
            if (useAuxHalf) cudaMalloc((void**)&d_aux_h, sizeof(__half) * capacity);
            std::fprintf(stderr,
                "[Allocate][Done] cap=%u d_pos_curr=%p d_pos_next=%p externalPingPong=0\n",
                capacity, (void*)d_pos_curr, (void*)d_pos_next);
            checkGuards("allocateInternal.after");
        }
    };

    inline void DeviceBuffers::release() {
        checkGuards("release.before");
        auto fre = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        // 主缓冲
        fre(d_pos_curr); fre(d_pos_next); fre(d_vel); fre(d_delta); fre(d_lambda); fre(d_density); fre(d_aux);
        // half 镜像
        fre(d_pos_curr_h4); fre(d_pos_next_h4); fre(d_vel_h4); fre(d_delta_h4); fre(d_prev_pos_h4);
        fre(d_lambda_h); fre(d_density_h); fre(d_aux_h); fre(d_render_pos_h4);
        capacity =0; externalPingPong = false;
        usePosHalf = useVelHalf = useLambdaHalf = useDensityHalf = useAuxHalf = useRenderHalf = false;
        checkGuards("release.after");
    }

} // namespace sim