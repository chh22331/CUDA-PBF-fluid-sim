#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "parameters.h"
#include "numeric_utils.h"
#include "logging.h"
#include "device_globals.cuh"

namespace sim {

    struct DeviceBuffers {
        // ===== FP32 主缓冲 =====
        float4* d_pos = nullptr;       // alias of d_pos_curr
        float4* d_vel = nullptr;       // alias of d_vel_curr
        float4* d_pos_pred = nullptr;  // alias of d_pos_next
        float*  d_lambda = nullptr;
        float4* d_delta = nullptr;
        float*  d_density = nullptr;
        float*  d_aux = nullptr;

        // Ping-pong FP32
        float4* d_pos_curr = nullptr;
        float4* d_pos_next = nullptr;
        float4* d_vel_curr = nullptr;
        float4* d_vel_prev = nullptr;

        // 运行期容量与外部绑定
        uint32_t capacity = 0;
        bool posPredExternal = false;   // 单外部预测镜像
        bool externalPingPong = false;  // 双外部 ping-pong
        bool ownsPosBuffers = true;

        // 统计/诊断缓冲
        float*    d_rho = nullptr;
        float*    d_constraint = nullptr;
        float*    d_sumGrad2 = nullptr;
        uint32_t* d_neighbors = nullptr;

        uint32_t guardA = 0xA11CE5u;
        uint32_t guardB = 0xBEEFBEEFu;

        void allocateRho(uint32_t cap) {
            if (d_rho) cudaFree(d_rho);
            cudaMalloc(&d_rho, sizeof(float) * cap);
        }
        void releaseRho() {
            if (d_rho) { cudaFree(d_rho); d_rho = nullptr; }
        }

        void allocate(uint32_t cap) {
            checkGuards("allocate.before");
            if (cap == 0) cap = 1;
            if (capacity == cap) return;
            release();
            capacity = cap;
            ownsPosBuffers = true; externalPingPong = false; posPredExternal = false;

            auto alloc = [&](void** p, size_t elemSize) { cudaMalloc(p, elemSize * capacity); };
            alloc((void**)&d_pos_curr, sizeof(float4));
            alloc((void**)&d_pos_next, sizeof(float4));
            d_pos = d_pos_curr;
            d_pos_pred = d_pos_next;
            alloc((void**)&d_vel_curr, sizeof(float4)); d_vel = d_vel_curr;
            alloc((void**)&d_lambda, sizeof(float));
            alloc((void**)&d_delta, sizeof(float4));
            alloc((void**)&d_density, sizeof(float));
            alloc((void**)&d_aux, sizeof(float));

            BindDeviceGlobalsFrom(*this);
            checkGuards("allocate.after");
        }

        void bindExternalPosPingPong(float4* ptrA, float4* ptrB, uint32_t cap) {
            checkGuards("bindExternalPingPong.before");
            if (ownsPosBuffers) {
                if (d_pos_curr) cudaFree(d_pos_curr);
                if (d_pos_next) cudaFree(d_pos_next);
                d_pos_curr = d_pos_next = nullptr;
                d_pos = d_pos_curr; d_pos_pred = d_pos_next;
                ownsPosBuffers = false;
            }
            capacity = cap;
            d_pos_curr = ptrA;
            d_pos_next = ptrB;
            d_pos = d_pos_curr;
            d_pos_pred = d_pos_next;
            posPredExternal = false;
            externalPingPong = true;
            BindDeviceGlobalsFrom(*this);
            checkGuards("bindExternalPingPong.after");
        }

        void swapPositionPingPong() {
            checkGuards("swapPositionPingPong.before");
            if (externalPingPong) {
                std::swap(d_pos_curr, d_pos_next);
                d_pos = d_pos_curr;
                d_pos_pred = d_pos_next;
                BindDeviceGlobalsFrom(*this);
                checkGuards("swapPositionPingPong.after");
                return;
            }
            std::swap(d_pos_curr, d_pos_next);
            d_pos = d_pos_curr;
            d_pos_pred = d_pos_next;
            BindDeviceGlobalsFrom(*this);
            checkGuards("swapPositionPingPong.after.internal");
        }

        void bindExternalPosPred(float4* ptr) {
            checkGuards("bindExternalPred.before");
            if (d_pos_pred && !posPredExternal && ownsPosBuffers) cudaFree(d_pos_pred);
            d_pos_pred = ptr;
            posPredExternal = true;
            externalPingPong = false;
            BindDeviceGlobalsFrom(*this);
            checkGuards("bindExternalPred.after");
        }
        void detachExternalPosPred() {
            checkGuards("detachExternalPred.before");
            d_pos_pred = nullptr; posPredExternal = false; externalPingPong = false; BindDeviceGlobalsFrom(*this);
            checkGuards("detachExternalPred.after");
        }

        void releaseConstraintData() {
            if (d_constraint) { cudaFree(d_constraint); d_constraint = nullptr; }
            if (d_sumGrad2)   { cudaFree(d_sumGrad2);   d_sumGrad2 = nullptr; }
            if (d_neighbors)  { cudaFree(d_neighbors);  d_neighbors = nullptr; }
        }
        void allocateConstraintData(uint32_t cap) {
            if (d_constraint) cudaFree(d_constraint);
            if (d_sumGrad2)   cudaFree(d_sumGrad2);
            if (d_neighbors)  cudaFree(d_neighbors);
            cudaMalloc(&d_constraint, sizeof(float) * cap);
            cudaMalloc(&d_sumGrad2, sizeof(float) * cap);
            cudaMalloc(&d_neighbors, sizeof(uint32_t) * cap);
            cudaMemset(d_constraint, 0, sizeof(float) * cap);
            cudaMemset(d_sumGrad2, 0, sizeof(float) * cap);
            cudaMemset(d_neighbors, 0, sizeof(uint32_t) * cap);
        }

        void checkGuards(const char* tag) const {
            if (guardA != 0xA11CE5u || guardB != 0xBEEFBEEFu) {
                std::fprintf(stderr,
                    "[GuardCorrupt][%s] guardA=%08X guardB=%08X posPredExternal=%d externalPingPong=%d capacity=%u\n",
                    tag, guardA, guardB, (int)posPredExternal, (int)externalPingPong, capacity);
            }
        }

        void release() {
            checkGuards("release.before");
            auto fre = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
            if (ownsPosBuffers) { fre(d_pos_curr); fre(d_pos_next); }
            fre(d_vel_curr); fre(d_vel_prev);
            if (ownsPosBuffers) { fre(d_pos); fre(d_pos_pred); } else { d_pos = nullptr; d_pos_pred = nullptr; }
            fre(d_lambda); fre(d_delta); fre(d_density); fre(d_aux);
            fre(d_rho);
            fre(d_constraint); fre(d_sumGrad2); fre(d_neighbors);
            capacity = 0; posPredExternal = false; externalPingPong = false; ownsPosBuffers = true;
            checkGuards("release.after");
        }

        ~DeviceBuffers() { release(); }
    };

} // namespace sim