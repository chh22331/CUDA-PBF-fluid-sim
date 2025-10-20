#include "precision_traits.cuh"
#include <cuda_runtime.h>
#include <cstdio>

namespace sim {

    // 唯一定义（与头文件中的 extern 对应）
    __device__ __constant__ DevicePrecisionView g_precisionView;

    bool UpdateDevicePrecisionView(const DeviceBuffers& bufs, const SimPrecision& pr) {
        DevicePrecisionView h{};
        auto isPackedHalf = [](NumericType t) {
            return t == NumericType::FP16_Packed;
            };

        h.useHalfPos = (isPackedHalf(pr.positionStore) && bufs.d_pos_h4 != nullptr) ? 1 : 0;
        h.useHalfVel = (isPackedHalf(pr.velocityStore) && bufs.d_vel_h4 != nullptr) ? 1 : 0;
        h.useHalfPosPred = (isPackedHalf(pr.predictedPosStore) && bufs.d_pos_pred_h4 != nullptr) ? 1 : 0;
        h.useHalfGeneric = h.useHalfPos;

        h.d_pos_h4 = bufs.d_pos_h4;
        h.d_vel_h4 = bufs.d_vel_h4;
        h.d_pos_pred_h4 = bufs.d_pos_pred_h4;

        cudaError_t e = cudaMemcpyToSymbol(g_precisionView, &h, sizeof(h));
        if (e != cudaSuccess) {
            std::fprintf(stderr, "[PrecisionTraits] cudaMemcpyToSymbol failed (%s)\n", cudaGetErrorString(e));
            return false;
        }
        return true;
    }

} // namespace sim