#include "precision_traits.cuh"
#include <cuda_runtime.h>
#include <cstdio>

namespace sim {

    // 唯一定义（与头文件中的 extern 对应）
    __device__ __constant__ DevicePrecisionView g_precisionView;

    bool UpdateDevicePrecisionView(const DeviceBuffers& bufs, const SimPrecision& pr) {
        DevicePrecisionView h{};
        auto isPackedHalf = [](NumericType t) { return t == NumericType::FP16_Packed || t == NumericType::FP16; };

        h.useHalfPos = (isPackedHalf(pr.positionStore) && bufs.d_pos_h4) ? 1 : 0;
        h.useHalfVel = (isPackedHalf(pr.velocityStore) && bufs.d_vel_h4) ? 1 : 0;
        h.useHalfPosPred = (isPackedHalf(pr.predictedPosStore) && bufs.d_pos_pred_h4) ? 1 : 0;
        h.useHalfGeneric = h.useHalfPos;
        h.useHalfLambda = (isPackedHalf(pr.lambdaStore) && bufs.d_lambda_h) ? 1 : 0;
        h.useHalfDensity = (isPackedHalf(pr.densityStore) && bufs.d_density_h) ? 1 : 0;
        h.useHalfAux = (isPackedHalf(pr.auxStore) && bufs.d_aux_h) ? 1 : 0;
        h.useHalfRender = (isPackedHalf(pr.renderTransfer) && bufs.d_render_pos_h4) ? 1 : 0; // 新增

        h.d_pos_h4 = bufs.d_pos_h4;
        h.d_vel_h4 = bufs.d_vel_h4;
        h.d_pos_pred_h4 = bufs.d_pos_pred_h4;
        h.d_lambda_h = bufs.d_lambda_h;
        h.d_density_h = bufs.d_density_h;
        h.d_aux_h = bufs.d_aux_h;
        h.d_render_pos_h4 = bufs.d_render_pos_h4; // 新增

        cudaError_t e = cudaMemcpyToSymbol(g_precisionView, &h, sizeof(h));
        if (e != cudaSuccess) {
            std::fprintf(stderr, "[PrecisionTraits] cudaMemcpyToSymbol failed (%s)\n", cudaGetErrorString(e));
            return false;
        }
        return true;
    }

} // namespace sim