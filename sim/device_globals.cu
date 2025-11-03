#include "device_globals.cuh"
#include "device_buffers.cuh"

namespace sim {

    // 定义设备侧符号指针（初始为 nullptr）
    __device__ float4* g_pos = nullptr;
    __device__ float4* g_vel = nullptr;
    __device__ float4* g_pos_pred = nullptr;
    __device__ float4* g_delta = nullptr;
    __device__ float* g_lambda = nullptr;
    // 新增：幽灵粒子计数
    __device__ uint32_t g_ghostCount = 0;
    // 新增：原生 half 指针设备符号
    __device__ Half4* g_pos_h4 = nullptr;
    __device__ Half4* g_vel_h4 = nullptr;
    __device__ Half4* g_pos_pred_h4 = nullptr;

    __device__ uint32_t g_boundaryClampCountFp32 =0; //统计 FP32 边界钳制事件
    __device__ uint32_t g_boundaryClampCountHalf =0; //统计 Half 边界钳制事件

    // 新增：XSPH 诊断计数定义
    __device__ uint32_t g_xsphNaNCount = 0;
    __device__ uint32_t g_xsphAnomalyCount = 0;

    static void CopyPtrToSymbol(float4* const* hPtr, float4** symbol) {
        // 这里的 symbol 是编译期常量符号地址；cudaMemcpyToSymbol 目的参数必须是符号名，不是运行时值
        // 由于需对不同类型调用，实际分开展示。此函数仅用于统一写法时的概念示例。
    }

    void BindDeviceGlobals(float4* d_pos_curr,
        float4* d_vel_curr,
        float4* d_pos_next,
        float4* d_delta_ptr,
        float* d_lambda_ptr) {
        // 每次执行仅拷贝几个指针 (5 * 8 字节)，极低开销
        cudaMemcpyToSymbol(g_pos,       &d_pos_curr,   sizeof(float4*));
        cudaMemcpyToSymbol(g_vel,       &d_vel_curr,   sizeof(float4*));
        cudaMemcpyToSymbol(g_pos_pred,  &d_pos_next,   sizeof(float4*));
        cudaMemcpyToSymbol(g_delta,     &d_delta_ptr,  sizeof(float4*));
        cudaMemcpyToSymbol(g_lambda,    &d_lambda_ptr, sizeof(float*));
    }

    void BindDeviceGlobalsFrom(const DeviceBuffers& bufs) {
        BindDeviceGlobals(bufs.d_pos_curr,
                          bufs.d_vel_curr,
                          bufs.d_pos_next,
                          bufs.d_delta,
                          bufs.d_lambda);
        if (bufs.nativeHalfActive) {
            // 写入 half 主存储设备符号（用于原生 half kernel直接访问）
            cudaMemcpyToSymbol(g_pos_h4,      &bufs.d_pos_h4,      sizeof(Half4*));
            cudaMemcpyToSymbol(g_vel_h4,      &bufs.d_vel_h4,      sizeof(Half4*));
            cudaMemcpyToSymbol(g_pos_pred_h4, &bufs.d_pos_pred_h4, sizeof(Half4*));
        }
        else {
            //置空避免误用
            Half4* nullH = nullptr;
            cudaMemcpyToSymbol(g_pos_h4,      &nullH, sizeof(Half4*));
            cudaMemcpyToSymbol(g_vel_h4,      &nullH, sizeof(Half4*));
            cudaMemcpyToSymbol(g_pos_pred_h4, &nullH, sizeof(Half4*));
        }
    }

    // 新增：上传幽灵粒子数
    void UploadGhostCount(uint32_t ghostCount) {
        cudaMemcpyToSymbol(g_ghostCount, &ghostCount, sizeof(uint32_t));
    }

    uint32_t ReadAndResetBoundaryClampCounts(uint32_t* halfCountOut){
     uint32_t hFp32=0,hHalf=0; cudaMemcpyFromSymbol(&hFp32,g_boundaryClampCountFp32,sizeof(uint32_t)); cudaMemcpyFromSymbol(&hHalf,g_boundaryClampCountHalf,sizeof(uint32_t));
     uint32_t zero=0; cudaMemcpyToSymbol(g_boundaryClampCountFp32,&zero,sizeof(uint32_t)); cudaMemcpyToSymbol(g_boundaryClampCountHalf,&zero,sizeof(uint32_t));
     if(halfCountOut) *halfCountOut = hHalf; return hFp32;
    }

    uint32_t ReadAndResetXsphDiag(uint32_t* anomalyCountOut) {
        uint32_t nanC = 0, anomC = 0;
        cudaMemcpyFromSymbol(&nanC,  g_xsphNaNCount,     sizeof(uint32_t));
        cudaMemcpyFromSymbol(&anomC, g_xsphAnomalyCount, sizeof(uint32_t));
        uint32_t zero = 0;
        cudaMemcpyToSymbol(g_xsphNaNCount,     &zero, sizeof(uint32_t));
        cudaMemcpyToSymbol(g_xsphAnomalyCount, &zero, sizeof(uint32_t));
        if (anomalyCountOut) *anomalyCountOut = anomC;
        return nanC;
    }

} // namespace sim