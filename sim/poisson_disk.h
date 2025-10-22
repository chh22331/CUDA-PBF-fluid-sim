#pragma once
#include <vector>
#include <random>
#include <cuda_runtime.h>

namespace sim {
// 生成半径 R 圆盘内 Poisson-disk 采样点，最小间距 rMin，目标数量 target。
// 结果写入 outPts（float2: 局部平面坐标），若不足 target 则返回已找到的点。
void poisson_disk_in_circle(float R, float rMin, int target,
                            std::mt19937& rng,
                            std::vector<float2>& outPts);
}
