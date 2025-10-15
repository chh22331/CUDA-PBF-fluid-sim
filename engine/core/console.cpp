#include "console.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>

namespace {

inline float3 f3_add(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline float3 f3_sub(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline float3 f3_mul(const float3& a, float s)         { return make_float3(a.x * s, a.y * s, a.z * s); }
inline float  f3_dot(const float3& a, const float3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float3 f3_cross(const float3& a, const float3& b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline float  f3_len(const float3& v) { return std::sqrt((((0.0f) > (f3_dot(v, v))) ? (0.0f) : (f3_dot(v, v)))); }
inline float3 f3_norm(const float3& v) {
    float len = f3_len(v);
    if (len <= 1e-20f) return make_float3(0,0,1);
    return f3_mul(v, 1.0f / len);
}
inline float  f3_proj_extent(const float3& halfExt, const float3& axisUnit) {
    return std::fabs(halfExt.x * axisUnit.x) + std::fabs(halfExt.y * axisUnit.y) + std::fabs(halfExt.z * axisUnit.z);
}
inline int3 make_grid_dim(const float3& mins, const float3& maxs, float cell) {
    auto clamp_ge1 = [](int v) { return (((1) > (v)) ? (1) : (v)); };
    float3 size = f3_sub(maxs, mins);
    int dx = clamp_ge1(static_cast<int>(std::ceil(size.x / cell)));
    int dy = clamp_ge1(static_cast<int>(std::ceil(size.y / cell)));
    int dz = clamp_ge1(static_cast<int>(std::ceil(size.z / cell)));
    return make_int3(dx, dy, dz);
}

} // namespace

namespace console {

RuntimeConsole& Instance() {
    static RuntimeConsole g_console;
    return g_console;
}

static bool ShouldAutoMapMixed(const RuntimeConsole::Simulation& s) {
    // 满足条件：开启 useMixedPrecision 且 precision 保持默认 FP32 且允许 autoMap
    const auto& pc = s.precision;
    if (!s.useMixedPrecision) return false;
    if (!pc.autoMapFromLegacyMixedFlag) return false;
    // 用户若已经修改（出现任何非 FP32 或阶段覆盖）则不自动映射
    bool allDefault =
        pc.positionStore == NumericType::FP32 &&
        pc.velocityStore == NumericType::FP32 &&
        pc.predictedPosStore == NumericType::FP32 &&
        pc.lambdaStore == NumericType::FP32 &&
        pc.densityStore == NumericType::FP32 &&
        pc.auxStore == NumericType::FP32 &&
        pc.renderTransfer == NumericType::FP32 &&
        pc.coreCompute == NumericType::FP32 &&
        !pc.useStageOverrides &&
        pc.fp16StageMask == 0;
    return allDefault;
}

void BuildSimParams(const RuntimeConsole& c, sim::SimParams& out) {
    // 原有赋值保持
    out.numParticles = c.sim.numParticles;
    out.maxParticles = c.sim.maxParticles;
    out.dt = c.sim.dt;
    out.cfl = c.sim.cfl;
    out.gravity = c.sim.gravity;
    out.restDensity = c.sim.restDensity;
    out.solverIters = c.sim.solverIters;
    out.maxNeighbors = (c.perf.neighbor_cap > 0) ? c.perf.neighbor_cap : c.sim.maxNeighbors;
    out.useMixedPrecision = c.sim.useMixedPrecision;
    out.sortEveryN = (((1) > (c.sim.sortEveryN)) ? (1) : (c.sim.sortEveryN));
    out.boundaryRestitution = c.sim.boundaryRestitution;
    out.pbf = c.sim.pbf;
    out.xsph_c = c.sim.xsph_c;

    float h = (c.sim.smoothingRadius > 0.f) ? c.sim.smoothingRadius : 0.02f;
    if (c.sim.deriveHFromRadius) {
        const float r = (((1e-8f) > (c.sim.particleRadiusWorld)) ? (1e-8f) : (c.sim.particleRadiusWorld));
        const float h_from_r = c.sim.h_over_r * r;
        if (h_from_r > 0.0f) h = h_from_r;
    }
    out.kernel = sim::MakeKernelCoeffs(h);

    float mass = c.sim.particleMass;
    switch (c.sim.massMode) {
    case RuntimeConsole::Simulation::MassMode::UniformLattice: {
        const float factor = (c.sim.lattice_spacing_factor_h > 0.f) ? c.sim.lattice_spacing_factor_h : 1.0f;
        const float spacing = factor * h;
        mass = c.sim.restDensity * spacing * spacing * spacing;
        break;
    }
    case RuntimeConsole::Simulation::MassMode::SphereByRadius: {
        const float r = (((1e-8f) > (c.sim.particleRadiusWorld)) ? (1e-8f) : (c.sim.particleRadiusWorld));
        const float pi = 3.14159265358979323846f;
        const float vol = c.sim.particleVolumeScale * (4.0f / 3.0f) * pi * r * r * r;
        mass = vol * c.sim.restDensity;
        break;
    }
    case RuntimeConsole::Simulation::MassMode::Explicit:
    default:
        break;
    }
    out.particleMass = mass;

    float cell = c.sim.cellSize;
    if (cell <= 0.0f) {
        float m = (c.perf.grid_cell_size_multiplier > 0.0f) ? c.perf.grid_cell_size_multiplier : 1.0f;
        cell = (((1e-6f) > (m * h)) ? (1e-6f) : (m * h));
    }
    out.grid.mins = c.sim.gridMins;
    out.grid.maxs = c.sim.gridMaxs;
    out.grid.cellSize = cell;
    auto clamp_ge1 = [](int v) { return (((1) > (v)) ? (1) : (v)); };
    float3 size = make_float3(out.grid.maxs.x - out.grid.mins.x,
        out.grid.maxs.y - out.grid.mins.y,
        out.grid.maxs.z - out.grid.mins.z);
    int dx = clamp_ge1((int)std::ceil(size.x / cell));
    int dy = clamp_ge1((int)std::ceil(size.y / cell));
    int dz = clamp_ge1((int)std::ceil(size.z / cell));
    out.grid.dim = make_int3(dx, dy, dz);

    // ========== 新增：混合精度映射 ==========
    out.precision = {}; // 先用默认值
    const auto& src = c.sim.precision;

    if (ShouldAutoMapMixed(c.sim)) {
        // 默认策略：仅位置/速度/预测位置使用 FP16_Packed；核心计算仍 FP32
        out.precision.positionStore = sim::NumericType::FP16_Packed;
        out.precision.velocityStore = sim::NumericType::FP16_Packed;
        out.precision.predictedPosStore = sim::NumericType::FP16_Packed;
        out.precision.lambdaStore = sim::NumericType::FP32;
        out.precision.densityStore = sim::NumericType::FP32;
        out.precision.auxStore = sim::NumericType::FP32;
        out.precision.renderTransfer = sim::NumericType::FP32;
        out.precision.coreCompute = sim::NumericType::FP32;
        out.precision.forceFp32Accumulate = true;
    }
    else {
        // 用户已提供配置：逐字段复制
        auto mapNt = [](console::NumericType t) -> sim::NumericType {
            return static_cast<sim::NumericType>(static_cast<uint8_t>(t));
            };
        out.precision.positionStore = mapNt(src.positionStore);
        out.precision.velocityStore = mapNt(src.velocityStore);
        out.precision.predictedPosStore = mapNt(src.predictedPosStore);
        out.precision.lambdaStore = mapNt(src.lambdaStore);
        out.precision.densityStore = mapNt(src.densityStore);
        out.precision.auxStore = mapNt(src.auxStore);
        out.precision.renderTransfer = mapNt(src.renderTransfer);
        out.precision.coreCompute = mapNt(src.coreCompute);
        out.precision.forceFp32Accumulate = src.forceFp32Accumulate;
        out.precision.enableHalfIntrinsics = src.enableHalfIntrinsics;
        out.precision.useStageOverrides = src.useStageOverrides;
        out.precision.emissionCompute = mapNt(src.emissionCompute);
        out.precision.gridBuildCompute = mapNt(src.gridBuildCompute);
        out.precision.neighborCompute = mapNt(src.neighborCompute);
        out.precision.densityCompute = mapNt(src.densityCompute);
        out.precision.lambdaCompute = mapNt(src.lambdaCompute);
        out.precision.integrateCompute = mapNt(src.integrateCompute);
        out.precision.velocityCompute = mapNt(src.velocityCompute);
        out.precision.boundaryCompute = mapNt(src.boundaryCompute);
        out.precision.xsphCompute = mapNt(src.xsphCompute);
        out.precision.fp16StageMask = src.fp16StageMask;
        out.precision.adaptivePrecision = src.adaptivePrecision;
        out.precision.densityErrorTolerance = src.densityErrorTolerance;
        out.precision.lambdaVarianceTolerance = src.lambdaVarianceTolerance;
        out.precision.adaptCheckEveryN = src.adaptCheckEveryN;
        // 若用户使用 FP16 (非 Packed)，M1 自动升级到 FP16_Packed（仅镜像阶段）
        if (out.precision.positionStore == sim::NumericType::FP16)
            out.precision.positionStore = sim::NumericType::FP16_Packed;
        if (out.precision.velocityStore == sim::NumericType::FP16)
            out.precision.velocityStore = sim::NumericType::FP16_Packed;
        if (out.precision.predictedPosStore == sim::NumericType::FP16)
            out.precision.predictedPosStore = sim::NumericType::FP16_Packed;
    }
    // 现有 out.precision 基础字段已写入，此处补充阶段覆盖与 fp16StageMask 逻辑
        const auto& pc = c.sim.precision;
    auto mapNt = [](console::NumericType t)->sim::NumericType {
        return static_cast<sim::NumericType>(static_cast<uint8_t>(t));
        };

    if (pc.useStageOverrides) {
        out.precision.emissionCompute = mapNt(pc.emissionCompute);
        out.precision.gridBuildCompute = mapNt(pc.gridBuildCompute);
        out.precision.neighborCompute = mapNt(pc.neighborCompute);
        out.precision.densityCompute = mapNt(pc.densityCompute);
        out.precision.lambdaCompute = mapNt(pc.lambdaCompute);
        out.precision.integrateCompute = mapNt(pc.integrateCompute);
        out.precision.velocityCompute = mapNt(pc.velocityCompute);
        out.precision.boundaryCompute = mapNt(pc.boundaryCompute);
        out.precision.xsphCompute = mapNt(pc.xsphCompute);
    }
    else {
        // 统一用 coreCompute，若 fp16StageMask 指定位启用 FP16（Packed 未必保证，仍依赖存储类型）
        sim::NumericType coreT = mapNt(pc.coreCompute);
        auto wantFp16 = [&](uint32_t bit) {
            return (pc.fp16StageMask & bit) != 0;
            };
        auto choose = [&](uint32_t bit)->sim::NumericType {
            return wantFp16(bit) ? sim::NumericType::FP16 : coreT;
            };
        using B = console::ComputeStageBits;
        out.precision.emissionCompute = choose(B::Stage_Emission);
        out.precision.gridBuildCompute = choose(B::Stage_GridBuild);
        out.precision.neighborCompute = choose(B::Stage_NeighborGather);
        out.precision.densityCompute = choose(B::Stage_Density);
        out.precision.lambdaCompute = choose(B::Stage_LambdaSolve);
        out.precision.integrateCompute = choose(B::Stage_Integration);
        out.precision.velocityCompute = choose(B::Stage_VelocityUpdate);
        out.precision.boundaryCompute = choose(B::Stage_Boundary);
        out.precision.xsphCompute = choose(B::Stage_XSPH);
    }
}

void BuildDeviceParams(const RuntimeConsole& c, sim::DeviceParams& out) {
    sim::SimParams sp{};
    BuildSimParams(c, sp);
    out = sim::MakeDeviceParams(sp);
}

void BuildEmitParams(const RuntimeConsole& c, sim::EmitParams& out) {
    out.nozzlePos   = c.sim.nozzlePos;
    out.nozzleDir   = c.sim.nozzleDir;
    out.nozzleRadius= c.sim.nozzleRadius;
    out.nozzleSpeed = c.sim.nozzleSpeed;
    const float eps = (((1e-6f) > (c.sim.recycleYOffset)) ? (1e-6f) : (c.sim.recycleYOffset));
    out.recycleY = c.sim.gridMins.y + eps;
}

void GenerateCubeMixCenters(const RuntimeConsole& c, std::vector<float3>& outCenters) {
    outCenters.clear();

    if (c.sim.demoMode != RuntimeConsole::Simulation::DemoMode::CubeMix)
        return;

    const auto& s = c.sim;

    // 每层的团数（与 PrepareCubeMix 逻辑一致）
    std::vector<uint32_t> layerCounts;
    layerCounts.resize(s.cube_layers, 0);
    {
        uint32_t remaining = s.cube_group_count;
        for (uint32_t L = 0; L < s.cube_layers; ++L) {
            uint32_t toAssign = remaining / (s.cube_layers - L);
            if (toAssign == 0) toAssign = 1;
            layerCounts[L] = toAssign;
            remaining -= toAssign;
        }
    }

    const float h = (((1e-6f) > (s.smoothingRadius)) ? (1e-6f) : (s.smoothingRadius));

    outCenters.reserve(s.cube_group_count);

    uint32_t gCursor = 0;
    for (uint32_t layer = 0; layer < s.cube_layers && gCursor < s.cube_group_count; ++layer) {
        uint32_t groupsInLayer = layerCounts[layer];
        uint32_t nx = (uint32_t)std::ceil(std::sqrt(double(groupsInLayer)));
        uint32_t nz = (uint32_t)std::ceil(double(groupsInLayer) / std::max<uint32_t>(1u, nx));
        float layerY = s.cube_base_height + layer * s.cube_layer_spacing_world;

        // 层扰动
        if (s.cube_layer_jitter_enable) {
            std::mt19937 lrng(s.cube_layer_jitter_seed ^ layer);
            std::uniform_real_distribution<float> U(-1.f, 1.f);
            float amp = s.cube_layer_jitter_scale_h * h;
            float dy = U(lrng) * amp;
            layerY += dy;
        }

        for (uint32_t k = 0; k < groupsInLayer && gCursor < s.cube_group_count; ++k) {
            uint32_t ix = k % nx;
            uint32_t iz = k / nx;
            float cx = ix * s.cube_group_spacing_world;
            float cz = iz * s.cube_group_spacing_world;
            float cy = layerY;

            outCenters.push_back(make_float3(cx, cy, cz));
            ++gCursor;
        }
    }
}

void BuildRenderInitParams(const RuntimeConsole& c, gfx::RenderInitParams& out) {
    out.width  = c.app.width;
    out.height = c.app.height;
    out.vsync  = c.app.vsync;
}

// 简单三次根近似四舍五入
static inline uint32_t round_cuberoot(uint64_t v) {
    if (v == 0) return 0;
    double r = std::cbrt(double(v));
    return (uint32_t)std::max<int64_t>(1, (int64_t)llround(r));
}

// 生成亮色并与邻接团颜色保持距离
static bool GenDistinctColor(std::mt19937& rng,
    float minComp,
    float minDist,
    const std::vector<int>& adjacency, // indices 已有颜色的邻接节点
    const std::vector<std::array<float, 3>>& colors,
    int groupIndex,
    std::array<float, 3>& outColor,
    int retryMax)
{
    std::uniform_real_distribution<float> U(minComp, 1.0f);
    for (int attempt = 0; attempt < retryMax; ++attempt) {
        std::array<float, 3> c{ U(rng), U(rng), U(rng) };
        // 简单归一亮度（可选）
        float len = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
        if (len > 1e-6f) {
            float s = 1.0f / len;
            c[0] *= s; c[1] *= s; c[2] *= s;
        }
        bool ok = true;
        if (!adjacency.empty()) {
            for (int nb : adjacency) {
                if (nb < 0 || nb >= (int)colors.size()) continue;
                const auto& o = colors[nb];
                float dx = c[0] - o[0], dy = c[1] - o[1], dz = c[2] - o[2];
                float d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < minDist * minDist) { ok = false; break; }
            }
        }
        if (ok) { outColor = c; return true; }
    }
    // 失败退化：使用随机色不再强制距离
    outColor = { U(rng), U(rng), U(rng) };
    return false;
}

// 自动分解 & 域拟合 & 颜色生成
void PrepareCubeMix(RuntimeConsole& c) {
    if (c.sim.demoMode != RuntimeConsole::Simulation::DemoMode::CubeMix)
        return;

    auto& s = c.sim;

    // —— 自动分解 —— //
    if (s.cube_auto_partition) {
        uint64_t target = (uint64_t)std::max<uint32_t>(1u, s.numParticles);
        // 初始估计：边长 L 和组数 G
        uint32_t L = round_cuberoot(target);
        if (L == 0) L = 1;
        uint64_t per = (uint64_t)L * (uint64_t)L * (uint64_t)L;
        uint32_t G = (uint32_t)std::max<uint64_t>(1ull, target / per);
        if (G == 0) G = 1;
        // 调整 L 使 G*L^3 尽量接近 target
        while ((uint64_t)G * (uint64_t)L * (uint64_t)L * (uint64_t)L > target && L > 1) {
            --L;
        }
        while ((uint64_t)G * (uint64_t)(L + 1) * (uint64_t)(L + 1) * (uint64_t)(L + 1) <= target) {
            ++L;
        }
        // 若总数仍不足，可尝试增大 G
        per = (uint64_t)L * (uint64_t)L * (uint64_t)L;
        while ((uint64_t)(G + 1) * per <= target && (G + 1) <= s.cube_group_count_max) {
            ++G;
        }

        s.cube_edge_particles = L;
        s.cube_group_count = std::min<uint32_t>(G, s.cube_group_count_max);
        s.cube_particles_per_group = L * L * L;
        s.numParticles = s.cube_group_count * s.cube_particles_per_group;
    }
    else {
        s.cube_particles_per_group = s.cube_edge_particles * s.cube_edge_particles * s.cube_edge_particles;
        s.numParticles = s.cube_group_count * s.cube_particles_per_group;
    }

    if (s.cube_layers < 1) s.cube_layers = 1;
    if (s.cube_layers > s.cube_group_count) s.cube_layers = s.cube_group_count;

    // —— 布局 —— //
    // 每层的团数（最后一层可能少）
    std::vector<uint32_t> layerCounts;
    layerCounts.resize(s.cube_layers, 0);
    {
        uint32_t remaining = s.cube_group_count;
        for (uint32_t L = 0; L < s.cube_layers; ++L) {
            uint32_t toAssign = remaining / (s.cube_layers - L); // 平均分配
            if (toAssign == 0) toAssign = 1;
            layerCounts[L] = toAssign;
            remaining -= toAssign;
        }
    }

    const float h = (((1e-6f) > (s.smoothingRadius)) ? (1e-6f) : (s.smoothingRadius));
    const float spacing = h * s.cube_lattice_spacing_factor_h;
    const float cubeSideWorld = spacing * s.cube_edge_particles;

    // 临时记录每团的网格坐标（ix, iz, layer）用于相邻判断
    struct GroupLayout { int layer; int ix; int iz; };
    std::vector<GroupLayout> layouts;
    layouts.reserve(s.cube_group_count);

    // 计算整体包围盒（不含 margin）
    float minX = 1e30f, maxX = -1e30f;
    float minY = 1e30f, maxY = -1e30f;
    float minZ = 1e30f, maxZ = -1e30f;

    uint32_t gCursor = 0;
    for (uint32_t layer = 0; layer < s.cube_layers; ++layer) {
        uint32_t groupsInLayer = layerCounts[layer];
        // 近似正方分布
        uint32_t nx = (uint32_t)std::ceil(std::sqrt(double(groupsInLayer)));
        uint32_t nz = (uint32_t)std::ceil(double(groupsInLayer) / std::max<uint32_t>(1u, nx));
        float layerY = s.cube_base_height + layer * s.cube_layer_spacing_world;

        // 层扰动（可选）
        if (s.cube_layer_jitter_enable) {
            std::mt19937 lrng(s.cube_layer_jitter_seed ^ layer);
            std::uniform_real_distribution<float> U(-1.f, 1.f);
            float amp = s.cube_layer_jitter_scale_h * h;
            float dy = U(lrng) * amp;
            layerY += dy;
        }

        for (uint32_t k = 0; k < groupsInLayer && gCursor < s.cube_group_count; ++k) {
            uint32_t ix = k % nx;
            uint32_t iz = k / nx;
            float cx = ix * s.cube_group_spacing_world;
            float cz = iz * s.cube_group_spacing_world;
            float cy = layerY;

            // 更新包围盒
            minX = (((minX) < (cx - cubeSideWorld * 0.5f)) ? (minX) : (cx - cubeSideWorld * 0.5f));
            maxX = (((maxX) > (cx + cubeSideWorld * 0.5f)) ? (maxX) : (cx + cubeSideWorld * 0.5f));
            minY = (((minY) < (cy - cubeSideWorld * 0.5f)) ? (minY) : (cy - cubeSideWorld * 0.5f));
            maxY = (((maxY) > (cy + cubeSideWorld * 0.5f)) ? (maxY) : (cy + cubeSideWorld * 0.5f));
            minZ = (((minZ) < (cz - cubeSideWorld * 0.5f)) ? (minZ) : (cz - cubeSideWorld * 0.5f));
            maxZ = (((maxZ) > (cz + cubeSideWorld * 0.5f)) ? (maxZ) : (cz + cubeSideWorld * 0.5f));

            layouts.push_back({ (int)layer, (int)ix, (int)iz });
            ++gCursor;
        }
    }

    if (layouts.empty()) {
        // 兜底：至少一个
        layouts.push_back({ 0,0,0 });
        minX = minY = minZ = 0.f;
        maxX = maxY = maxZ = cubeSideWorld;
    }

    // —— 域拟合 —— //
    if (s.cube_auto_fit_domain) {
        float3 size = make_float3(maxX - minX, maxY - minY, maxZ - minZ);
        float3 mins = make_float3(minX, (((0.f) < (minY - cubeSideWorld * 0.5f)) ? (0.f) : (minY - cubeSideWorld * 0.5f)), minZ);
        float3 maxs = make_float3(minX + size.x, minY + size.y, minZ + size.z);

        // 放大 margin
        float3 center = make_float3((mins.x + maxs.x) * 0.5f,
            (mins.y + maxs.y) * 0.5f,
            (mins.z + maxs.z) * 0.5f);
        float scale = (((1.0f) > (s.cube_domain_margin_scale)) ? (1.0f) : (s.cube_domain_margin_scale));
        float3 half = make_float3((maxs.x - mins.x) * 0.5f * scale,
            (maxs.y - mins.y) * 0.5f * scale,
            (maxs.z - mins.z) * 0.5f * scale);
        s.gridMins = make_float3(center.x - half.x, (((0.f) > (center.y - half.y)) ? (0.f) : (center.y - half.y)), center.z - half.z);
        s.gridMaxs = make_float3(center.x + half.x, center.y + half.y, center.z + half.z);
    }

    // —— 颜色生成 —— //
    if (s.cube_color_enable) {
        std::mt19937 rng(s.cube_color_seed);
        std::vector<std::array<float, 3>> colors;
        colors.resize(s.cube_group_count);

        // 预构建邻接：同层内曼哈顿距离=1
        std::vector<std::vector<int>> adjacency(s.cube_group_count);
        for (int i = 0; i < (int)s.cube_group_count; ++i) {
            for (int j = 0; j < (int)s.cube_group_count; ++j) {
                if (i == j) continue;
                if (layouts[i].layer != layouts[j].layer) continue;
                int dx = std::abs(layouts[i].ix - layouts[j].ix);
                int dz = std::abs(layouts[i].iz - layouts[j].iz);
                if (dx + dz == 1) {
                    adjacency[i].push_back(j);
                }
            }
        }

        for (uint32_t g = 0; g < s.cube_group_count; ++g) {
            std::array<float, 3> col{};
            GenDistinctColor(rng,
                s.cube_color_min_component,
                s.cube_color_avoid_adjacent_similarity ? s.cube_color_min_distance : 0.f,
                adjacency[g],
                colors,
                (int)g,
                col,
                s.cube_color_retry_max);
            s.cube_group_colors[g][0] = col[0];
            s.cube_group_colors[g][1] = col[1];
            s.cube_group_colors[g][2] = col[2];
        }
    }

    // CubeMix 模式禁用连续喷口
    s.faucetFillEnable = false;
}

void ApplyRendererRuntime(const RuntimeConsole& c, gfx::RendererD3D12& r) {
    gfx::CameraParams cam{};
    cam.eye    = c.renderer.eye;
    cam.at     = c.renderer.at;
    cam.up     = c.renderer.up;
    cam.fovYDeg= c.renderer.fovYDeg;
    cam.nearZ  = c.renderer.nearZ;
    cam.farZ   = c.renderer.farZ;
    r.SetCamera(cam);

    gfx::VisualParams vis{};
    vis.particleRadiusPx = c.renderer.particleRadiusPx;
    vis.thicknessScale   = c.renderer.thicknessScale;
    if (c.viewer.enabled) {
        vis.clearColor[0] = c.viewer.background_color[0];
        vis.clearColor[1] = c.viewer.background_color[1];
        vis.clearColor[2] = c.viewer.background_color[2];
        vis.clearColor[3] = c.viewer.background_color[3];
    } else {
        vis.clearColor[0] = c.renderer.clearColor[0];
        vis.clearColor[1] = c.renderer.clearColor[1];
        vis.clearColor[2] = c.renderer.clearColor[2];
        vis.clearColor[3] = c.renderer.clearColor[3];
    }
    r.SetVisual(vis);
}

void FitCameraToDomain(RuntimeConsole& c) {
    const float3 mins = c.sim.gridMins;
    const float3 maxs = c.sim.gridMaxs;
    const float3 center = f3_mul(f3_add(mins, maxs), 0.5f);
    const float3 halfExtent = f3_mul(f3_sub(maxs, mins), 0.5f);

    float3 forward = f3_sub(c.renderer.at, c.renderer.eye);
    if (f3_len(forward) <= 1e-6f) forward = make_float3(0, 0, -1);
    forward = f3_norm(forward);

    float3 up = c.renderer.up;
    if (f3_len(up) <= 1e-6f || std::fabs(f3_dot(up, forward)) > 0.999f) up = make_float3(0, 1, 0);
    up = f3_norm(up);

    float3 right = f3_cross(forward, up);
    if (f3_len(right) <= 1e-6f) {
        forward = make_float3(0, 0, -1);
        up = make_float3(0, 1, 0);
        right = f3_cross(forward, up);
    }
    right = f3_norm(right);
    up = f3_cross(right, forward);

    const float halfUp = f3_proj_extent(halfExtent, up);
    const float halfRight = f3_proj_extent(halfExtent, right);
    const float halfForward = f3_proj_extent(halfExtent, forward);

    const float aspect = (c.app.height > 0u) ? (static_cast<float>(c.app.width) / static_cast<float>(c.app.height)) : 1.0f;
    const float fovY = (((1e-3f) > (c.renderer.fovYDeg * 3.14159265358979323846f / 180.0f)) ? (1e-3f) : (c.renderer.fovYDeg * 3.14159265358979323846f / 180.0f));
    const float fovX = 2.0f * std::atan(std::tan(fovY * 0.5f) * aspect);

    const float dV = halfUp / std::tan(fovY * 0.5f);
    const float dH = halfRight / std::tan(fovX * 0.5f);
    float d = (((dV) > (dH)) ? (dV) : (dH));

    const float pad = 0.05f * f3_len(halfExtent);
    d += pad; // 将 pad 合并到 d，统一处理

    // 保持朝向与原逻辑一致
    c.renderer.at = center;
    c.renderer.eye = f3_sub(center, f3_mul(forward, d + halfForward));

    // 修正可见深度范围：
    //  - 到“近端面”的距离 ≈ d
    //  - 到“远端面”的距离 ≈ d + 2*halfForward
    const float rawFar = d + 2.0f * halfForward;    // 修正点：原实现少了一个 halfForward
    const float rawNear = d;

    // 近裁剪面取较小值提升深度精度，但不小于 0.01
    const float nearClamp = (((0.01f) > (rawFar * 0.001f)) ? (0.01f) : (rawFar * 0.001f));
    c.renderer.nearZ = (((rawNear) < (nearClamp)) ? (rawNear) : (nearClamp));
    c.renderer.farZ = (((c.renderer.nearZ + 0.1f) > (rawFar)) ? (c.renderer.nearZ + 0.1f) : (rawFar));
}

} // namespace console