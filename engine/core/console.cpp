#include "console.h"
#include <algorithm>
#include <cmath>

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

void BuildSimParams(const RuntimeConsole& c, sim::SimParams& out) {
    // 运行规模
    out.numParticles = c.sim.numParticles;
    out.maxParticles = c.sim.maxParticles;

    // 动力学与数值控制
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

    // XSPH 系数从 console 透传
    out.xsph_c = c.sim.xsph_c;

    // h 推导选择
    float h = (c.sim.smoothingRadius > 0.f) ? c.sim.smoothingRadius : 0.02f;
    if (c.sim.deriveHFromRadius) {
        const float r = (((1e-8f) > (c.sim.particleRadiusWorld)) ? (1e-8f) : (c.sim.particleRadiusWorld));
        const float h_from_r = c.sim.h_over_r * r;
        if (h_from_r > 0.0f) h = h_from_r;
    }

    // 核系数
    out.kernel = sim::MakeKernelCoeffs(h);

    // 质量定义
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

    // 网格参数
    float cell = c.sim.cellSize;
    if (cell <= 0.0f) {
        float m = (c.perf.grid_cell_size_multiplier > 0.0f) ? c.perf.grid_cell_size_multiplier : 1.0f;
        cell = (((1e-6f) > (m * h)) ? (1e-6f) : (m * h));
    }
    out.grid.mins = c.sim.gridMins;
    out.grid.maxs = c.sim.gridMaxs;
    out.grid.cellSize = cell;
    out.grid.dim = make_grid_dim(out.grid.mins, out.grid.maxs, cell);
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

void BuildRenderInitParams(const RuntimeConsole& c, gfx::RenderInitParams& out) {
    out.width  = c.app.width;
    out.height = c.app.height;
    out.vsync  = c.app.vsync;
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