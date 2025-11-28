#define NOMINMAX

#include "console.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>
#include <vector>
#include "../../sim/numeric_utils.h"
#include "../../sim/logging.h" // 新增：统一日志接口

namespace {

	inline float3 f3_add(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
	inline float3 f3_sub(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
	inline float3 f3_mul(const float3& a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
	inline float f3_dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
	inline float3 f3_cross(const float3& a, const float3& b) {
		return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
	}
	inline float f3_len(const float3& v) { return std::sqrt((((0.0f) > (f3_dot(v, v))) ? (0.0f) : (f3_dot(v, v)))); }
	inline float3 f3_norm(const float3& v) {
		float len = f3_len(v);
		if (len <= 1e-20f) return make_float3(0, 0, 1);
		return f3_mul(v, 1.0f / len);
	}
	inline float f3_proj_extent(const float3& halfExt, const float3& axisUnit) {
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
		// 原有赋值保持
		out.numParticles = c.sim.numParticles;
		out.maxParticles = c.sim.maxParticles;
		out.dt = c.sim.dt;
		out.cfl = c.sim.cfl;
		out.gravity = c.sim.gravity;
		out.restDensity = c.sim.restDensity;
		out.solverIters = c.sim.solverIters;
		out.maxNeighbors = (c.perf.neighbor_cap > 0) ? c.perf.neighbor_cap : c.sim.maxNeighbors;
		out.boundaryRestitution = c.sim.boundaryRestitution;
		out.pbf = c.sim.pbf;
		out.xsph_c = c.sim.xsph_c;

		// ====== 新增：覆盖 pbf 的动态稳定性开关 ======
		out.pbf.xpbd_enable = c.sim.xpbd_enable ? 1 : 0;
		out.pbf.compliance = (c.sim.xpbd_enable ? c.sim.xpbd_compliance : 0.0f);
		out.pbf.lambda_warm_start_enable = c.sim.lambda_warm_start_enable ? 1 : 0;
		out.pbf.lambda_warm_start_decay = c.sim.lambda_warm_start_decay;
		out.pbf.semi_implicit_integration_enable = c.sim.integrate_semi_implicit ? 1 : 0;

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
	}

	void BuildDeviceParams(const RuntimeConsole& c, sim::DeviceParams& out) {
		sim::SimParams sp{};
		BuildSimParams(c, sp);
		out = sim::MakeDeviceParams(sp);
	}

	void GenerateCubeMixCenters(const RuntimeConsole& c, std::vector<float3>& outCenters) {
		outCenters.clear();
		if (c.sim.demoMode != RuntimeConsole::Simulation::DemoMode::CubeMix)
			return;
		const auto& s = c.sim;
		std::vector<uint32_t> layerCounts;
		layerCounts.resize(s.cube_layers, 0);
		uint32_t remaining = s.cube_group_count;
		for (uint32_t L = 0; L < s.cube_layers; ++L) {
			uint32_t toAssign = remaining / (s.cube_layers - L);
			if (toAssign == 0) toAssign = 1;
			layerCounts[L] = toAssign;
			remaining -= toAssign;
		}
		const float h = (((1e-6f) > (s.smoothingRadius)) ? (1e-6f) : (s.smoothingRadius));
		outCenters.reserve(s.cube_group_count);
		uint32_t gCursor = 0;
		for (uint32_t layer = 0; layer < s.cube_layers && gCursor < s.cube_group_count; ++layer) {
			uint32_t groupsInLayer = layerCounts[layer];
			uint32_t nx = (uint32_t)std::ceil(std::sqrt(double(groupsInLayer)));
			uint32_t nz = (uint32_t)std::ceil(double(groupsInLayer) / std::max<uint32_t>(1u, nx));
			float layerY = s.cube_base_height + layer * s.cube_layer_spacing_world;
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
		out.width = c.app.width;
		out.height = c.app.height;
		out.vsync = c.app.vsync;
	}

	static inline uint32_t round_cuberoot(uint64_t v) {
		if (v == 0) return 0;
		double r = std::cbrt(double(v));
		return (uint32_t)std::max<int64_t>(1, (int64_t)llround(r));
	}

	static bool GenDistinctColor(std::mt19937& rng,
		float minComp,
		float minDist,
		const std::vector<int>& adjacency,
		const std::vector<std::array<float, 3>>& colors,
		int groupIndex,
		std::array<float, 3>& outColor,
		int retryMax)
	{
		std::uniform_real_distribution<float> U(minComp, 1.0f);
		for (int attempt = 0; attempt < retryMax; ++attempt) {
			std::array<float, 3> c{ U(rng), U(rng), U(rng) };
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
		outColor = { U(rng), U(rng), U(rng) };
		return false;
	}

	void PrepareCubeMix(RuntimeConsole& c) {
		if (c.sim.demoMode != RuntimeConsole::Simulation::DemoMode::CubeMix)
			return;
		auto& s = c.sim;
		if (s.cube_auto_partition) {
			uint64_t target = (uint64_t)std::max<uint32_t>(1u, s.numParticles);
			uint32_t L = round_cuberoot(target);
			if (L == 0) L = 1;
			uint64_t per = (uint64_t)L * (uint64_t)L * (uint64_t)L;
			uint32_t G = (uint32_t)std::max<uint64_t>(1ull, target / per);
			if (G == 0) G = 1;
			while ((uint64_t)G * (uint64_t)L * (uint64_t)L * (uint64_t)L > target && L > 1) { --L; }
			while ((uint64_t)G * (uint64_t)(L + 1) * (uint64_t)(L + 1) * (uint64_t)(L + 1) <= target) { ++L; }
			per = (uint64_t)L * (uint64_t)L * (uint64_t)L;
			while ((uint64_t)(G + 1) * per <= target && (G + 1) <= s.cube_group_count_max) { ++G; }
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
		const float spacing = h * s.cube_lattice_spacing_factor_h;
		const float cubeSideWorld = spacing * s.cube_edge_particles;
		struct GroupLayout { int layer; int ix; int iz; };
		std::vector<GroupLayout> layouts; layouts.reserve(s.cube_group_count);
		float minX = 1e30f, maxX = -1e30f;
		float minY = 1e30f, maxY = -1e30f;
		float minZ = 1e30f, maxZ = -1e30f;
		uint32_t gCursor = 0;
		for (uint32_t layer = 0; layer < s.cube_layers; ++layer) {
			uint32_t groupsInLayer = layerCounts[layer];
			uint32_t nx = (uint32_t)std::ceil(std::sqrt(double(groupsInLayer)));
			uint32_t nz = (uint32_t)std::ceil(double(groupsInLayer) / std::max<uint32_t>(1u, nx));
			float layerY = s.cube_base_height + layer * s.cube_layer_spacing_world;
			if (s.cube_layer_jitter_enable) {
				std::mt19937 lrng(s.cube_layer_jitter_seed ^ layer);
				std::uniform_real_distribution<float> U(-1.f, 1.f);
				float amp = s.cube_layer_jitter_scale_h * h;
				float dy = U(lrng) * amp; layerY += dy;
			}
			for (uint32_t k = 0; k < groupsInLayer && gCursor < s.cube_group_count; ++k) {
				uint32_t ix = k % nx; uint32_t iz = k / nx;
				float cx = ix * s.cube_group_spacing_world;
				float cz = iz * s.cube_group_spacing_world;
				float cy = layerY;
				minX = std::min(minX, cx - cubeSideWorld * 0.5f);
				maxX = std::max(maxX, cx + cubeSideWorld * 0.5f);
				minY = std::min(minY, cy - cubeSideWorld * 0.5f);
				maxY = std::max(maxY, cy + cubeSideWorld * 0.5f);
				minZ = std::min(minZ, cz - cubeSideWorld * 0.5f);
				maxZ = std::max(maxZ, cz + cubeSideWorld * 0.5f);
				layouts.push_back({ (int)layer, (int)ix, (int)iz });
				++gCursor;
			}
		}
		if (layouts.empty()) {
			layouts.push_back({ 0,0,0 });
			minX = minY = minZ = 0.f; maxX = maxY = maxZ = cubeSideWorld;
		}
		if (s.cube_auto_fit_domain) {
			float3 size = make_float3(maxX - minX, maxY - minY, maxZ - minZ);
			float3 mins = make_float3(minX, (((0.f) < (minY - cubeSideWorld * 0.5f)) ? (0.f) : (minY - cubeSideWorld * 0.5f)), minZ);
			float3 maxs = make_float3(minX + size.x, minY + size.y, minZ + size.z);
			float3 center = make_float3((mins.x + maxs.x) * 0.5f, (mins.y + maxs.y) * 0.5f, (mins.z + maxs.z) * 0.5f);
			float scale = (((1.0f) > (s.cube_domain_margin_scale)) ? (1.0f) : (s.cube_domain_margin_scale));
			float3 half = make_float3((maxs.x - mins.x) * 0.5f * scale, (maxs.y - mins.y) * 0.5f * scale, (maxs.z - mins.z) * 0.5f * scale);
			s.gridMins = make_float3(center.x - half.x, (((0.f) > (center.y - half.y)) ? (0.f) : (center.y - half.y)), center.z - half.z);
			s.gridMaxs = make_float3(center.x + half.x, center.y + half.y, center.z + half.z);
		}
		if (s.cube_color_enable) {
			std::mt19937 rng(s.cube_color_seed);
			std::vector<std::array<float, 3>> colors; colors.resize(s.cube_group_count);
			std::vector<std::vector<int>> adjacency(s.cube_group_count);
			for (int i = 0; i < (int)s.cube_group_count; ++i) {
				for (int j = 0; j < (int)s.cube_group_count; ++j) {
					if (i == j) continue; if (layouts[i].layer != layouts[j].layer) continue;
					int dx = std::abs(layouts[i].ix - layouts[j].ix);
					int dz = std::abs(layouts[i].iz - layouts[j].iz);
					if (dx + dz == 1) adjacency[i].push_back(j);
				}
			}
			for (uint32_t g = 0; g < s.cube_group_count; ++g) {
				std::array<float, 3> col{};
				GenDistinctColor(rng, s.cube_color_min_component,
					s.cube_color_avoid_adjacent_similarity ? s.cube_color_min_distance : 0.f,
					adjacency[g], colors, (int)g, col, s.cube_color_retry_max);
				s.cube_group_colors[g][0] = col[0]; s.cube_group_colors[g][1] = col[1]; s.cube_group_colors[g][2] = col[2];
			}
		}
	}

	void ApplyRendererRuntime(const RuntimeConsole& c, gfx::RendererD3D12& r) {
		gfx::CameraParams cam{};
		cam.eye = c.renderer.eye; cam.at = c.renderer.at; cam.up = c.renderer.up;
		cam.fovYDeg = c.renderer.fovYDeg; cam.nearZ = c.renderer.nearZ; cam.farZ = c.renderer.farZ;
		r.SetCamera(cam);

		// 渲染模式
		r.SetRenderMode(c.renderer.renderMode);

		gfx::VisualParams vis{};
		vis.particleRadiusPx = c.renderer.particleRadiusPx;

		// 速度模式下根据控制参数决定 thicknessScale 用作速度归一化系数
		if (c.renderer.renderMode == gfx::RendererD3D12::RenderMode::SpeedColor) {
    		float scale = c.renderer.thicknessScale;

    		if (c.renderer.speedColorAutoScale) {
        	// 自动：根据 speedColorMaxSpeedHint 把速度映射到 [0,1]
        	float hint = (c.renderer.speedColorMaxSpeedHint > 1e-6f)
            ? c.renderer.speedColorMaxSpeedHint
            : 1e-6f;
        	scale = 1.0f / hint;
    		}

    	vis.thicknessScale = scale;
		}
		else {
    		// 原 group/palette 模式继续用 thicknessScale 做厚度参数
   			vis.thicknessScale = c.renderer.thicknessScale;
		}

		if (c.viewer.enabled) {
			vis.clearColor[0] = c.viewer.background_color[0];
			vis.clearColor[1] = c.viewer.background_color[1];
			vis.clearColor[2] = c.viewer.background_color[2];
			vis.clearColor[3] = c.viewer.background_color[3];
		}
		else {
			vis.clearColor[0] = c.renderer.clearColor[0];
			vis.clearColor[1] = c.renderer.clearColor[1];
			vis.clearColor[2] = c.renderer.clearColor[2];
			vis.clearColor[3] = c.renderer.clearColor[3];
		}
		r.SetVisual(vis);
	}
} // namespace console