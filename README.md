# PBF-X / 基于位置的流体模拟

<img width="2559" height="1530" alt="屏幕截图 2025-12-05 225717" src="https://github.com/user-attachments/assets/5c1e37ae-113b-49da-bd9b-15fefc25d808" />

PBF-X is a Windows-focused Position-Based Fluids sandbox that pairs a lightweight Direct3D 12 engine with CUDA-accelerated simulation kernels.  
PBF-X 是一个面向 Windows 的基于位置的流体沙盒，结合了轻量级的 Direct3D 12 引擎与 CUDA 加速的仿真内核。  
This is a just-for-fun project; everyone is welcome to explore or chat, but hands-on technical support might not always be available.  
本项目只是个人的兴趣创作，欢迎所有人前来交流，不过无法保证能够随时提供技术支持。

## Overview / 项目简介
- Core gameplay-independent systems (configuration, console, renderer, interop helpers) live under `engine/`.  
  核心的通用系统（配置、控制台、渲染、互操作工具）位于 `engine/`。
- Simulation logic plus CUDA kernels are kept in `sim/`, covering graph construction, phase scheduling, and GPU kernels such as `pbf_lambda.cu`.  
  仿真逻辑与 CUDA 内核集中在 `sim/`，包含图构建、阶段调度以及 `pbf_lambda.cu` 等 GPU 内核。

## Project Layout / 仓库结构
- `engine/core/`, `engine/gfx/`, `engine/interop/` – platform layer, configuration, D3D12 renderer, and bridge helpers.  
  `engine/core/`、`engine/gfx/`、`engine/interop/` —— 平台层、配置、D3D12 渲染器与桥接工具。
- `sim/` – CUDA + CPU simulation stages (graph builder, kernels, statistics).  
  `sim/` —— CUDA 与 CPU 混合的仿真阶段（图构建、内核、统计）。
- `build/` – Default CMake/Ninja output directory.  
  `build/` —— 默认的 CMake/Ninja 构建输出目录。
- `x64/Release/` – Visual Studio Release artifacts (e.g., `PBF-X.exe`).  
  `x64/Release/` —— Visual Studio Release 构建产物（如 `PBF-X.exe`）。

## Prerequisites / 环境依赖
- Windows 10/11 with a GPU that supports CUDA SM 75+.  
  运行 Windows 10/11，GPU 需支持 CUDA SM 75 及以上。
- CMake ≥ 3.20 and Ninja (default generator).  
  CMake 3.20 及以上版本和 Ninja（默认生成器）。
- Visual Studio 2022 (Desktop development with C++) or clang-cl + lld-link toolchain.  
  Visual Studio 2022（含 C++ 桌面开发组件）或 clang-cl + lld-link 工具链。
- NVIDIA CUDA Toolkit (matching your driver, 12.x recommended).  
  NVIDIA CUDA Toolkit（与驱动匹配，推荐 12.x）。
- Latest Windows SDK + DirectX 12 runtime.  
  最新 Windows SDK 与 DirectX 12 运行库。

## Configure & Build / 配置与构建
### CMake + Ninja
1. Configure with CUDA enabled (default `ON`):  
   `cmake -S . -B build -DPBFX_ENABLE_CUDA=ON`  
   启用 CUDA（默认 `ON`）执行配置：`cmake -S . -B build -DPBFX_ENABLE_CUDA=ON`
2. Build the Release binary:  
   `cmake --build build --config Release`  
   编译 Release 版本：`cmake --build build --config Release`

### Visual Studio Workflow / 使用 Visual Studio
1. Open `PBF-X.sln` or generate via CMake GUI with the Visual Studio generator.  
   打开 `PBF-X.sln` 或使用 CMake GUI 生成 VS 解决方案。
2. Ensure `PBFX_ENABLE_CUDA` stays checked so CUDA kernels compile; toggle only if you lack a CUDA-capable GPU.  
   请保持 `PBFX_ENABLE_CUDA` 选中，以编译 CUDA 内核；仅在没有 CUDA GPU 时关闭。
3. Select the `Release` x64 configuration and build `pbf_app`.  
   选择 x64 `Release` 配置并生成 `pbf_app`。

### Build Options / 构建选项
- `PBFX_ENABLE_CUDA` (ON/OFF): Enables CUDA targets, injects CUDA include paths into `pbf_core`, and selects SM architectures 75/80/86/89/90.  
  `PBFX_ENABLE_CUDA`（开/关）：启用 CUDA 目标、为 `pbf_core` 添加 CUDA 头路径，并固定 SM 架构为 75/80/86/89/90。
- Override CUDA architectures if needed:  
  `cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="89;90"`  
  可自定义 CUDA 架构：`cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="89;90"`

## Run / 运行
- Standalone executable (after Ninja build): `.\build\pbf_app.exe`  
  Ninja 构建后直接运行：`.\build\pbf_app.exe`
- Visual Studio output: `.\x64\Release\PBF-X.exe`  
  Visual Studio 构建产物：`.\x64\Release\PBF-X.exe`
- Launch with a GPU connected display for the D3D12 swap chain; use the in-app console to adjust runtime parameters.  
  启动时需连接 GPU 显示输出以创建 D3D12 交换链；可使用应用内控制台调节运行时参数。

## Runtime Console / 运行时控制台
- Editable defaults live in `engine/core/console.h` under `console::RuntimeConsole`; tweaking fields like `sim.h_over_r = 2.5f`, `sim.solverIters`, or `renderer.particleRadiusPx` lets you script new presets.  
  `engine/core/console.h` 中的 `console::RuntimeConsole` 集中了所有默认参数，可调整 `sim.h_over_r = 2.5f`、`sim.solverIters`、`renderer.particleRadiusPx` 等字段以定义新的预设。
- Helper functions (`console::BuildSimParams`, `console::ApplyRendererRuntime`, etc.) propagate these values into `sim::SimParams`, `sim::DeviceParams`, and `gfx::RenderInitParams` at startup or during hot reload.  
  启动或热重载时，`console::BuildSimParams`、`console::ApplyRendererRuntime` 等辅助函数会把这些数值注入 `sim::SimParams`、`sim::DeviceParams` 与 `gfx::RenderInitParams`。

## Development Notes / 开发注意事项
- Language standards: C++17 and CUDA C++ for engine/sim code; use 4-space indentation, braces on the same line.  
  语言规范：引擎/仿真代码使用 C++17 与 CUDA C++；缩进为 4 空格，左括号与语句同一行。
- Naming: Types use PascalCase, functions camelCase, members prefixed with `m_`, globals with `g_`.  
  命名：类型 PascalCase，函数 camelCase，成员加 `m_` 前缀，全局变量加 `g_` 前缀。
- Keep headers focused (`sim/`, `engine/core/`) and explicitly separate CPU/GPU data paths (see `device_buffers.cuh`).  
  头文件保持精简（`sim/`、`engine/core/`），并显式区分 CPU/GPU 数据路径（参考 `device_buffers.cuh`）。

## Validation / 验证
- There is no automated test suite yet; validate by running `pbf_app` and exercising key scenarios (camera flight, UI toggles, simulation resets).  
  当前尚无自动化测试；请运行 `pbf_app`，覆盖关键场景（相机飞行、UI 开关、仿真重置）。
- Compare HUD stats (particle count, FPS, solver iterations) before/after your change to ensure regressions are caught early.  
  修改前后对比 HUD 统计（粒子数量、FPS、求解迭代）以尽早发现回归。

## Performance Snapshot / 性能快照
- On an RTX 4060 Laptop GPU, Release builds running two solver iterations with a smoothing ratio `h_over_r = 2.5` (smoothing radius ≈ 2.5× particle radius) sustain roughly 1,000,000 particles at ~60 FPS.  
  在 RTX 4060 Laptop GPU 上，Release 构建、两次求解迭代并设置 `h_over_r = 2.5`（平滑半径约等于粒子半径的 2.5 倍）时，大约可以维持 1,000,000 粒子并保持 60 FPS 左右。

## Troubleshooting / 常见问题
- Missing CUDA runtime headers usually indicates `PBFX_ENABLE_CUDA` was disabled or the CUDA Toolkit path is not visible to CMake. Verify `find_package(CUDAToolkit REQUIRED)` succeeds during configuration.  
  找不到 CUDA 头文件通常是 `PBFX_ENABLE_CUDA` 被关闭或 CUDA Toolkit 路径不可见；请确保配置阶段 `find_package(CUDAToolkit REQUIRED)` 成功。
- Linker errors with clang-cl on Windows can be resolved by keeping the provided `/INCREMENTAL:NO` flags and ensuring lld-link is selected.  
  在 Windows 上使用 clang-cl 出现链接错误时，请保留 `/INCREMENTAL:NO` 等提供的标志并确认选择了 lld-link。
- D3D12 initialization failures often stem from running over Remote Desktop without a hardware adapter—launch locally or use a GPU-capable remote solution.  
  D3D12 初始化失败多半因为远程桌面缺少硬件适配器——请在本地或具备 GPU 支持的远程环境运行。
