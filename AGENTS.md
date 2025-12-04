# Repository Guidelines

## Project Structure & Modules
- Core engine code lives in `engine/` (`core/`, `gfx/`, `interop/`).
- Simulation logic and CUDA kernels are in `sim/` (e.g. `simulator.{h,cpp}`, `*.cu`).
- CMake/Ninja builds land in `build/`; Visual Studio builds land in `x64/Release/`.

## Build, Run, and Development
- Configure CMake (with CUDA):  
  `cmake -S . -B build -DPBFX_ENABLE_CUDA=ON`
- Build with Ninja (default generator):  
  `cmake --build build --config Release`
- Run the standalone app:  
  `.\build\pbf_app.exe` or `.\x64\Release\PBF-X.exe`
- When editing CUDA code, keep `PBFX_ENABLE_CUDA` enabled so kernels are compiled.

## Coding Style & Naming
- Language: C++17 and CUDA C++ for core/sim.
- Indent with 4 spaces, no tabs; brace on same line as control statement/declaration.
- Types use `PascalCase` (`Simulator`, `FreeFlyCamera`); functions/methods use `camelCase` (`initialize`, `seedBoxLattice`); members are prefixed with `m_`, globals with `g_`.
- Prefer small, focused headers in `sim/` and `engine/core/` and keep GPU/CPU boundaries explicit (see `device_buffers.cuh`, `simulation_context.h`).

## Testing & Validation
- There is no dedicated automated test suite yet; validate changes by:
  - Running `pbf_app` and exercising relevant camera, UI, and simulation scenarios.
  - Comparing particle counts, FPS, and stats in the HUD before/after a change.
- If you add tests, integrate them via CTest and CMake (`add_test`) and place helpers alongside the code they cover.

## Commits & Pull Requests
- Keep commits small and focused; use short imperative messages describing the change (e.g. `tune lambda step`, `refactor graph builder`).
- For pull requests, include:
  - A brief high-level description and rationale.
  - Notes on performance impact (FPS, GPU time) and how you measured it.
  - Any new configuration flags or console commands added for debugging.
