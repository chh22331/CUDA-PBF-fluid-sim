cmake_minimum_required(VERSION 3.20)

# Allow building without CUDA (toolchain issues) by toggling an option.
option(PBFX_ENABLE_CUDA "Enable CUDA builds" ON)

if (PBFX_ENABLE_CUDA)
  # CUDA 13 dropped support for compute_70, so force a safe architecture list
  # into the cache to avoid CMake's default that still includes 70.
  set(CMAKE_CUDA_ARCHITECTURES
      75 80 86 89 90
      CACHE STRING "CUDA SM architectures to build for" FORCE)
  enable_language(CUDA)
endif()

project(PBF_X LANGUAGES CXX C)

# Use C++17 - required by std::filesystem, std::clamp, inline variables, etc.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Keep CMake/Ninja artifacts under build/bin (runtime) and build/lib (archives)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(CMAKE_CONFIGURATION_TYPES)
  foreach(cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${cfg}" cfg_upper)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR}/lib/${cfg})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR}/bin/${cfg})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_BINARY_DIR}/bin/${cfg})
  endforeach()
endif()

if (MSVC)
  # Work around CMake + clang using 'ar'-style flags ('qc') with MSVC lib.exe.
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> /OUT:<TARGET> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> /OUT:<TARGET> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_FINISH "")
  if (PBFX_ENABLE_CUDA)
    set(CMAKE_CUDA_ARCHIVE_CREATE "<CMAKE_AR> /OUT:<TARGET> <OBJECTS>")
    set(CMAKE_CUDA_ARCHIVE_APPEND "<CMAKE_AR> /OUT:<TARGET> <OBJECTS>")
    set(CMAKE_CUDA_ARCHIVE_FINISH "")
  endif()

  # When using clang with lld-link on Windows, pass MSVC-style /flags via -Xlinker.
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-Xlinker /debug -Xlinker /INCREMENTAL" CACHE STRING "" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "-Xlinker /debug -Xlinker /INCREMENTAL" CACHE STRING "" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-Xlinker /INCREMENTAL:NO" CACHE STRING "" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "-Xlinker /INCREMENTAL:NO" CACHE STRING "" FORCE)
  endif()
endif()

if (PBFX_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# Core library with config + console implementation
add_library(pbf_core STATIC
  engine/core/config.cpp
  engine/core/console.cpp)

target_include_directories(pbf_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if (PBFX_ENABLE_CUDA)
  # console.h pulls in cuda_runtime.h; ensure core sees CUDA include dirs
  target_link_libraries(pbf_core PUBLIC CUDA::cudart)
endif()

# Simulation / CUDA library
add_library(pbf_cuda STATIC)

if (PBFX_ENABLE_CUDA)
  target_sources(pbf_cuda PRIVATE
    # Legacy sample kernel (kept for compatibility)
    kernel.cu

    # Host-side simulation pipeline
    sim/graph_builder.cpp
    sim/kernel_dispatcher.cpp
    sim/phases.cpp
    sim/poisson_disk.cpp
    sim/post_ops.cpp
    sim/simulator.cpp

    # CUDA kernels
    sim/boundary.cu
    sim/cell_compact.cu
    sim/cell_ranges.cu
    sim/device_globals.cu
    sim/device_pos_state.cu
    sim/hash_keys.cu
    sim/integrate.cu
    sim/pbf_delta_apply.cu
    sim/pbf_lambda.cu
    sim/sort_pairs.cu
    sim/stats.cu
    sim/stats_bruteforce.cu
    sim/velocity_globals.cu
    sim/xsph.cu)

  set_target_properties(pbf_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(pbf_cuda PUBLIC CUDA::cudart pbf_core)
else()
  message(WARNING "Building without CUDA; using CPU fallback for run_vector_add.")
  target_sources(pbf_cuda PRIVATE sim/kernel_fallback.cpp)
endif()

# App + Gfx (WIN32 to use wWinMain)
add_executable(pbf_app WIN32 app_main.cpp
  engine/gfx/d3d12_device.cpp
  engine/gfx/renderer.cpp)

target_link_libraries(pbf_app PRIVATE pbf_cuda pbf_core)
if (PBFX_ENABLE_CUDA)
  target_link_libraries(pbf_app PRIVATE CUDA::cudart)
endif()

# Windows libs
if (WIN32)
  target_link_libraries(pbf_app PRIVATE d3d12 dxgi)
  target_compile_definitions(pbf_app PRIVATE UNICODE _UNICODE NOMINMAX)
endif()
