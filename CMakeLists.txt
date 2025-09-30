cmake_minimum_required(VERSION 3.20)

# Allow building without CUDA (toolchain issues) by toggling an option.
option(PBFX_ENABLE_CUDA "Enable CUDA builds" ON)

if (PBFX_ENABLE_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
  enable_language(CUDA)
endif()

project(PBF_X LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (PBFX_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# Core minimal lib with config implementation
add_library(pbf_core STATIC
  engine/core/config.cpp)

target_include_directories(pbf_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# CUDA sample as a static library (or fallback)
add_library(pbf_cuda STATIC)

if (PBFX_ENABLE_CUDA)
  target_sources(pbf_cuda PRIVATE kernel.cu)
  set_target_properties(pbf_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(pbf_cuda PUBLIC CUDA::cudart)
else()
  message(WARNING "Building without CUDA; using CPU fallback for run_vector_add.")
  target_sources(pbf_cuda PRIVATE engine/sim/kernel_fallback.cpp)
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
