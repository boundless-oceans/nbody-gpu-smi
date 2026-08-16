cmake_minimum_required(VERSION 3.20)

project(nbody-gpu-smi LANGUAGES CXX CUDA)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 架构（由 scripts/build.sh 自动检测传入，默认 RTX 40 系 sm_89）
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# CUDA Toolkit（提供 CUDA::cudart 等 targets）
find_package(CUDAToolkit REQUIRED)

# ======================================================
# 依赖检查：OpenGL（系统自带，一般都有）
# ======================================================
find_package(OpenGL QUIET)
if(NOT OpenGL_FOUND)
    message(FATAL_ERROR
        "[依赖缺失] OpenGL 开发库未找到。\n"
        "请安装: sudo apt install libgl1-mesa-dev")
endif()

# ======================================================
# 依赖检查：GLEW（OpenGL 扩展加载库，系统包）
# ======================================================
find_package(GLEW QUIET)
if(NOT GLEW_FOUND)
    message(FATAL_ERROR
        "[依赖缺失] GLEW 未找到。\n"
        "请安装: sudo apt install libglew-dev\n"
        "或自动安装: ./scripts/setup_env.sh --install")
endif()

# ======================================================
# 依赖检查：GLFW（窗口库，系统包）
# ======================================================
find_package(glfw3 QUIET)
set(GLFW_LIBS glfw)
if(NOT glfw3_FOUND)
    message(FATAL_ERROR
        "[依赖缺失] GLFW 未找到。\n"
        "请安装: sudo apt install libglfw3-dev\n"
        "或自动安装: ./scripts/setup_env.sh --install")
endif()

# ======================================================
# 依赖：Dear ImGui（纯源码库）
#   策略：优先使用本地 third_party/imgui（离线可用）；
#         缺失时自动 FetchContent 从 GitHub 拉取。
# ======================================================
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui)

if(NOT EXISTS ${IMGUI_DIR}/imgui.h)
    message(STATUS "ImGui 本地源码未找到，自动从 GitHub 拉取...")
    include(FetchContent)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.90.6
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(imgui)
    set(IMGUI_DIR ${imgui_SOURCE_DIR})
else()
    message(STATUS "使用本地 ImGui 源码: ${IMGUI_DIR}")
endif()

# ======================================================
# 物理模块源（CUDA 内核 + 头文件）
# ======================================================
set(PHYSICS_SOURCES
    src/physics/body_system_cuda_kernel.cu
)

set(PHYSICS_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ======================================================
# GUI 可执行文件（S1 工程骨架：GLFW + ImGui 面板）
# ======================================================
add_executable(nbody
    src/main.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(nbody PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${PHYSICS_INCLUDE_DIRS}
)

# 编译选项
target_compile_definitions(nbody PRIVATE
    IMGUI_IMPL_OPENGL_LOADER_GLEW   # ImGui OpenGL3 后端使用 GLEW 加载函数
)

target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# 链接
target_link_libraries(nbody PRIVATE
    OpenGL::GL
    GLEW::GLEW
    ${GLFW_LIBS}
)

# ======================================================
# Benchmark 可执行文件（S2 验证点：无窗口物理性能）
# 用法: ./nbody_benchmark [numBodies] [iterations]
# ======================================================
add_executable(nbody_benchmark
    src/benchmark.cpp
    ${PHYSICS_SOURCES}
)

target_include_directories(nbody_benchmark PRIVATE
    ${PHYSICS_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}   # .cpp 编译需显式 CUDA 头文件路径（vector_types.h 等）
)

target_compile_options(nbody_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

target_compile_definitions(nbody_benchmark PRIVATE
    CUDA_ARCHITECTURES_CURRENT="${CMAKE_CUDA_ARCHITECTURES}"
)

target_link_libraries(nbody_benchmark PRIVATE
    CUDA::cudart
)