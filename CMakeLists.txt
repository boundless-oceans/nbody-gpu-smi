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

# ======================================================
# 依赖：OpenGL + GLEW + GLFW
# ======================================================
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# ======================================================
# 依赖：Dear ImGui（FetchContent 拉取源码）
# ======================================================
include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.90.6
)
FetchContent_MakeAvailable(imgui)

# ======================================================
# 可执行文件
# ======================================================
add_executable(nbody
    src/main.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(nbody PRIVATE
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
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
    glfw
)