# N-Body GPU Simulator

基于 NVIDIA CUDA Samples 官方 nbody 示例扩展的交互式 GPU N 体引力模拟器。

## 项目说明

以 NVIDIA CUDA Samples 的 `Samples/5_Domain_Specific/nbody` 为蓝本（其算法源自《GPU Gems 3》第 31 章《Fast N-Body Simulation with CUDA》），在官方物理内核与 CUDA-GL 零拷贝渲染管线之上，构建带交互控制界面的实时引力模拟器，支持粒子数量（N）调节、质量差异化分布、预设天体场景等交互功能。

## 技术栈

| 层次 | 技术 |
|---|---|
| 物理计算 | CUDA 11.8（共享内存 Tiling、双缓冲、常量内存、GPU 归约） |
| 编程语言 | C++17 |
| 窗口 / 事件 | GLFW 3 |
| 图形 API | OpenGL（现代渲染管线） |
| UI 框架 | Dear ImGui |
| 构建系统 | CMake |
| CPU 对照 | OpenMP（正确性验证） |

## 许可证

本项目包含自 NVIDIA CUDA Samples 移植的代码，遵循 BSD 3-Clause License：

- 源文件头部保留 NVIDIA 原始版权声明（Copyright (c) 2022, NVIDIA CORPORATION）
- 仓库根目录保留 LICENSE 文件（与 NVIDIA 官方一致）
- 二进制或构建产物的分发须附带版权声明与免责条款
- 不得使用 NVIDIA CORPORATION 名称及其贡献者名称进行背书或推广
- 另有 NVIDIA CUDA EULA 附加条款

## 参考

- NVIDIA CUDA Samples: https://github.com/NVIDIA/cuda-samples
- GPU Gems 3, Chapter 31: Fast N-Body Simulation with CUDA
- NVIDIA CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Dear ImGui: https://github.com/ocornut/imgui
- GLFW: https://www.glfw.org/