// N-Body GPU Simulator
// S1: GLFW 窗口 + ImGui 面板
// S3-1: CUDA-GL Interop 验证
// S3-3: 物理(PBO) + 粒子渲染(VAO) 实时展示
//
// 基于 NVIDIA CUDA Samples 扩展（BSD-3-Clause）
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
// See LICENSES for details.

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "physics/body_system.h"
#include "physics/body_system_cuda.h"
#include "physics/cuda_check.h"
#include "render/particle_renderer.h"

// 窗口尺寸
constexpr int WINDOW_WIDTH  = 1280;
constexpr int WINDOW_HEIGHT = 720;

// 模拟参数（S3-3 固定值，S6/S7 会做成 ImGui 控件）
constexpr int   NUM_BODIES    = 16384;
constexpr float TIMESTEP      = 0.016f;
constexpr float CLUSTER_SCALE = 1.54f;
constexpr float VELOCITY_SCALE = 8.0f;

// GLFW 错误回调
static void glfwErrorCallback(int error, const char* description)
{
    std::fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main()
{
    // --------------------------------
    // 1. 初始化 GLFW
    // --------------------------------
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "N-Body GPU Simulator", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // 垂直同步

    // --------------------------------
    // 2. 初始化 GLEW
    // --------------------------------
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(err));
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // 开启深度测试
    glEnable(GL_DEPTH_TEST);

    // --------------------------------
    // 3. 初始化 Dear ImGui
    // --------------------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // 清屏色（深空背景）
    float clearColor[4] = { 0.05f, 0.05f, 0.08f, 1.0f };

    // --------------------------------
    // 4. 创建物理系统（PBO 模式：位置由 InteropBuffer 提供）
    //    必须在 GL 上下文创建后再构造（InteropBuffer 需要 GL context 建 PBO）
    // --------------------------------
    BodySystemCUDA<float> system(NUM_BODIES, 256, /*usePBO=*/true);
    system.setSoftening(0.00125f);
    system.setDamping(0.995f);

    // 生成初始场景（壳层分布，视觉更像"星群"）
    srand(42);
    std::vector<float> hPos(NUM_BODIES * 4);
    std::vector<float> hVel(NUM_BODIES * 4);
    std::vector<float> hColor(NUM_BODIES * 4);

    randomizeBodies(NBODY_CONFIG_SHELL,
                    hPos.data(),
                    hVel.data(),
                    hColor.data(),
                    CLUSTER_SCALE,
                    VELOCITY_SCALE,
                    NUM_BODIES,
                    true);

    system.setArray(BODYSYSTEM_POSITION, hPos.data());
    system.setArray(BODYSYSTEM_VELOCITY, hVel.data());

    // --------------------------------
    // 5. 创建粒子渲染器（绑定 CUDA 写好的 PBO）
    // --------------------------------
    ParticleRenderer renderer;
    renderer.setPBO(system.getCurrentReadPBO(), system.getNumBodies());

    // 运行状态
    bool  paused  = false;
    float simTime = 0.0f;

    printf("[S3-3] 启动: %d 粒子, PBO 渲染模式\n", NUM_BODIES);

    // --------------------------------
    // 6. 主循环
    // --------------------------------
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // ---- 物理模拟（PBO 模式内部 map/unmap + 内核计算）----
        if (!paused) {
            system.update(TIMESTEP);
            simTime += TIMESTEP;
            // 双缓冲翻转后，渲染当前读缓冲
            renderer.setPBO(system.getCurrentReadPBO(), system.getNumBodies());
        }

        // ---- ImGui 帧 ----
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("N-Body Simulator");
        ImGui::Text("Welcome to N-Body GPU Simulator");
        ImGui::Separator();

        ImGui::Text("S3-3: 实时粒子渲染");
        ImGui::BulletText("粒子数: %u", system.getNumBodies());
        ImGui::BulletText("模拟时间: %.2f s", simTime);
        ImGui::BulletText("PBO 渲染: %s", system.usesPBO() ? "ON" : "OFF");
        ImGui::BulletText("状态: %s", paused ? "暂停(空格继续)" : "运行中(空格暂停)");
        ImGui::Separator();

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                    1000.0f / io.Framerate, io.Framerate);

        ImGui::End();

        // ---- 键盘：空格暂停/继续（ImGui 不捕获键盘时生效）----
        if (ImGui::IsKeyPressed(ImGuiKey_Space) && !io.WantCaptureKeyboard) {
            paused = !paused;
        }

        // ---- 渲染 ----
        ImGui::Render();

        int displayW, displayH;
        glfwGetFramebufferSize(window, &displayW, &displayH);
        glViewport(0, 0, displayW, displayH);
        glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 先渲染粒子（带深度），再渲染 ImGui 叠加
        renderer.render();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        // 清空残留 GL 错误
        while (glGetError() != GL_NO_ERROR) {
        }
    }

    // --------------------------------
    // 7. 清理（system 析构自动释放 PBO/显存）
    // --------------------------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}