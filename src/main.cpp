// N-Body GPU Simulator
// S1: GLFW 窗口 + ImGui 面板
// S3-1: CUDA-GL Interop 验证
// S3-3: 物理(PBO) + 粒子渲染(VAO) 实时展示
// S5: 质量映射（蓝=轻 → 红=重）
// S6: 场景预设 + 质量模式选择（ImGui 控制）
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
#include "scene_generator.h"

// 窗口尺寸
constexpr int WINDOW_WIDTH  = 1280;
constexpr int WINDOW_HEIGHT = 720;

// 模拟参数
constexpr int   NUM_BODIES    = 16384;
constexpr float TIMESTEP      = 0.016f;

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
    // 4. 创建物理系统（PBO 模式）
    // --------------------------------
    BodySystemCUDA<float> system(NUM_BODIES, 256, /*usePBO=*/true);
    system.setSoftening(0.00125f);
    system.setDamping(0.995f);

    // 场景配置（S6：UI 可切换）
    SceneConfig cfg;
    cfg.scene     = SceneType::RANDOM_CLOUD;
    cfg.massMode  = MassMode::CENTRAL_BODY;
    cfg.numBodies = NUM_BODIES;

    // 场景数据缓冲
    std::vector<float> hPos(NUM_BODIES * 4);
    std::vector<float> hVel(NUM_BODIES * 4);
    float              maxMass = 1.0f;

    // --------------------------------
    // 5. 创建粒子渲染器
    // --------------------------------
    ParticleRenderer renderer;
    renderer.setPBO(system.getCurrentReadPBO(), system.getNumBodies());
    renderer.setMaxMass(maxMass);

    // 运行状态
    bool  paused  = false;
    float simTime = 0.0f;

    // ---- 场景重置函数（切换到新场景/质量模式时调用）----
    auto resetScene = [&]() {
        srand(42);
        maxMass = generateScene(cfg, hPos, hVel);
        system.setArray(BODYSYSTEM_POSITION, hPos.data());
        system.setArray(BODYSYSTEM_VELOCITY, hVel.data());
        renderer.setMaxMass(maxMass);
        renderer.setPBO(system.getCurrentReadPBO(), system.getNumBodies());
        simTime = 0.0f;
        printf("[S6] 场景: %s, 质量: %s, maxMass=%.1f\n", sceneName(cfg.scene), massModeName(cfg.massMode), maxMass);
    };

    // 初始场景
    resetScene();

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
            renderer.setPBO(system.getCurrentReadPBO(), system.getNumBodies());
        }

        // ---- ImGui 帧 ----
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("N-Body Simulator");

        // ---- S6: 场景与质量模式选择 ----
        ImGui::Text("场景预设:");
        int selScene = (int)cfg.scene;
        if (ImGui::Combo("##scene", &selScene,
                         "随机粒子云\0双粒子互绕\0格点扰动\0双星系统\0旋转星系盘\0太阳系\0")) {
            cfg.scene = (SceneType)selScene;
        }

        ImGui::Text("质量模式:");
        int selMass = (int)cfg.massMode;
        if (ImGui::Combo("##mass", &selMass, "均匀\0随机分布\0中心天体\0")) {
            cfg.massMode = (MassMode)selMass;
        }

        if (ImGui::Button("应用并重置场景")) {
            resetScene();
        }
        ImGui::SameLine();
        if (ImGui::Button("重置（同场景）")) {
            paused = false;
            resetScene();
        }

        ImGui::Separator();
        ImGui::Text("当前: %s / %s", sceneName(cfg.scene), massModeName(cfg.massMode));
        ImGui::BulletText("粒子数: %u", system.getNumBodies());
        ImGui::BulletText("模拟时间: %.2f s", simTime);
        ImGui::BulletText("质量范围: 1.0 ~ %.1f", maxMass);
        ImGui::BulletText("PBO 渲染: %s", system.usesPBO() ? "ON" : "OFF");
        ImGui::BulletText("状态: %s", paused ? "暂停(空格继续)" : "运行中(空格暂停)");
        ImGui::Separator();

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                    1000.0f / io.Framerate, io.Framerate);

        ImGui::End();

        // ---- 键盘：空格暂停/继续 ----
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
    // 7. 清理
    // --------------------------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}