// N-Body GPU Simulator
// S1: GLFW 窗口 + Dear ImGui 控制面板
// S3-1: CUDA-GL Interop 验证（PBO 注册 + CUDA 写 / GL 读回）
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

#include "render/interop_buffer.h"
#include "physics/cuda_check.h"

// 窗口尺寸
constexpr int WINDOW_WIDTH  = 1280;
constexpr int WINDOW_HEIGHT = 720;

// S3-1 测试粒子数（非最终模拟 N，仅验证 interop 链路）
constexpr int TEST_NUM_PARTICLES = 1024;

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
    // 2. 初始化 GLEW（OpenGL 扩展加载）
    // --------------------------------
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(err));
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

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
    // 4. S3-1: CUDA-GL Interop 验证
    //    PBO → 注册 CUDA → CUDA 写 → GL 读回 → 对比
    // --------------------------------
    bool interopOK = false;
    std::string interopMsg = "未测试";

    {
        // 4.1 创建 interop 缓冲（每粒子 4×float = xyz + mass）
        InteropBuffer buffer(TEST_NUM_PARTICLES, 4 * sizeof(float));

        // 4.2 构造模式数据（CUDA 侧通过 mapped pointer 写入）
        std::vector<float> writeData(TEST_NUM_PARTICLES * 4);
        for (int i = 0; i < TEST_NUM_PARTICLES; ++i) {
            writeData[i * 4 + 0] = (float)i;          // x = 索引
            writeData[i * 4 + 1] = (float)(i * 2);    // y
            writeData[i * 4 + 2] = (float)(i * 3);    // z
            writeData[i * 4 + 3] = 1.0f;              // mass
        }

        // 4.3 CUDA 锁定 PBO 并写数据
        cudaGraphicsResource* res = buffer.mapForCuda();
        {
            size_t bytes            = 0;
            void*  mappedPtr        = nullptr;
            CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(&mappedPtr, &bytes, res));
            CHECK_CUDA(cudaMemcpy(mappedPtr, writeData.data(), bytes, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        buffer.unmapFromCuda();

        // 4.4 OpenGL 读回 PBO 内容（此刻应能读到 CUDA 写入的数据）
        std::vector<float> readData(TEST_NUM_PARTICLES * 4, 0.0f);
        glBindBuffer(GL_ARRAY_BUFFER, buffer.getPBO());
        void* glPtr = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
        if (glPtr) {
            std::memcpy(readData.data(), glPtr, TEST_NUM_PARTICLES * 4 * sizeof(float));
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // 4.5 对比 CUDA 写入与 GL 读回的数据
        bool match = (glPtr != nullptr);
        if (match) {
            for (int i = 0; i < TEST_NUM_PARTICLES * 4; ++i) {
                if (readData[i] != writeData[i]) {
                    match = false;
                    break;
                }
            }
        }

        interopOK = match;
        interopMsg = match ? "PASS" : "FAIL";
        std::printf("[S3-1] CUDA-GL Interop 验证: %s\n", match ? "PASS" : "FAIL");
        if (!match) {
            std::fprintf(stderr, "[S3-1] 数据对比失败（CUDA 写入 vs GL 读回）\n");
        }
    }

    // --------------------------------
    // 5. 主循环
    // --------------------------------
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // 开始 ImGui 帧
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ---- 控制面板 ----
        ImGui::Begin("N-Body Simulator");

        ImGui::Text("Welcome to N-Body GPU Simulator");
        ImGui::Separator();

        ImGui::Text("S3-1: CUDA-GL Interop");
        ImGui::BulletText("PBO 粒子数: %d", TEST_NUM_PARTICLES);
        ImGui::BulletText("状态: %s", interopMsg.c_str());
        if (interopOK) {
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "CUDA 写入 → GL 读回 数据一致，Interop 链路打通");
        }
        else {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Interop 验证失败，请查看终端输出");
        }
        ImGui::Separator();

        ImGui::Text("S1: 工程骨架（占位）");
        ImGui::TextWrapped(
            "后续将在此添加：\n"
            "  - N 滑条（粒子数）\n"
            "  - 质量模式选择\n"
            "  - 预设场景按钮\n"
            "  - 运行控制（开始/暂停/单步/重置）\n"
            "  - FPS 显示");

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                    1000.0f / io.Framerate, io.Framerate);

        ImGui::End();

        // 渲染
        ImGui::Render();
        int displayW, displayH;
        glfwGetFramebufferSize(window, &displayW, &displayH);
        glViewport(0, 0, displayW, displayH);
        glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        // 清空残留 GL 错误
        while (glGetError() != GL_NO_ERROR) {
        }
    }

    // --------------------------------
    // 6. 清理
    // --------------------------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}