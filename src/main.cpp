// N-Body GPU Simulator - S1 工程骨架
// GLFW 窗口 + Dear ImGui 空控制面板
//
// 基于 NVIDIA CUDA Samples 扩展（BSD-3-Clause）
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
// See LICENSES for details.

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cstdio>

// 窗口尺寸
constexpr int WINDOW_WIDTH = 1280;
constexpr int WINDOW_HEIGHT = 720;

// GLFW 错误回调
static void glfwErrorCallback(int error, const char* description)
{
    std::fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// 检查 OpenGL 错误
static bool checkGLError(const char* tag)
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::fprintf(stderr, "OpenGL Error [%s]: 0x%04X\n", tag, err);
        return false;
    }
    return true;
}

int main(int argc, char** argv)
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
    // 4. 主循环
    // --------------------------------
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // 开始 ImGui 帧
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ---- 控制面板（S1：空壳面板）----
        ImGui::Begin("N-Body Simulator");

        ImGui::Text("Welcome to N-Body GPU Simulator");
        ImGui::Separator();

        // S1 阶段仅显示占位信息，S6/S7 将添加真实控件
        ImGui::Text("S1: 工程骨架");
        ImGui::TextWrapped(
            "本面板为占位界面。后续将在此添加：\n"
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

        // S1：仅渲染 ImGui，粒子渲染（S4）后续加入
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        // 如果 CLEAR_COLOR 不匹配，会频繁报错，这里主动清除
        //（glewExperimental 模式下 glGetError 可能残留，忽略无害错误）
        while (glGetError() != GL_NO_ERROR) {
            // 清空残留错误
        }
    }

    // --------------------------------
    // 5. 清理
    // --------------------------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}