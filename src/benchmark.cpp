/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * -------------------------------------------------------------------------
 * N-Body GPU Simulator - Benchmark 入口
 * S2: 无窗口物理性能基准
 * S3-2: 支持 --pbo（PBO 模式经隐藏 GL 窗口，验证 interop 物理正确性）
 * 用法: ./nbody_benchmark [numBodies] [iterations] [--pbo]
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "physics/body_system.h"
#include "physics/body_system_cuda.h"
#include "physics/cuda_check.h"

// 默认参数（与 NVIDIA 官方 nbody 默认一致）
static constexpr int   DEFAULT_NUM_BODIES     = 24576;
static constexpr int   DEFAULT_ITERATIONS     = 5;
static constexpr int   DEFAULT_BLOCK_SIZE     = 256;
static constexpr float TIMESTEP               = 0.016f;
static constexpr float CLUSTER_SCALE          = 1.54f;
static constexpr float VELOCITY_SCALE         = 8.0f;
static constexpr int   FLOPS_PER_INTERACTION  = 20;

// 单精度性能统计（与 NVIDIA 版 computePerfStats 一致）
void computePerfStats(int numBodies, int iterations, float milliseconds, double& interactionsPerSecond, double& gflops)
{
    interactionsPerSecond = (float)numBodies * (float)numBodies;
    interactionsPerSecond *= 1e-9 * iterations * 1000 / milliseconds;
    gflops = interactionsPerSecond * (float)FLOPS_PER_INTERACTION;
}

// =============================================
// S3-2: 在隐藏 GL 窗口上下文下创建 InteropBuffer
// （PBO 属于 GL 资源，需要 GL context 才能创建）
// =============================================
#include <GL/glew.h>
#include <GLFW/glfw3.h>

static bool initHiddenGLContext()
{
    if (!glfwInit()) {
        std::fprintf(stderr, "[S3-2] glfwInit failed\n");
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);  // 隐藏窗口（无桌面）

    GLFWwindow* win = glfwCreateWindow(1, 1, "hidden", nullptr, nullptr);
    if (!win) {
        std::fprintf(stderr, "[S3-2] Failed to create hidden GLFW window\n");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(win);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::fprintf(stderr, "[S3-2] glewInit failed\n");
        glfwDestroyWindow(win);
        glfwTerminate();
        return false;
    }

    return true;
}

static void shutdownHiddenGLContext() { glfwTerminate(); }

// 运行一次模拟，返回迭代 ms 与最终位置（拷回 CPU）
struct SimResult
{
    float               milliseconds = 0.0f;
    bool                valid        = false;
    std::vector<float>  finalPos;
};

template <typename T>
SimResult runSimulation(int numBodies, int iterations, bool usePBO)
{
    SimResult result;

    // 创建物理系统（usePBO 时位置缓冲走 InteropBuffer）
    BodySystemCUDA<T> system(numBodies, DEFAULT_BLOCK_SIZE, usePBO);
    system.setSoftening(0.00125f);
    system.setDamping(0.995f);

    // 生成初始场景（固定 seed，保证两种模式输入完全一致）
    srand(42);
    std::vector<T>      hPos(numBodies * 4);
    std::vector<T>      hVel(numBodies * 4);
    std::vector<float>  hColor(numBodies * 4);

    randomizeBodies(NBODY_CONFIG_SHELL,
                    hPos.data(),
                    hVel.data(),
                    hColor.data(),
                    CLUSTER_SCALE,
                    VELOCITY_SCALE,
                    numBodies,
                    true);

    system.setArray(BODYSYSTEM_POSITION, hPos.data());
    system.setArray(BODYSYSTEM_VELOCITY, hVel.data());

    // 预热一次（编译 kernel、加载设备）
    system.update(TIMESTEP);

    // 计时
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    CHECK_CUDA(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < iterations; ++i) {
        system.update(TIMESTEP);
    }
    CHECK_CUDA(cudaEventRecord(stopEvent, 0));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    CHECK_CUDA(cudaEventElapsedTime(&result.milliseconds, startEvent, stopEvent));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));

    // 位置数据拷回（比较用）
    T* pos = system.getArray(BODYSYSTEM_POSITION);
    result.finalPos.assign(pos, pos + numBodies * 4);

    // 有限性验证
    result.valid = true;
    for (float v : result.finalPos) {
        if (!std::isfinite(v)) {
            result.valid = false;
            break;
        }
    }

    return result;
}

int main(int argc, char** argv)
{
    int  numBodies  = DEFAULT_NUM_BODIES;
    int  iterations = DEFAULT_ITERATIONS;
    bool usePBO     = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--pbo") == 0) {
            usePBO = true;
        }
        else if (numBodies == DEFAULT_NUM_BODIES && iterations == DEFAULT_ITERATIONS) {
            numBodies = std::atoi(argv[i]);
            if (i + 1 < argc && std::atoi(argv[i + 1]) > 0) {
                iterations = std::atoi(argv[++i]);
            }
        }
        else {
            std::fprintf(stderr, "未知参数: %s\n", argv[i]);
            return -1;
        }
    }

    printf("> CUDA N-Body Benchmark\n");
    printf("> Single precision floating point simulation\n");
    printf("> %d bodies, %d iterations, blockSize=%d\n", numBodies, iterations, DEFAULT_BLOCK_SIZE);
    printf("> Mode: %s\n\n", usePBO ? "PBO (CUDA-GL Interop)" : "Plain (cudaMalloc)");

    // 打印 GPU 信息
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return -1;
    }
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
    printf("GPU Device 0: \"%s\" with compute capability %d.%d\n\n", props.name, props.major, props.minor);

    // 双模式一致性对比：普通模式 + PBO 模式各跑一次（同输入）
    if (!usePBO) {
        printf("--- Plain mode ---\n");
        SimResult plainResult = runSimulation<float>(numBodies, iterations, false);
        double    plainIps = 0, plainGflops = 0;
        computePerfStats(numBodies, iterations, plainResult.milliseconds, plainIps, plainGflops);
        printf("%d bodies, total time for %d iterations: %.3f ms\n", numBodies, iterations, plainResult.milliseconds);
        printf("= %.3f billion interactions per second\n", plainIps);
        printf("= %.3f single-precision GFLOP/s at %d flops per interaction\n", plainGflops, FLOPS_PER_INTERACTION);

        printf("\n--- PBO mode ---\n");
        if (!initHiddenGLContext()) {
            return -1;
        }
        SimResult pboResult = runSimulation<float>(numBodies, iterations, true);
        shutdownHiddenGLContext();
        double pboIps = 0, pboGflops = 0;
        computePerfStats(numBodies, iterations, pboResult.milliseconds, pboIps, pboGflops);
        printf("%d bodies, total time for %d iterations: %.3f ms\n", numBodies, iterations, pboResult.milliseconds);
        printf("= %.3f billion interactions per second\n", pboIps);
        printf("= %.3f single-precision GFLOP/s at %d flops per interaction\n", pboGflops, FLOPS_PER_INTERACTION);

        // 一致性对比：普通模式 vs PBO 模式最终位置
        bool match = plainResult.valid && pboResult.valid
                     && plainResult.finalPos.size() == pboResult.finalPos.size();
        if (match) {
            for (size_t i = 0; i < plainResult.finalPos.size(); ++i) {
                if (std::fabs(plainResult.finalPos[i] - pboResult.finalPos[i]) > 1e-6f) {
                    match = false;
                    break;
                }
            }
        }
        printf("\n[S3-2] Plain vs PBO 位置一致性: %s\n", match ? "PASS" : "FAIL");
        return match ? 0 : -1;
    }

    // 单独 PBO 模式
    if (!initHiddenGLContext()) {
        return -1;
    }
    SimResult pboResult = runSimulation<float>(numBodies, iterations, true);
    shutdownHiddenGLContext();

    double ips = 0, gflops = 0;
    computePerfStats(numBodies, iterations, pboResult.milliseconds, ips, gflops);
    printf("%d bodies, total time for %d iterations: %.3f ms\n", numBodies, iterations, pboResult.milliseconds);
    printf("= %.3f billion interactions per second\n", ips);
    printf("= %.3f single-precision GFLOP/s at %d flops per interaction\n", gflops, FLOPS_PER_INTERACTION);
    printf("\n[Validation] Positions finite: %s\n", pboResult.valid ? "PASS" : "FAIL");

    return pboResult.valid ? 0 : -1;
}