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
 * N-Body GPU Simulator - Benchmark 入口（S2 验证点）
 * 用法: ./nbody_benchmark [numBodies] [iterations]
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
static constexpr int DEFAULT_NUM_BODIES   = 24576;
static constexpr int DEFAULT_ITERATIONS   = 5;
static constexpr int DEFAULT_BLOCK_SIZE   = 256;
static constexpr float TIMESTEP           = 0.016f;
static constexpr float CLUSTER_SCALE      = 1.54f;
static constexpr float VELOCITY_SCALE     = 8.0f;
static constexpr int   FLOPS_PER_INTERACTION = 20;

// 单精度性能统计（与 NVIDIA 版 computePerfStats 一致）
void computePerfStats(int numBodies, int iterations, float milliseconds, double& interactionsPerSecond, double& gflops)
{
    interactionsPerSecond = (float)numBodies * (float)numBodies;
    interactionsPerSecond *= 1e-9 * iterations * 1000 / milliseconds;
    gflops = interactionsPerSecond * (float)FLOPS_PER_INTERACTION;
}

int main(int argc, char** argv)
{
    int numBodies = DEFAULT_NUM_BODIES;
    int iterations = DEFAULT_ITERATIONS;

    if (argc > 1) {
        numBodies = std::atoi(argv[1]);
    }
    if (argc > 2) {
        iterations = std::atoi(argv[2]);
    }

    printf("> CUDA N-Body Benchmark\n");
    printf("> Single precision floating point simulation\n");
    printf("> %d bodies, %d iterations, blockSize=%d\n\n", numBodies, iterations, DEFAULT_BLOCK_SIZE);

    // 打印 GPU 信息
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return -1;
    }
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
    printf("GPU Device 0: \"%s\" with compute capability %d.%d\n", props.name, props.major, props.minor);

    // 创建物理系统
    BodySystemCUDA<float> system(numBodies, DEFAULT_BLOCK_SIZE);
    system.setSoftening(0.00125f);
    system.setDamping(0.995f);

    // 生成初始场景（随机）
    srand(42);
    std::vector<float> hPos(numBodies * 4);
    std::vector<float> hVel(numBodies * 4);
    std::vector<float> hColor(numBodies * 4);

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

    // 预热一次（加载设备、编译 kernel）
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

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

    double interactionsPerSecond = 0;
    double gflops                = 0;
    computePerfStats(numBodies, iterations, milliseconds, interactionsPerSecond, gflops);

    printf("\n%d bodies, total time for %d iterations: %.3f ms\n", numBodies, iterations, milliseconds);
    printf("= %.3f billion interactions per second\n", interactionsPerSecond);
    printf("= %.3f single-precision GFLOP/s at %d flops per interaction\n", gflops, FLOPS_PER_INTERACTION);

    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));

    // 正确性冒烟测试：验证能量有限非 NaN
    float* pos = system.getArray(BODYSYSTEM_POSITION);
    bool  valid = true;
    for (int i = 0; i < numBodies * 4 && valid; ++i) {
        valid = std::isfinite(pos[i]);
    }
    printf("\n[Validation] Positions finite: %s\n", valid ? "PASS" : "FAIL");
    return valid ? 0 : -1;
}