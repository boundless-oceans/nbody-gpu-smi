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
 * N-Body GPU Simulator - CUDA 引力计算内核（共享内存 Tiling）
 * 基于 NVIDIA CUDA Samples nbody 精简迁移：
 *   - 剔除 GLUT / PBO / 多 GPU，仅保留核心物理计算
 *   - 保留: 共享内存 tiling、双缓冲、常量内存、cooperative groups、
 *          __restrict__、rsqrtf 快速指令、float/double 模板
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>

#include "body_system.h"
#include "cuda_check.h"

namespace cg = cooperative_groups;

// 常量内存：softening 平方（GPU 侧广播读取优化）
__constant__ float softeningSquared;
__constant__ double softeningSquared_fp64;

cudaError_t setSofteningSquared(float softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared, &softeningSq, sizeof(float), 0, cudaMemcpyHostToDevice);
}

cudaError_t setSofteningSquared(double softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared_fp64, &softeningSq, sizeof(double), 0, cudaMemcpyHostToDevice);
}

// 动态共享内存访问辅助
template <class T> struct SharedMemory
{
    __device__ inline operator T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// 快速倒数开方（float/double 特化）
template <typename T> __device__ T rsqrt_T(T x) { return rsqrt(x); }

template <> __device__ float rsqrt_T<float>(float x) { return rsqrtf(x); }

template <> __device__ double rsqrt_T<double>(double x) { return rsqrt(x); }

template <typename T> __device__ T getSofteningSquared() { return softeningSquared; }
template <> __device__ double getSofteningSquared<double>() { return softeningSquared_fp64; }

// 单对粒子引力交互（GPU Gems 3 算法核心，含 FLOPS 注释）
template <typename T>
__device__ typename vec3<T>::Type
bodyBodyInteraction(typename vec3<T>::Type ai, typename vec4<T>::Type bi, typename vec4<T>::Type bj)
{
    typename vec3<T>::Type r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    T distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += getSofteningSquared<T>();

    // invDistCube = 1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    T invDist     = rsqrt_T(distSqr);
    T invDistCube = invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = bj.w * invDistCube;

    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

// 计算单个天体的加速度（共享内存 tiling：每个 block 协作加载 tile 复用）
template <typename T>
__device__ typename vec3<T>::Type
computeBodyAccel(typename vec4<T>::Type bodyPos, typename vec4<T>::Type* positions, int numTiles, cg::thread_block cta)
{
    typename vec4<T>::Type* sharedPos = SharedMemory<typename vec4<T>::Type>();

    typename vec3<T>::Type acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++) {
        sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

        // 协作加载后同步，确保整个 tile 就绪
        cg::sync(cta);

// 来自 GPU Gems 3 的 tile_calculation 循环展开
#pragma unroll 128

        for (unsigned int counter = 0; counter < blockDim.x; counter++) {
            acc = bodyBodyInteraction<T>(acc, bodyPos, sharedPos[counter]);
        }

        // 计算完当前 tile 后同步，避免覆盖共享内存
        cg::sync(cta);
    }

    return acc;
}

// 主内核：计算每个天体的加速度 → 更新速度/位置（双缓冲读写分离）
template <typename T>
__global__ void integrateBodies(typename vec4<T>::Type* __restrict__ newPos,
                                typename vec4<T>::Type* __restrict__ oldPos,
                                typename vec4<T>::Type* vel,
                                unsigned int            numBodies,
                                float                   deltaTime,
                                float                   damping,
                                int                     numTiles)
{
    // 处理到线程块分组
    cg::thread_block cta   = cg::this_thread_block();
    int              index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numBodies) {
        return;
    }

    typename vec4<T>::Type position = oldPos[index];

    typename vec3<T>::Type accel = computeBodyAccel<T>(position, oldPos, numTiles, cta);

    // acceleration = force / mass（质量已约去）
    // new velocity = old velocity + acceleration * deltaTime
    typename vec4<T>::Type velocity = vel[index];

    velocity.x += accel.x * deltaTime;
    velocity.y += accel.y * deltaTime;
    velocity.z += accel.z * deltaTime;

    // 阻尼
    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * deltaTime
    position.x += velocity.x * deltaTime;
    position.y += velocity.y * deltaTime;
    position.z += velocity.z * deltaTime;

    // 写入新位置和新速度
    newPos[index] = position;
    vel[index]    = velocity;
}

// 集成入口（单 GPU，无条件 PBO / 多设备路径）
template <typename T>
void integrateNbodySystem(typename vec4<T>::Type* dPosOld,
                          typename vec4<T>::Type* dPosNew,
                          typename vec4<T>::Type* dVel,
                          unsigned int            numBodies,
                          float                   deltaTime,
                          float                   damping,
                          int                     blockSize)
{
    int numBlocks = (numBodies + blockSize - 1) / blockSize;
    int numTiles  = (numBodies + blockSize - 1) / blockSize;
    int sharedMemSize = blockSize * 4 * sizeof(T);  // 4 分量位置

    integrateBodies<T>
        <<<numBlocks, blockSize, sharedMemSize>>>(dPosNew, dPosOld, dVel, numBodies, deltaTime, damping, numTiles);

    CHECK_CUDA_KERNEL();
}

// 显式实例化 float / double
template void integrateNbodySystem<float>(float4* dPosOld,
                                          float4* dPosNew,
                                          float4* dVel,
                                          unsigned int numBodies,
                                          float deltaTime,
                                          float damping,
                                          int blockSize);

template void integrateNbodySystem<double>(double4* dPosOld,
                                           double4* dPosNew,
                                           double4* dVel,
                                           unsigned int numBodies,
                                           float deltaTime,
                                           float damping,
                                           int blockSize);