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
 * N-Body GPU Simulator - 单 GPU CUDA 物理系统
 * 基于 NVIDIA CUDA Samples nbody 精简迁移：
 *   - 剔除 GLUT / 多 GPU / 系统内存路径
 *   - 保留: 双缓冲、常量内存 softening、update 循环、get/setArray
 *   - S3-2: 新增 PBO 模式（位置缓冲由 CUDA-GL Interop 提供，供渲染零拷贝）
 */

#ifndef NBODY_PHYSICS_BODY_SYSTEM_CUDA_H
#define NBODY_PHYSICS_BODY_SYSTEM_CUDA_H

#include <cassert>
#include <cstring>
#include <memory>

#include <cuda_runtime.h>

#include "body_system.h"
#include "cuda_check.h"
#include "render/interop_buffer.h"

// 内核入口（在 body_system_cuda_kernel.cu 中实现并显式实例化）
template <typename T>
void integrateNbodySystem(typename vec4<T>::Type* dPosOld,
                          typename vec4<T>::Type* dPosNew,
                          typename vec4<T>::Type* dVel,
                          unsigned int            numBodies,
                          float                   deltaTime,
                          float                   damping,
                          int                     blockSize);

// 设置 softening 到常量内存
cudaError_t setSofteningSquared(float softeningSq);
cudaError_t setSofteningSquared(double softeningSq);

// CUDA 物理系统（单 GPU）
// usePBO=true 时：位置缓冲由 InteropBuffer(PBO) 提供，CUDA 直接写进
//   OpenGL 可渲染的缓冲（需已有 GL context）；速度缓冲仍 cudaMalloc。
template <typename T> class BodySystemCUDA : public BodySystem<T>
{
public:
    explicit BodySystemCUDA(unsigned int numBodies, unsigned int blockSize = 256, bool usePBO = false)
        : m_numBodies(numBodies)
        , m_blockSize(blockSize)
        , m_bInitialized(false)
        , m_usePBO(usePBO)
        , m_damping(0.995f)
        , m_softening(0.0f)
        , m_hPos(0)
        , m_hVel(0)
        , m_dPos{0, 0}
        , m_dVel(0)
        , m_posPBO{nullptr, nullptr}
        , m_currentRead(0)
        , m_currentWrite(1)
    {
        _initialize(numBodies);
        setSoftening(0.00125f);
        setDamping(0.995f);
    }

    ~BodySystemCUDA() { _finalize(); }

    // 推进一个时间步
    void update(T deltaTime) override
    {
        assert(m_bInitialized);

        if (m_usePBO) {
            // ---- PBO 模式：位置读写缓冲来自 InteropBuffer ----
            // 1. 锁定两个 PBO（写缓冲带 discard 语义，通知 GL 整个内容重写）
            m_posPBO[m_currentRead]->mapForCuda();
            m_posPBO[m_currentWrite]->mapForCuda();

            // 2. 获取 CUDA 可用的设备指针
            void* posOldPtr = nullptr;
            void* posNewPtr = nullptr;
            size_t bytes   = 0;
            m_posPBO[m_currentRead]->getMappedDevicePointer(&posOldPtr, &bytes);
            m_posPBO[m_currentWrite]->getMappedDevicePointer(&posNewPtr, &bytes);

            // 3. 内核计算（位置直接读写 PBO 显存）
            integrateNbodySystem<T>(reinterpret_cast<typename vec4<T>::Type*>(posOldPtr),
                                    reinterpret_cast<typename vec4<T>::Type*>(posNewPtr),
                                    m_dVel,
                                    m_numBodies,
                                    (float)deltaTime,
                                    (float)m_damping,
                                    m_blockSize);

            // 4. 解锁 PBO，归还给 GL
            m_posPBO[m_currentRead]->unmapFromCuda();
            m_posPBO[m_currentWrite]->unmapFromCuda();
        }
        else {
            // ---- 普通模式：位置缓冲为 cudaMalloc ----
            integrateNbodySystem<T>(m_dPos[m_currentRead],
                                    m_dPos[m_currentWrite],
                                    m_dVel,
                                    m_numBodies,
                                    (float)deltaTime,
                                    (float)m_damping,
                                    m_blockSize);
        }

        // 交换读写缓冲角色
        unsigned int tmp = m_currentRead;
        m_currentRead    = m_currentWrite;
        m_currentWrite   = tmp;
    }

    void setSoftening(T softening) override
    {
        T softeningSq = softening * softening;
        CHECK_CUDA(setSofteningSquared(softeningSq));
    }

    void setDamping(T damping) override { m_damping = damping; }

    // 拷贝 GPU 数据回 CPU 宿主内存
    T* getArray(BodyArray array) override
    {
        assert(m_bInitialized);

        T* hdata = 0;

        switch (array) {
        default:
        case BODYSYSTEM_POSITION:
            hdata = m_hPos;
            if (m_usePBO) {
                m_posPBO[m_currentRead]->mapForCuda();
                void* dptr = nullptr;
                size_t bytes = 0;
                m_posPBO[m_currentRead]->getMappedDevicePointer(&dptr, &bytes);
                CHECK_CUDA(cudaMemcpy(hdata, dptr, m_numBodies * 4 * sizeof(T), cudaMemcpyDeviceToHost));
                m_posPBO[m_currentRead]->unmapFromCuda();
            }
            else {
                CHECK_CUDA(cudaMemcpy(
                    hdata, m_dPos[m_currentRead], m_numBodies * 4 * sizeof(T), cudaMemcpyDeviceToHost));
            }
            break;

        case BODYSYSTEM_VELOCITY:
            hdata = m_hVel;
            CHECK_CUDA(cudaMemcpy(hdata, m_dVel, m_numBodies * 4 * sizeof(T), cudaMemcpyDeviceToHost));
            break;
        }

        return hdata;
    }

    // 从 CPU 宿主内存上传数据到 GPU
    void setArray(BodyArray array, const T* data) override
    {
        assert(m_bInitialized);

        // 重置缓冲状态（N 变化时由外部重新生成场景）
        m_currentRead  = 0;
        m_currentWrite = 1;

        switch (array) {
        default:
        case BODYSYSTEM_POSITION:
            if (m_usePBO) {
                m_posPBO[m_currentRead]->mapForCuda();
                void* dptr = nullptr;
                size_t bytes = 0;
                m_posPBO[m_currentRead]->getMappedDevicePointer(&dptr, &bytes);
                CHECK_CUDA(cudaMemcpy(dptr, data, m_numBodies * 4 * sizeof(T), cudaMemcpyHostToDevice));
                m_posPBO[m_currentRead]->unmapFromCuda();
            }
            else {
                CHECK_CUDA(cudaMemcpy(
                    m_dPos[m_currentRead], data, m_numBodies * 4 * sizeof(T), cudaMemcpyHostToDevice));
            }
            break;

        case BODYSYSTEM_VELOCITY:
            CHECK_CUDA(cudaMemcpy(m_dVel, data, m_numBodies * 4 * sizeof(T), cudaMemcpyHostToDevice));
            break;
        }
    }

    unsigned int getNumBodies() const override { return m_numBodies; }

    // 当前 PBO 位置缓冲的句柄（S3-3 渲染用；仅 PBO 模式下有效）
    unsigned int getCurrentReadPBO() const
    {
        return m_usePBO ? m_posPBO[m_currentRead]->getPBO() : 0;
    }

    bool usesPBO() const { return m_usePBO; }

private:
    void _initialize(int numBodies)
    {
        assert(!m_bInitialized);
        m_numBodies = numBodies;

        unsigned int memSize = sizeof(T) * 4 * numBodies;

        // 宿主内存（初始场景/读取结果用）
        m_hPos = new T[numBodies * 4];
        m_hVel = new T[numBodies * 4];
        std::memset(m_hPos, 0, memSize);
        std::memset(m_hVel, 0, memSize);

        // 速度缓冲始终 cudaMalloc（PBO 只承载位置）
        CHECK_CUDA(cudaMalloc((void**)&m_dVel, memSize));

        if (m_usePBO) {
            // PBO 模式：位置由两个 InteropBuffer 提供（双缓冲）
            m_posPBO[0] = std::make_unique<InteropBuffer>(m_numBodies, 4 * sizeof(T));
            m_posPBO[1] = std::make_unique<InteropBuffer>(m_numBodies, 4 * sizeof(T));
        }
        else {
            // 普通模式：位置用 cudaMalloc 双缓冲
            CHECK_CUDA(cudaMalloc((void**)&m_dPos[0], memSize));
            CHECK_CUDA(cudaMalloc((void**)&m_dPos[1], memSize));
        }

        m_bInitialized = true;
    }

    void _finalize()
    {
        assert(m_bInitialized);

        delete[] m_hPos;
        delete[] m_hVel;

        // PBO 模式：InteropBuffer 析构自动 unregister + 删 PBO
        m_posPBO[0].reset();
        m_posPBO[1].reset();

        // 普通模式：释放 cudaMalloc 的缓冲
        if (!m_usePBO) {
            CHECK_CUDA(cudaFree(m_dPos[0]));
            CHECK_CUDA(cudaFree(m_dPos[1]));
        }
        CHECK_CUDA(cudaFree(m_dVel));

        m_bInitialized = false;
    }

private:
    unsigned int m_numBodies;
    unsigned int m_blockSize;
    bool         m_bInitialized;
    bool         m_usePBO;

    T m_damping;
    T m_softening;

    // 宿主内存
    T* m_hPos;
    T* m_hVel;

    // GPU 内存（普通模式：位置双缓冲 + 速度）
    typename vec4<T>::Type* m_dPos[2];
    typename vec4<T>::Type* m_dVel;

    // PBO 模式：位置双缓冲（InteropBuffer 内部持有 PBO + CUDA 资源）
    std::unique_ptr<InteropBuffer> m_posPBO[2];

    unsigned int m_currentRead;
    unsigned int m_currentWrite;
};

#endif  // NBODY_PHYSICS_BODY_SYSTEM_CUDA_H