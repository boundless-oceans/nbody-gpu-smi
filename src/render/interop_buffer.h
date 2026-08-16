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
 * N-Body GPU Simulator - S3-1: CUDA-GL Interop 缓冲封装
 * PBO(像素缓冲对象) 由 OpenGL 创建，注册给 CUDA 后由 CUDA 直接读写，
 * OpenGL 再用同一 PBO 渲染 → 全程 GPU↔GPU，零 CPU 拷贝。
 */

#ifndef NBODY_RENDER_INTEROP_BUFFER_H
#define NBODY_RENDER_INTEROP_BUFFER_H

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// CUDA-GL Interop 缓冲（单缓冲版本，S3-1 验证用）
class InteropBuffer
{
public:
    // numParticles: 粒子数；vec4Size: 每粒子字节数（位置 xyz+质量 = 4×float）
    InteropBuffer(unsigned int numParticles, size_t vec4Size);
    ~InteropBuffer();

    InteropBuffer(const InteropBuffer&)            = delete;
    InteropBuffer& operator=(const InteropBuffer&) = delete;

    // 锁定 PBO 供 CUDA 映射访问（返回可直接写的设备指针）
    cudaGraphicsResource* mapForCuda();

    // 解锁 PBO（CUDA 写完后必须调用，GL 才能安全读取）
    void unmapFromCuda();

    // 查询：PBO 句柄 / 粒子数 / 每粒子字节数
    unsigned int getPBO() const { return m_pbo; }
    unsigned int getNumParticles() const { return m_numParticles; }
    size_t       getVec4Size() const { return m_vec4Size; }

    // 是否已正确初始化（注册成功）
    bool isValid() const { return m_bValid; }

private:
    unsigned int m_pbo;
    unsigned int m_numParticles;
    size_t       m_vec4Size;
    cudaGraphicsResource* m_cudaResource;
    bool                  m_bValid;
};

#endif  // NBODY_RENDER_INTEROP_BUFFER_H