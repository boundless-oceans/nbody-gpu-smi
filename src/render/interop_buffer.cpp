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
 * N-Body GPU Simulator - S3-1: CUDA-GL Interop 缓冲实现
 */

#include <GL/glew.h>

#include <cstdio>
#include <cstdlib>

#include "interop_buffer.h"

// 检查 GL / CUDA 错误（S3 阶段局部错误处理）
namespace {

void checkGlError(const char* tag)
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::fprintf(stderr, "[InteropBuffer] OpenGL Error [%s]: 0x%04X\n", tag, err);
        std::exit(EXIT_FAILURE);
    }
}

void checkCudaError(cudaError_t err, const char* tag)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[InteropBuffer] CUDA Error [%s]: %s\n", tag, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

InteropBuffer::InteropBuffer(unsigned int numParticles, size_t vec4Size)
    : m_pbo(0)
    , m_numParticles(numParticles)
    , m_vec4Size(vec4Size)
    , m_cudaResource(nullptr)
    , m_bValid(false)
{
    size_t bufferSize = numParticles * vec4Size;

    // 1. 创建 PBO（OpenGL 像素缓冲对象）
    glGenBuffers(1, &m_pbo);
    checkGlError("glGenBuffers");

    glBindBuffer(GL_ARRAY_BUFFER, m_pbo);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
    checkGlError("glBufferData");

    // 验证 PBO 分配大小
    GLint size = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
    if ((GLsizeiptr)size != bufferSize) {
        std::fprintf(stderr, "[InteropBuffer] PBO allocation failed: requested %zu, got %d\n", bufferSize, size);
        std::exit(EXIT_FAILURE);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // 2. 注册 PBO 给 CUDA（核心 interop 操作）
    checkCudaError(
        cudaGraphicsGLRegisterBuffer(&m_cudaResource, m_pbo, cudaGraphicsMapFlagsWriteDiscard),
        "cudaGraphicsGLRegisterBuffer");

    m_bValid = true;
    std::printf("[InteropBuffer] PBO registered: %u particles, %zu bytes/particle, total %zu bytes\n",
                m_numParticles, m_vec4Size, bufferSize);
}

InteropBuffer::~InteropBuffer()
{
    if (m_cudaResource) {
        cudaGraphicsUnregisterResource(m_cudaResource);
        m_cudaResource = nullptr;
    }

    if (m_pbo) {
        glDeleteBuffers(1, &m_pbo);
        m_pbo = 0;
    }
}

cudaGraphicsResource* InteropBuffer::mapForCuda()
{
    if (!m_bValid || !m_cudaResource) {
        std::fprintf(stderr, "[InteropBuffer] mapForCuda called on invalid buffer\n");
        std::exit(EXIT_FAILURE);
    }

    // 锁定 PBO，禁止 GL 访问，返回 CUDA 可写指针
    checkCudaError(cudaGraphicsMapResources(1, &m_cudaResource, 0), "cudaGraphicsMapResources");

    return m_cudaResource;
}

void InteropBuffer::unmapFromCuda()
{
    if (!m_bValid || !m_cudaResource) {
        std::fprintf(stderr, "[InteropBuffer] unmapFromCuda called on invalid buffer\n");
        std::exit(EXIT_FAILURE);
    }

    // 解锁 PBO，归还给 GL 使用
    checkCudaError(cudaGraphicsUnmapResources(1, &m_cudaResource, 0), "cudaGraphicsUnmapResources");
}

void InteropBuffer::getMappedDevicePointer(void** devicePtr, size_t* bytes) const
{
    if (!m_bValid || !m_cudaResource) {
        std::fprintf(stderr, "[InteropBuffer] getMappedDevicePointer called on invalid buffer\n");
        std::exit(EXIT_FAILURE);
    }

    // 返回 CUDA 映射后的设备指针（仅 mapForCuda 之后有效）
    checkCudaError(cudaGraphicsResourceGetMappedPointer(devicePtr, bytes, m_cudaResource),
                   "cudaGraphicsResourceGetMappedPointer");
}
