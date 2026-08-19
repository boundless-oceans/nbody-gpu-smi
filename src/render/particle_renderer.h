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
 * N-Body GPU Simulator - S3-3: 粒子渲染器（现代 OpenGL）
 * 通过 VAO 直接绑定 PBO（GL_ARRAY_BUFFER）作为顶点数据源，
 * 与 CUDA 计算结果零拷贝共享显存。GLSL 330 shader 点精灵绘制。
 */

#ifndef NBODY_RENDER_PARTICLE_RENDERER_H
#define NBODY_RENDER_PARTICLE_RENDERER_H

#include <cstddef>

// 粒子渲染器：属性布局 = 每粒子 4×float（xyz + mass）
class ParticleRenderer
{
public:
    ParticleRenderer();
    ~ParticleRenderer();

    ParticleRenderer(const ParticleRenderer&)            = delete;
    ParticleRenderer& operator=(const ParticleRenderer&) = delete;

    // 绑定要渲染的 PBO（数据在 CUDA 侧写好后调用 render 即可）
    // pbo: OpenGL PBO 句柄；numParticles: 粒子数
    void setPBO(unsigned int pbo, unsigned int numParticles);

    // 设置质量映射基准（log 归一化用）：粒子质量最大值
    // 质量越接近 maxMass → 颜色越红、粒子越大
    void setMaxMass(float maxMass);

    // 渲染一帧（绑定 PBO 数据 → GL_POINTS 绘制 → 解绑）
    void render();

private:
    void buildShaders();

private:
    unsigned int m_vao;        // 顶点数组对象
    unsigned int m_vbo;        // 占位 VBO（不持有数据，仅让 VAO 合法）
    unsigned int m_shader;     // 点精灵着色器程序
    unsigned int m_pbo;        // 当前绑定的 PBO（0 = 未设置）
    unsigned int m_numParticles;
    float        m_pointScale;  // 点大小缩放（视距相关，S3-3 固定值）
    float        m_maxMass;     // 质量映射基准（S5，log 归一化）
};

#endif  // NBODY_RENDER_PARTICLE_RENDERER_H