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
 * N-Body GPU Simulator - S3-3: 粒子渲染器实现
 * - 编译 GLSL 330 点精灵着色器
 * - VAO 直接绑定 PBO 作为顶点源（每粒子 4×float = xyz + mass）
 * - 手写 model/view/projection 矩阵（无第三方数学库，S3-3 固定视角）
 */

#include <GL/glew.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "particle_renderer.h"

// 基础 4x4 矩阵（列主序，兼容 OpenGL）
struct Mat4
{
    float m[16];

    static Mat4 identity()
    {
        Mat4 r = {};
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    // 透视投影（fovY 弧度）
    static Mat4 perspective(float fovY, float aspect, float znear, float zfar)
    {
        Mat4 r   = {};
        float f  = 1.0f / std::tan(fovY / 2.0f);
        r.m[0]   = f / aspect;
        r.m[5]   = f;
        r.m[10]  = (zfar + znear) / (znear - zfar);
        r.m[11]  = -1.0f;
        r.m[14]  = (2.0f * zfar * znear) / (znear - zfar);
        return r;
    }

    // 平移
    static Mat4 translation(float x, float y, float z)
    {
        Mat4 r = identity();
        r.m[12] = x;
        r.m[13] = y;
        r.m[14] = z;
        return r;
    }

    // 矩阵乘法（C = A * B）
    static Mat4 multiply(const Mat4& a, const Mat4& b)
    {
        Mat4 r = {};
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    sum += a.m[k * 4 + row] * b.m[col * 4 + k];
                }
                r.m[col * 4 + row] = sum;
            }
        }
        return r;
    }
};

// ---------------- GLSL 着色器 ----------------
// 顶点着色器：位置 → 裁剪空间，点大小随视距缩放
static const char* vertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec4 aPos;   // xyz + mass

uniform mat4 uMVP;
uniform float uPointScale;

void main()
{
    gl_Position = uMVP * vec4(aPos.xyz, 1.0);
    // 点大小随视距缩放（深度越大点越小）
    gl_PointSize = uPointScale / max(1.0, -gl_Position.z);
}
)";

// 片元着色器：圆形粒子（暖色）
static const char* fragmentShaderSrc = R"(
#version 330 core
uniform vec4 uColor;

out vec4 fragColor;

void main()
{
    // 圆点：距中心超过半径 → 丢弃边缘像素
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25)
        discard;
    fragColor = uColor;
}
)";

// ---------------- shader 编译辅助 ----------------
static GLuint compileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[ParticleRenderer] Shader compile error:\n%s\n", log);
        std::exit(EXIT_FAILURE);
    }
    return shader;
}

static GLuint linkProgram(GLuint vs, GLuint fs)
{
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint ok = GL_FALSE;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[ParticleRenderer] Program link error:\n%s\n", log);
        std::exit(EXIT_FAILURE);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

ParticleRenderer::ParticleRenderer()
    : m_vao(0)
    , m_vbo(0)
    , m_shader(0)
    , m_pbo(0)
    , m_numParticles(0)
    , m_pointScale(80.0f)
{
    buildShaders();

    // 创建 VAO + 占位 VBO（VAO 必须至少绑定一个 VBO 才能合法）
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    // 占位：VAO 记录属性格式，真实数据运行时由 setPBO 换成 PBO 提供
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

ParticleRenderer::~ParticleRenderer()
{
    if (m_shader) glDeleteProgram(m_shader);
    if (m_vbo)     glDeleteBuffers(1, &m_vbo);
    if (m_vao)     glDeleteVertexArrays(1, &m_vao);
}

void ParticleRenderer::buildShaders()
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    m_shader  = linkProgram(vs, fs);
}

void ParticleRenderer::setPBO(unsigned int pbo, unsigned int numParticles)
{
    m_pbo          = pbo;
    m_numParticles = numParticles;
}

void ParticleRenderer::render()
{
    if (m_pbo == 0 || m_numParticles == 0) {
        return;
    }

    glUseProgram(m_shader);

    // ---- MVP 矩阵（S3-3 固定视角） ----
    // 视图矩阵 = 相机在 (0,0,-100)，看向原点
    const float viewDist = -100.0f;
    const float aspect   = 1280.0f / 720.0f;

    Mat4 proj  = Mat4::perspective(60.0f * 3.14159265f / 180.0f, aspect, 0.1f, 1000.0f);
    Mat4 view  = Mat4::translation(0.0f, 0.0f, viewDist);
    Mat4 model = Mat4::identity();
    Mat4 mvp   = Mat4::multiply(Mat4::multiply(proj, view), model);

    glUniformMatrix4fv(glGetUniformLocation(m_shader, "uMVP"), 1, GL_FALSE, mvp.m);
    glUniform1f(glGetUniformLocation(m_shader, "uPointScale"), m_pointScale);
    glUniform4f(glGetUniformLocation(m_shader, "uColor"), 1.0f, 0.6f, 0.3f, 1.0f);

    // ---- 绑定 PBO 作为顶点数据源 ----
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_pbo);
    // 重新关联 PBO 到 VAO 的 attribute 0（每次渲染前绑定，因为 PBO 可能换）
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glDrawArrays(GL_POINTS, 0, m_numParticles);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}