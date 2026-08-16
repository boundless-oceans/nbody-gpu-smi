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
 * N-Body GPU Simulator - 物理模块（基于 NVIDIA CUDA Samples nbody 精简迁移）
 * 剔除 GLUT / 渲染 / 多 GPU 依赖，保留核心物理内核接口。
 */

#ifndef NBODY_PHYSICS_BODY_SYSTEM_H
#define NBODY_PHYSICS_BODY_SYSTEM_H

#include <algorithm>
#include <cmath>
#include <vector_types.h>

// 初始场景配置
enum NBodyConfig { NBODY_CONFIG_RANDOM, NBODY_CONFIG_SHELL, NBODY_CONFIG_EXPAND, NBODY_NUM_CONFIGS };

// 体数据数组类型
enum BodyArray {
    BODYSYSTEM_POSITION,
    BODYSYSTEM_VELOCITY,
};

// ---------- vec3 / vec4 类型映射（float3/float4/double3/double4） ----------
template <typename T> struct vec3
{
    typedef float Type;
};  // dummy
template <> struct vec3<float>
{
    typedef float3 Type;
};
template <> struct vec3<double>
{
    typedef double3 Type;
};

template <typename T> struct vec4
{
    typedef float Type;
};  // dummy
template <> struct vec4<float>
{
    typedef float4 Type;
};
template <> struct vec4<double>
{
    typedef double4 Type;
};

// 位置/速度数组均为 4 分量（xyz + mass），质量在 float4.w
static constexpr int VEC4_STRIDE = 4;

// ---------- 抽象基类 ----------
template <typename T> class BodySystem
{
public:
    BodySystem() {}
    virtual ~BodySystem() {}

    virtual void update(T deltaTime) = 0;

    virtual void setSoftening(T softening) = 0;
    virtual void setDamping(T damping)     = 0;

    virtual T   *getArray(BodyArray array)                = 0;
    virtual void setArray(BodyArray array, const T *data) = 0;

    virtual unsigned int getNumBodies() const = 0;
};

// ---------- 工具向量函数 ----------
inline float3 scalevec(float3 &vector, float scalar)
{
    float3 rt = vector;
    rt.x *= scalar;
    rt.y *= scalar;
    rt.z *= scalar;
    return rt;
}

inline float normalize(float3 &vector)
{
    float dist = sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);

    if (dist > 1e-6) {
        vector.x /= dist;
        vector.y /= dist;
        vector.z /= dist;
    }

    return dist;
}

inline float dot(float3 v0, float3 v1) { return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z; }

inline float3 cross(float3 v0, float3 v1)
{
    float3 rt;
    rt.x = v0.y * v1.z - v0.z * v1.y;
    rt.y = v0.z * v1.x - v0.x * v1.z;
    rt.z = v0.x * v1.y - v0.y * v1.x;
    return rt;
}

// ---------- 初始场景随机生成（S6 场景预设的基础） ----------
template <typename T>
void randomizeBodies(NBodyConfig config,
                     T          *pos,
                     T          *vel,
                     float      *color,
                     float       clusterScale,
                     float       velocityScale,
                     int         numBodies,
                     bool        vec4vel)
{
    switch (config) {
    default:
    case NBODY_CONFIG_RANDOM: {
        float scale  = clusterScale * std::max<float>(1.0f, numBodies / (1024.0f));
        float vscale = velocityScale * scale;

        int p = 0, v = 0;
        int i = 0;

        while (i < numBodies) {
            float3 point;
            point.x      = rand() / (float)RAND_MAX * 2 - 1;
            point.y      = rand() / (float)RAND_MAX * 2 - 1;
            point.z      = rand() / (float)RAND_MAX * 2 - 1;
            float lenSqr = dot(point, point);

            if (lenSqr > 1)
                continue;

            float3 velocity;
            velocity.x = rand() / (float)RAND_MAX * 2 - 1;
            velocity.y = rand() / (float)RAND_MAX * 2 - 1;
            velocity.z = rand() / (float)RAND_MAX * 2 - 1;
            lenSqr     = dot(velocity, velocity);

            if (lenSqr > 1)
                continue;

            pos[p++] = point.x * scale;  // pos.x
            pos[p++] = point.y * scale;  // pos.y
            pos[p++] = point.z * scale;  // pos.z
            pos[p++] = 1.0f;             // mass

            vel[v++] = velocity.x * vscale;  // vel.x
            vel[v++] = velocity.y * vscale;  // vel.y
            vel[v++] = velocity.z * vscale;  // vel.z

            if (vec4vel)
                vel[v++] = 1.0f;  // inverse mass

            i++;
        }
    } break;

    case NBODY_CONFIG_SHELL: {
        float scale  = clusterScale;
        float vscale = scale * velocityScale;
        float inner  = 2.5f * scale;
        float outer  = 4.0f * scale;

        int p = 0, v = 0;
        int i = 0;

        while (i < numBodies) {
            float x, y, z;
            x = rand() / (float)RAND_MAX * 2 - 1;
            y = rand() / (float)RAND_MAX * 2 - 1;
            z = rand() / (float)RAND_MAX * 2 - 1;

            float3 point = {x, y, z};
            float  len   = normalize(point);

            if (len > 1)
                continue;

            pos[p++] = point.x * (inner + (outer - inner) * rand() / (float)RAND_MAX);
            pos[p++] = point.y * (inner + (outer - inner) * rand() / (float)RAND_MAX);
            pos[p++] = point.z * (inner + (outer - inner) * rand() / (float)RAND_MAX);
            pos[p++] = 1.0f;

            x           = 0.0f;
            y           = 0.0f;
            z           = 1.0f;
            float3 axis = {x, y, z};
            normalize(axis);

            if (1 - dot(point, axis) < 1e-6) {
                axis.x = point.y;
                axis.y = point.x;
                normalize(axis);
            }

            float3 vv = {(float)pos[4 * i], (float)pos[4 * i + 1], (float)pos[4 * i + 2]};
            vv        = cross(vv, axis);
            vel[v++]  = vv.x * vscale;
            vel[v++]  = vv.y * vscale;
            vel[v++]  = vv.z * vscale;

            if (vec4vel)
                vel[v++] = 1.0f;

            i++;
        }
    } break;

    case NBODY_CONFIG_EXPAND: {
        float scale = clusterScale * numBodies / (1024.f);

        if (scale < 1.0f)
            scale = clusterScale;

        float vscale = scale * velocityScale;

        int p = 0, v = 0;

        for (int i = 0; i < numBodies;) {
            float3 point;

            point.x = rand() / (float)RAND_MAX * 2 - 1;
            point.y = rand() / (float)RAND_MAX * 2 - 1;
            point.z = rand() / (float)RAND_MAX * 2 - 1;

            float lenSqr = dot(point, point);

            if (lenSqr > 1)
                continue;

            pos[p++] = point.x * scale;   // pos.x
            pos[p++] = point.y * scale;   // pos.y
            pos[p++] = point.z * scale;   // pos.z
            pos[p++] = 1.0f;              // mass
            vel[v++] = point.x * vscale;  // vel.x
            vel[v++] = point.y * vscale;  // vel.y
            vel[v++] = point.z * vscale;  // vel.z

            if (vec4vel)
                vel[v++] = 1.0f;  // inverse mass

            i++;
        }
    } break;
    }

    if (color) {
        int v = 0;

        for (int i = 0; i < numBodies; i++) {
            color[v++] = rand() / (float)RAND_MAX;
            color[v++] = rand() / (float)RAND_MAX;
            color[v++] = rand() / (float)RAND_MAX;
            color[v++] = 1.0f;
        }
    }
}

#endif  // NBODY_PHYSICS_BODY_SYSTEM_H