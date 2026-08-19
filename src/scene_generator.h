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
 * N-Body GPU Simulator - S6: 场景生成器
 * 核心物理场景（默认）：
 *   RANDOM_CLOUD  随机粒子云
 *   BINARY_PAIR   双粒子互绕
 *   LATTICE_PERTURB 格点扰动
 * 宇宙彩蛋场景：
 *   BINARY_STAR   双星系统
 *   GALAXY_DISK   旋转星系盘
 *   SOLAR_SYSTEM  太阳系
 * 质量模式：
 *   UNIFORM       均匀质量
 *   RANDOM        随机质量（可调变异幅度）
 *   CENTRAL_BODY  中心天体（几个大质量恒星 + 小粒子）
 */

#ifndef NBODY_SCENE_GENERATOR_H
#define NBODY_SCENE_GENERATOR_H

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

// ---------- 场景配置 ----------
enum class SceneType {
    RANDOM_CLOUD,
    BINARY_PAIR,
    LATTICE_PERTURB,
    BINARY_STAR,
    GALAXY_DISK,
    SOLAR_SYSTEM,
    COUNT
};

enum class MassMode {
    UNIFORM,
    RANDOM,
    CENTRAL_BODY,
    COUNT
};

// 场景生成参数
struct SceneConfig
{
    SceneType scene        = SceneType::RANDOM_CLOUD;
    MassMode  massMode     = MassMode::UNIFORM;
    int       numBodies    = 16384;
    float     massSpread   = 20.0f;   // RANDOM 模式：最大质量倍数
    float     clusterScale = 1.0f;    // 全局尺寸缩放
};

// 场景名称（UI 显示用）
inline const char* sceneName(SceneType s)
{
    switch (s) {
    case SceneType::RANDOM_CLOUD:     return "随机粒子云";
    case SceneType::BINARY_PAIR:      return "双粒子互绕";
    case SceneType::LATTICE_PERTURB:  return "格点扰动";
    case SceneType::BINARY_STAR:      return "双星系统（彩蛋）";
    case SceneType::GALAXY_DISK:      return "旋转星系盘（彩蛋）";
    case SceneType::SOLAR_SYSTEM:     return "太阳系（彩蛋）";
    default:                          return "未知";
    }
}

inline const char* massModeName(MassMode m)
{
    switch (m) {
    case MassMode::UNIFORM:      return "均匀";
    case MassMode::RANDOM:       return "随机分布";
    case MassMode::CENTRAL_BODY: return "中心天体";
    default:                     return "未知";
    }
}

// ---------- 随机辅助 ----------
inline float rand01() { return rand() / (float)RAND_MAX; }
inline float randRange(float lo, float hi) { return lo + (hi - lo) * rand01(); }

// 在单位球内取均匀随机点（拒绝采样）
inline void randomUnitSphere(float& x, float& y, float& z)
{
    for (;;) {
        x = randRange(-1, 1);
        y = randRange(-1, 1);
        z = randRange(-1, 1);
        if (x * x + y * y + z * z <= 1.0f) return;
    }
}

// ---------- 场景生成主入口 ----------
// 填充 pos（每粒子 4×float = xyz + mass）与 vel（xyz 速度）
// 调用方负责 srand() 播种；返回最大质量（渲染映射用）
inline float generateScene(const SceneConfig& cfg, std::vector<float>& pos, std::vector<float>& vel)
{
    const int n = cfg.numBodies;
    pos.assign(n * 4, 0.0f);
    vel.assign(n * 4, 0.0f);

    const float scale = cfg.clusterScale;
    float       maxMass = 1.0f;

    // ============ 位置与速度生成（按场景） ============
    switch (cfg.scene) {
    case SceneType::RANDOM_CLOUD: {
        // 三维均匀随机球（质心静止）
        const float radius = 40.0f * scale;
        for (int i = 0; i < n; ++i) {
            float x, y, z;
            randomUnitSphere(x, y, z);
            float r = radius * std::cbrt(rand01());
            pos[i*4+0] = x * r;
            pos[i*4+1] = y * r;
            pos[i*4+2] = z * r;
            // 无轨道速度，纯引力坍缩（观察聚散）
        }
        break;
    }

    case SceneType::BINARY_PAIR: {
        // 两个大质量主星 + 环绕小粒子
        const float sep = 50.0f * scale;
        const float m1  = 200.0f;
        const float m2  = 200.0f;
        const int   numPlanets = n - 2;
        // 主星 1（左侧）
        pos[0*4+0] = -sep/2; pos[0*4+1] = 0; pos[0*4+2] = 0;
        vel[0*4+0] = 0; vel[0*4+1] = 0.5f; vel[0*4+2] = 0;
        // 主星 2（右侧，绕质心旋转）
        pos[1*4+0] =  sep/2; pos[1*4+1] = 0; pos[1*4+2] = 0;
        vel[1*4+0] = 0; vel[1*4+1] = -0.5f; vel[1*4+2] = 0;
        maxMass = m1;

        // 环绕小粒子：在双星周围随机轨道
        for (int i = 2; i < n; ++i) {
            float theta = randRange(0, 6.28318f);
            float phi   = randRange(0, 3.14159f);
            float rr    = randRange(sep/2 + 10.0f, sep/2 + 60.0f) * scale;
            pos[i*4+0] = rr * std::sin(phi) * std::cos(theta);
            pos[i*4+1] = rr * std::sin(phi) * std::sin(theta);
            pos[i*4+2] = rr * std::cos(phi);
            // 近似轨道速度：绕两星质心
            float v = std::sqrt((m1 + m2) / rr);
            vel[i*4+0] = -v * std::sin(theta) * std::sin(phi) * 0.3f;
            vel[i*4+1] = v * std::cos(phi) * 0.3f;
            vel[i*4+2] = 0.3f * v * std::sin(theta) * std::sin(phi);
        }
        break;
    }

    case SceneType::LATTICE_PERTURB: {
        // 规则格点 + 微小扰动
        int side = (int)std::ceil(std::cbrt((double)n));
        float spacing = 30.0f / std::max(side - 1, 1) * scale;
        int idx = 0;
        for (int ix = 0; ix < side && idx < n; ++ix)
        for (int iy = 0; iy < side && idx < n; ++iy)
        for (int iz = 0; iz < side && idx < n; ++iz, ++idx) {
            pos[idx*4+0] = (ix - side/2.0f) * spacing;
            pos[idx*4+1] = (iy - side/2.0f) * spacing;
            pos[idx*4+2] = (iz - side/2.0f) * spacing;
            // 微小速度扰动（占格点间隔一定比例）
            vel[idx*4+0] = randRange(-1, 1) * spacing * 0.05f;
            vel[idx*4+1] = randRange(-1, 1) * spacing * 0.05f;
            vel[idx*4+2] = randRange(-1, 1) * spacing * 0.05f;
        }
        break;
    }

    case SceneType::BINARY_STAR: {
        // 两颗质量不同恒星 + 少量尘埃
        const float sep = 45.0f * scale;
        const float m1  = 300.0f;
        const float m2  = 100.0f;
        pos[0*4+0] = -sep/2; pos[0*4+1] = 0; pos[0*4+2] = 0;
        vel[0*4+0] = 0; vel[0*4+1] = 0.6f; vel[0*4+2] = 0;
        pos[1*4+0] =  sep/2; pos[1*4+1] = 0; pos[1*4+2] = 0;
        vel[1*4+0] = 0; vel[1*4+1] = -1.8f; vel[1*4+2] = 0;
        maxMass = m1;

        for (int i = 2; i < n; ++i) {
            float theta = randRange(0, 6.28318f);
            float rr    = randRange(sep + 10.0f, sep + 50.0f) * scale;
            pos[i*4+0] = rr * std::cos(theta);
            pos[i*4+1] = rr * std::sin(theta);
            pos[i*4+2] = randRange(-5, 5) * scale;
            float v = std::sqrt((m1 + m2) / rr);
            vel[i*4+0] = -v * std::sin(theta);
            vel[i*4+1] = v * std::cos(theta);
            vel[i*4+2] = 0;
        }
        break;
    }

    case SceneType::GALAXY_DISK: {
        // 旋转盘 + 中心大质量核
        const float Rmax   = 55.0f * scale;
        const float nucleusMass = 2000.0f;
        maxMass = nucleusMass;

        // 中心核（少量大质量粒子）
        int nucleusCount = std::min(50, n);
        for (int i = 0; i < nucleusCount; ++i) {
            float x, y, z;
            randomUnitSphere(x, y, z);
            float r = 4.0f * scale;
            pos[i*4+0] = x * r; pos[i*4+1] = y * r; pos[i*4+2] = z * r;
            vel[i*4+0] = 0; vel[i*4+1] = 0; vel[i*4+2] = 0;
        }

        // 盘粒子：开普勒速度（内快外慢）
        for (int i = nucleusCount; i < n; ++i) {
            float theta = randRange(0, 6.28318f);
            float r = randRange(5, Rmax);
            float x = r * std::cos(theta) ;
            float y = r * std::sin(theta);
            pos[i*4+0] = x * 0.8f + nucleusMass * 0.0f;  // 绕中心
            pos[i*4+1] = y * 0.8f;
            pos[i*4+2] = randRange(-3, 3) * scale;
            float v = std::sqrt(nucleusMass / r);
            vel[i*4+0] = -std::sin(theta) * v;
            vel[i*4+1] = std::cos(theta) * v;
            vel[i*4+2] = 0;
        }
        // 整体不旋转，保持盘稳定
        break;
    }

    case SceneType::SOLAR_SYSTEM: {
        // 中心太阳 + 8 颗行星轨道（每行星轨道放若干粒子增强视觉）
        const float sunMass = 1000.0f;
        maxMass = sunMass;
        // 太阳
        pos[0*4+0] = 0; pos[0*4+1] = 0; pos[0*4+2] = 0;
        vel[0*4+0] = 0; vel[0*4+1] = 0; vel[0*4+2] = 0;

        // 8 条轨道，每条均匀分布粒子
        const float orbitR[] = {12, 18, 25, 33, 42, 52, 63, 75};
        const int   perOrbit = n / 8;
        for (int o = 0; o < 8; ++o) {
            const float r = orbitR[o] * scale;
            const float v = std::sqrt(sunMass / r);
            const int   cnt = (o == 7) ? (n - 1 - 7 * perOrbit) : perOrbit;
            for (int k = 0; k < cnt; ++k) {
                int idx = 1 + o * perOrbit + k;
                if (idx >= n) break;
                float theta = 6.28318f * k / cnt;
                pos[idx*4+0] = r * std::cos(theta);
                pos[idx*4+1] = r * std::sin(theta);
                pos[idx*4+2] = randRange(-0.3f, 0.3f) * scale;
                vel[idx*4+0] = -v * std::sin(theta);
                vel[idx*4+1] = v * std::cos(theta);
                vel[idx*4+2] = 0;
            }
        }
        break;
    }

    default: break;
    }

    // ============ 质量模式（覆盖 pos 的 mass 分量） ============
    switch (cfg.massMode) {
    case MassMode::UNIFORM: {
        for (int i = 0; i < n; ++i) pos[i*4+3] = 1.0f;
        break;
    }

    case MassMode::RANDOM: {
        for (int i = 0; i < n; ++i) {
            // 对数均匀分布：1 ~ massSpread
            float m = std::exp(rand01() * std::log(cfg.massSpread));
            pos[i*4+3] = m;
            if (m > maxMass) maxMass = m;
        }
        break;
    }

    case MassMode::CENTRAL_BODY: {
        // 默认：少量大质量体（中心） + 大量小质量
        const int numHeavy = std::max(1, n / 50);
        for (int i = 0; i < n; ++i) {
            pos[i*4+3] = (i < numHeavy) ? 200.0f : 1.0f;
            if (i < numHeavy && 200.0f > maxMass) maxMass = 200.0f;
        }
        break;
    }
    }

    return maxMass;
}

#endif  // NBODY_SCENE_GENERATOR_H