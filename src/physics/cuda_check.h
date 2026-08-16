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
 * N-Body GPU Simulator - CUDA 错误检查宏（替代 NVIDIA helper_cuda）
 */

#ifndef NBODY_PHYSICS_CUDA_CHECK_H
#define NBODY_PHYSICS_CUDA_CHECK_H

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// 检查 CUDA 调用，失败时打印文件/行号并退出
inline void checkCudaErrorsImpl(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

// 检查 CUDA 内核启动错误（异步错误需 cudaGetLastError 捕获）
inline void checkCudaKernelError(const char* file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Kernel Error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

// 检查 CUDA 同步后的运行时错误
inline void checkCudaSyncError(const char* file, int line)
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Sync Error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) checkCudaErrorsImpl((call), __FILE__, __LINE__)
#define CHECK_CUDA_KERNEL() checkCudaKernelError(__FILE__, __LINE__)
#define CHECK_CUDA_SYNC() checkCudaSyncError(__FILE__, __LINE__)

#endif  // NBODY_PHYSICS_CUDA_CHECK_H