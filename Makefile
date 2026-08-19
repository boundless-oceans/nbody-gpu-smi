# N-Body GPU Simulator - 根目录 Makefile（CMake 封装）
# 与 NVIDIA CUDA Samples 一致：可执行文件输出到项目根目录
#
# 用法:
#   make              # 构建全部（默认 Release，自动检测 GPU 架构）
#   make nbody        # 仅构建 GUI 模拟器
#   make benchmark    # 仅构建性能基准
#   make run [N=100000]      # 运行 GUI 模拟器（可选指定额外参数）
#   make bench [N=24576]     # 运行性能基准（默认 24576 体 5 迭代）
#   make clean        # 清理构建目录与根目录产物
#   make rebuild      # 清理后重新构建

# ---------- 配置 ----------
BUILD_DIR  := build
GPU_ARCH   := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(GPU_ARCH),)
GPU_ARCH := 89
endif

BUILD_TYPE := Release

CMAKE := cmake
CMAKE_FLAGS := -B $(BUILD_DIR) -S . -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_CUDA_ARCHITECTURES=$(GPU_ARCH)
MAKE_FLAGS := -C $(BUILD_DIR) -j$(shell nproc)

# 根目录产物（CMake RUNTIME_OUTPUT_DIRECTORY = 项目根）
BIN_NBODY := nbody
BIN_BENCH := nbody_benchmark

# ---------- 目标 ----------
.PHONY: all configure nbody benchmark run bench clean rebuild

all: configure
	$(MAKE) $(MAKE_FLAGS)

# 仅配置（生成构建文件）
configure:
	$(CMAKE) $(CMAKE_FLAGS)

nbody: configure
	$(MAKE) $(MAKE_FLAGS) nbody

benchmark: configure
	$(MAKE) $(MAKE_FLAGS) nbody_benchmark

run: all
	./$(BIN_NBODY) $(RUN_ARGS)

bench: all
	./$(BIN_BENCH) $(if $(N),$(N),24576) $(if $(I),$(I),5)

clean:
	rm -rf $(BUILD_DIR) $(BIN_NBODY) $(BIN_BENCH)

rebuild: clean
	$(MAKE) all