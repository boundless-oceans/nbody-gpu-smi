#!/usr/bin/env bash
# 一键构建脚本（自动检测 GPU 架构）
# 用法: ./scripts/build.sh [--clean] [--debug]
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

# 检测 CUDA 架构
detect_arch() {
    if command -v nvidia-smi &>/dev/null; then
        local cap
        cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
        if [[ -n "$cap" ]]; then
            # compute_cap 形式如 89 / 86 / 75，即 sm_89
            echo "$cap"
            return 0
        fi
    fi
    echo -e "${YELLOW}警告: 无法自动检测 GPU 架构，使用默认 89 (RTX 40 系)${NC}" >&2
    echo "89"
}

# 解析参数
CLEAN=false
BUILD_TYPE="Release"
for arg in "$@"; do
    case $arg in
        --clean) CLEAN=true ;;
        --debug) BUILD_TYPE="Debug" ;;
        *) echo "未知参数: $arg" >&2; exit 1 ;;
    esac
done

ARCH="$(detect_arch)"
echo "=========================================="
echo "  N-Body GPU Simulator - 构建"
echo "=========================================="
echo -e "${GREEN}[INFO]${NC} GPU 架构: sm_$ARCH"
echo -e "${GREEN}[INFO]${NC} 构建类型: $BUILD_TYPE"
echo -e "${GREEN}[INFO]${NC} 构建目录: $BUILD_DIR"
echo "------------------------------------------"

if [[ "$CLEAN" == true ]]; then
    echo -e "${YELLOW}清理旧构建...${NC}"
    rm -rf "$BUILD_DIR"
fi

# 检查 CUDA 工具链
if ! command -v nvcc &>/dev/null; then
    echo -e "${RED}[错误]${NC} nvcc 未找到，请先运行 ./scripts/setup_env.sh"
    exit 1
fi

# 检查图形库
for pkg in libglfw3-dev libglew-dev; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        echo -e "${RED}[错误]${NC} 缺少 $pkg，请先运行 ./scripts/setup_env.sh --install"
        exit 1
    fi
done

cmake -B "$BUILD_DIR" -S "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH"

cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "------------------------------------------"
echo -e "${GREEN}✓ 构建完成！${NC}"
echo "  运行: ./scripts/run.sh"