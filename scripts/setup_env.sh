#!/usr/bin/env bash
# 环境检查与依赖安装脚本（Ubuntu/Debian）
# 用法: ./scripts/setup_env.sh [--install]
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "  N-Body GPU Simulator - 环境检查"
echo "=========================================="

# 记录缺失项
MISSING=()

# 检查 CUDA
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+' | head -1)
    echo -e "${GREEN}[OK]${NC} CUDA $CUDA_VERSION"
else
    echo -e "${RED}[NO]${NC} CUDA (nvcc 未找到)"
    MISSING+=("cuda")
fi

# 检查 GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${GREEN}[OK]${NC} GPU: $GPU_INFO"
else
    echo -e "${RED}[NO]${NC} GPU (nvidia-smi 未找到)"
    MISSING+=("gpu-driver")
fi

# 检查编译器
if command -v g++ &>/dev/null; then
    echo -e "${GREEN}[OK]${NC} g++ $(g++ --version | head -1 | grep -oP '\d+\.\d+\.\d+' | head -1)"
else
    echo -e "${RED}[NO]${NC} g++ (未找到)"
    MISSING+=("build-essential")
fi

# 检查 CMake
if command -v cmake &>/dev/null; then
    echo -e "${GREEN}[OK]${NC} CMake $(cmake --version | head -1 | grep -oP '\d+\.\d+\.\d+')"
else
    echo -e "${RED}[NO]${NC} CMake (未找到)"
    MISSING+=("cmake")
fi

# 检查 Git
if command -v git &>/dev/null; then
    echo -e "${GREEN}[OK]${NC} Git $(git --version | grep -oP '\d+\.\d+\.\d+')"
else
    echo -e "${RED}[NO]${NC} Git (未找到)"
    MISSING+=("git")
fi

# 检查图形库
check_pkg() {
    local pkg=$1
    local desc=$2
    if dpkg -s "$pkg" &>/dev/null; then
        echo -e "${GREEN}[OK]${NC} $desc ($pkg)"
    else
        echo -e "${RED}[NO]${NC} $desc (缺少 $pkg)"
        MISSING+=("$pkg")
    fi
}

check_pkg "libglfw3-dev"     "GLFW（窗口库）"
check_pkg "libglew-dev"      "GLEW（OpenGL 扩展）"
check_pkg "freeglut3-dev"    "GLUT（兼容备用）"
check_pkg "libgl1-mesa-dev"  "Mesa OpenGL 开发头文件"

echo "------------------------------------------"

if [ ${#MISSING[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ 所有依赖已就绪！${NC}"
    exit 0
fi

# 询问是否安装
for pkg in "${MISSING[@]}"; do
    echo -e "  缺少: ${YELLOW}$pkg${NC}"
done

if [ "${1:-}" == "--install" ]; then
    echo -e "${YELLOW}正在安装缺失依赖...${NC}"
    sudo apt update
    sudo apt install -y "${MISSING[@]}"
    echo -e "${GREEN}✓ 依赖安装完成${NC}"
else
    echo
    echo -e "提示: 运行 ${YELLOW}./scripts/setup_env.sh --install${NC} 自动安装缺失依赖"
    echo "      或手动执行: sudo apt install ${MISSING[*]}"
    exit 1
fi