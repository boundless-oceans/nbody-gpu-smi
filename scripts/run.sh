#!/usr/bin/env bash
# 运行模拟器脚本
# 用法: ./scripts/run.sh [--benchmark] [--debug] [--n=<粒子数>]
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
BINARY="$BUILD_DIR/nbody"

# 检查可执行文件
if [[ ! -x "$BINARY" ]]; then
    echo -e "${RED}[错误]${NC} 未找到可执行文件 $BINARY"
    echo -e "请先运行: ./scripts/build.sh"
    exit 1
fi

BENCHMARK=false
DEBUG=false
N_BODIES=""
for arg in "$@"; do
    case $arg in
        --benchmark) BENCHMARK=true ;;
        --debug) DEBUG=true ;;
        --n=*) N_BODIES="${arg#--n=}" ;;
        *) echo "未知参数: $arg" >&2; exit 1 ;;
    esac
done

# 检查图形显示环境（非 benchmark 模式）
if [[ "$BENCHMARK" == false ]] && [[ -z "${DISPLAY:-}" ]] && [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo -e "${RED}[错误]${NC} 未检测到显示环境（DISPLAY 为空）"
    echo "  远程环境可使用: ./scripts/run.sh --benchmark"
    exit 1
fi

ARGS=()
if [[ "$BENCHMARK" == true ]]; then
    ARGS+=("--benchmark")
    echo -e "${GREEN}[INFO]${NC} 基准测试模式"
fi
if [[ "$DEBUG" == true ]]; then
    ARGS+=("--debug")
fi
if [[ -n "$N_BODIES" ]]; then
    ARGS+=("--n=$N_BODIES")
    echo -e "${GREEN}[INFO]${NC} 粒子数: $N_BODIES"
fi

echo -e "${GREEN}[INFO]${NC} 启动模拟器..."
exec "$BINARY" "${ARGS[@]}"