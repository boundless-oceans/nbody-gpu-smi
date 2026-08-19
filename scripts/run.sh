#!/usr/bin/env bash
# 运行模拟器脚本
# 用法:
#   ./scripts/run.sh                       # GUI 模式（S1 骨架）
#   ./scripts/run.sh --benchmark           # CUDA 物理性能基准（S2）
#   ./scripts/run.sh --benchmark --n=50000 --iter=10
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
# 可执行文件输出在项目根目录（与 NVIDIA 官方仓库一致）
GUI_BINARY="$PROJECT_DIR/nbody"
BENCH_BINARY="$PROJECT_DIR/nbody_benchmark"

# 解析参数
MODE="gui"
N_BODIES=""
ITERATIONS=""
for arg in "$@"; do
    case $arg in
        --benchmark) MODE="benchmark" ;;
        --n=*) N_BODIES="${arg#--n=}" ;;
        --iter=*) ITERATIONS="${arg#--iter=}" ;;
        *) echo "未知参数: $arg" >&2; exit 1 ;;
    esac
done

if [[ "$MODE" == "benchmark" ]]; then
    if [[ ! -x "$BENCH_BINARY" ]]; then
        echo -e "${RED}[错误]${NC} 未找到 $BENCH_BINARY"
        echo -e "请先运行: ./scripts/build.sh"
        exit 1
    fi

    ARGS=()
    if [[ -n "$N_BODIES" ]]; then
        ARGS+=("$N_BODIES")
    fi
    if [[ -n "$ITERATIONS" ]]; then
        ARGS+=("$ITERATIONS")
    fi

    echo -e "${GREEN}[INFO]${NC} 运行 CUDA N-Body 性能基准..."
    exec "$BENCH_BINARY" "${ARGS[@]}"
fi

# GUI 模式
if [[ ! -x "$GUI_BINARY" ]]; then
    echo -e "${RED}[错误]${NC} 未找到可执行文件 $GUI_BINARY"
    echo -e "请先运行: ./scripts/build.sh"
    exit 1
fi

# 检查图形显示环境
if [[ -z "${DISPLAY:-}" ]] && [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo -e "${RED}[错误]${NC} 未检测到显示环境（DISPLAY 为空）"
    echo "  远程环境可用: ./scripts/run.sh --benchmark"
    exit 1
fi

if [[ -n "$N_BODIES" ]]; then
    echo -e "${YELLOW}[警告]${NC} GUI 模式暂不支持运行时设置 N（S7 加入），忽略 --n=$N_BODIES"
fi

echo -e "${GREEN}[INFO]${NC} 启动模拟器..."
exec "$GUI_BINARY"