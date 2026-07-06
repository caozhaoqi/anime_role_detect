#!/bin/bash
# =============================================================================
# ARD ML 依赖预下载脚本
# 提前下载 PyTorch 等重型依赖到本地缓存，加速 Docker 构建
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

CACHE_DIR="${HOME}/.cache/pip/downloads"
PYTORCH_VERSION="2.1.0+cpu"
TORCHVISION_VERSION="0.16.0+cpu"
TORCHAUDIO_VERSION="2.1.0+cpu"

mkdir -p "$CACHE_DIR"

download_file() {
    local url="$1"
    local filename="$2"
    local dest="$CACHE_DIR/$filename"

    if [[ -f "$dest" ]]; then
        info "  已存在，跳过: $filename"
        return
    fi

    info "  下载: $filename..."
    if curl -fL --progress-bar --retry 5 --retry-delay 10 "$url" -o "$dest"; then
        ok "  下载完成: $filename"
    else
        warn "  下载失败: $filename (将在构建时通过 pip 下载)"
        rm -f "$dest"
    fi
}

echo ""
echo "========================================="
echo "  ARD ML 依赖预下载"
echo "========================================="
echo "  缓存目录: $CACHE_DIR"
echo ""

echo "--- 下载 PyTorch 依赖 ---"
info "开始下载 PyTorch 相关包..."

download_file "https://download.pytorch.org/whl/cpu/torch-${PYTORCH_VERSION}.whl" \
    "torch-${PYTORCH_VERSION}.whl"

download_file "https://download.pytorch.org/whl/cpu/torchvision-${TORCHVISION_VERSION}.whl" \
    "torchvision-${TORCHVISION_VERSION}.whl"

download_file "https://download.pytorch.org/whl/cpu/torchaudio-${TORCHAUDIO_VERSION}.whl" \
    "torchaudio-${TORCHAUDIO_VERSION}.whl"

echo ""
echo "--- 下载 transformers 相关包 ---"
info "开始下载 transformers 相关包..."

TRANSFORMERS_VERSION="4.57.6"
download_file "https://files.pythonhosted.org/packages/py3/t/transformers/transformers-${TRANSFORMERS_VERSION}-py3-none-any.whl" \
    "transformers-${TRANSFORMERS_VERSION}-py3-none-any.whl"

echo ""
echo "--- 下载 sentence-transformers 相关包 ---"
info "开始下载 sentence-transformers 相关包..."

SENTENCE_TRANSFORMERS_VERSION="3.0.1"
download_file "https://files.pythonhosted.org/packages/py3/s/sentence_transformers/sentence_transformers-${SENTENCE_TRANSFORMERS_VERSION}-py3-none-any.whl" \
    "sentence_transformers-${SENTENCE_TRANSFORMERS_VERSION}-py3-none-any.whl"

echo ""
echo "========================================="
echo "  预下载完成!"
echo "========================================="
echo "  缓存文件位于: $CACHE_DIR"
echo ""
echo "  构建时使用预下载缓存:"
echo "    docker build --build-arg PIP_CACHE_DIR=$CACHE_DIR \\"
echo "                 -t ardc/ml-base:latest -f deployment/Dockerfile.ml-base ."
echo ""
