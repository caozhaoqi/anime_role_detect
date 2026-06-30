#!/bin/bash
set -e

# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1

GIT_REPO="https://github.com/caozhaoqi/anime_role_detect/"
REGISTRY="ardc"
BUILD_DIR="/tmp/ardc-build"
TAG=$(date +"%Y%m%d_%H%M%S")

DO_PUSH=false
for arg in "$@"; do
    case $arg in
        --push)
            DO_PUSH=true
            ;;
        --registry=*)
            REGISTRY="${arg#*=}"
            ;;
    esac
done

echo "========================================"
echo "  ARD 从源代码构建 K8s 镜像（优化版）"
echo "========================================"
echo "  Git 仓库: $GIT_REPO"
echo "  镜像仓库: $REGISTRY"
echo "  构建标签: $TAG"
echo "  构建目录: $BUILD_DIR"
echo ""
echo "  注意: 此脚本构建核心 3 个服务 + 基础镜像"
echo "  如需构建全部镜像，请使用:"
echo "    scripts/k8s/build_k8s_images.sh"
echo "========================================"

cleanup() {
    echo ""
    echo "🧹 清理构建目录..."
    rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

echo ""
echo "📥 克隆 Git 仓库..."
rm -rf "$BUILD_DIR"
git clone "$GIT_REPO" "$BUILD_DIR"

echo ""
echo "📁 进入构建目录..."
cd "$BUILD_DIR"

BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

BASE_IMAGE="$REGISTRY/base:$TAG"
ML_BASE_IMAGE="$REGISTRY/ml-base:$TAG"

echo ""
echo "📦 构建基础镜像..."
docker build -t "$BASE_IMAGE" \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg BUILD_TAG="$TAG" \
    -f deployment/Dockerfile.base \
    .

docker build -t "$ML_BASE_IMAGE" \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg BUILD_TAG="$TAG" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -f deployment/Dockerfile.ml-base \
    .

echo ""
echo "🔧 构建后端 API 镜像..."
docker build -t "$REGISTRY/api-service:$TAG" \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -f deployment/Dockerfile.api-service \
    .

echo ""
echo "🔧 构建模型服务镜像..."
docker build -t "$REGISTRY/model-service:$TAG" \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg BASE_IMAGE="$ML_BASE_IMAGE" \
    -f deployment/Dockerfile.model-service \
    .

echo ""
echo "🔧 构建前端镜像..."
docker build -t "$REGISTRY/frontend:$TAG" \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    -f deployment/Dockerfile.frontend \
    .

if [ "$DO_PUSH" = true ]; then
    echo ""
    echo "☁️ 推送镜像到仓库..."
    docker push "$REGISTRY/base:$TAG"
    docker push "$REGISTRY/ml-base:$TAG"
    docker push "$REGISTRY/api-service:$TAG"
    docker push "$REGISTRY/model-service:$TAG"
    docker push "$REGISTRY/frontend:$TAG"
fi

echo ""
echo "========================================"
echo "✅ 构建完成!"
echo "========================================"
echo ""
echo "📊 镜像列表:"
docker images | grep "$REGISTRY"

echo ""
echo "💡 部署命令:"
echo "   kubectl set image deployment/api-service api-service=$REGISTRY/api-service:$TAG -n anime-role-detect"
echo "   kubectl set image deployment/model-service model-service=$REGISTRY/model-service:$TAG -n anime-role-detect"
echo "   kubectl set image deployment/frontend frontend=$REGISTRY/frontend:$TAG -n anime-role-detect"
