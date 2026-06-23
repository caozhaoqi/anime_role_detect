#!/bin/bash
set -e

GIT_REPO="https://github.com/caozhaoqi/anime_role_detect/"
REGISTRY="harbor.example.com/anime-role-detect"
BUILD_DIR="/tmp/ardc-build"
TAG=$(date +"%Y%m%d_%H%M%S")

echo "========================================"
echo "  ARD 从源代码构建 K8s 镜像"
echo "========================================"
echo "  Git 仓库: $GIT_REPO"
echo "  镜像仓库: $REGISTRY"
echo "  构建标签: $TAG"
echo "  构建目录: $BUILD_DIR"
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

echo ""
echo "📦 安装依赖..."
pip install -q -r requirements.txt
pip install -q uvicorn gunicorn

echo ""
echo "🔧 构建后端 API 镜像..."
docker build -t "$REGISTRY/api-service:$TAG" \
    --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
    -f deployment/Dockerfile.api-service \
    .

echo ""
echo "🔧 构建模型服务镜像..."
docker build -t "$REGISTRY/model-service:$TAG" \
    --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
    -f deployment/Dockerfile.model-service \
    .

echo ""
echo "🔧 构建前端镜像..."
docker build -t "$REGISTRY/frontend:$TAG" \
    --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
    -f deployment/Dockerfile.frontend \
    .

echo ""
echo "☁️ 推送镜像到仓库..."
docker push "$REGISTRY/api-service:$TAG"
docker push "$REGISTRY/model-service:$TAG"
docker push "$REGISTRY/frontend:$TAG"

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
