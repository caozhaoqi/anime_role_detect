#!/bin/bash
set -e

REGISTRY="harbor.example.com/anime-role-detect"
TAG=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

echo "========================================"
echo "  ARD K8s 容器镜像构建脚本"
echo "========================================"
echo "  项目根目录: $PROJECT_ROOT"
echo "  镜像仓库: $REGISTRY"
echo "  镜像标签: $TAG"
echo "  构建日期: $BUILD_DATE"
echo "========================================"

cd "$PROJECT_ROOT"

build_image() {
    local service_name=$1
    local dockerfile=$2
    local context=$3
    local image_name="$REGISTRY/$service_name:$TAG"
    
    echo ""
    echo "🔧 构建 $service_name 镜像..."
    echo "   Dockerfile: $dockerfile"
    echo "   Context: $context"
    echo "   Image: $image_name"
    
    if docker build -t "$image_name" \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg BUILD_TAG="$TAG" \
        --file "$dockerfile" \
        "$context"; then
        echo "✅ $service_name 构建成功"
        echo "   推送到镜像仓库..."
        docker push "$image_name"
        echo "✅ $service_name 推送成功"
    else
        echo "❌ $service_name 构建失败"
        exit 1
    fi
}

echo ""
echo "📋 开始构建服务镜像..."

# 核心服务
build_image "api-service" "deployment/Dockerfile.api-service" "."
build_image "model-service" "deployment/Dockerfile.model-service" "."
build_image "frontend" "deployment/Dockerfile.frontend" "."

# 网关和服务
build_image "api-gateway" "deployment/Dockerfile.api-gateway" "."
build_image "multimedia-service" "deployment/Dockerfile.multimedia-service" "."

# 搜索服务
build_image "search-service" "deployment/Dockerfile.search-service" "."
build_image "search-worker" "deployment/Dockerfile.search-worker" "."

# 推理工作器
build_image "inference-worker" "deployment/Dockerfile.inference-worker" "."

# 监控工具
build_image "monitor-dashboard" "deployment/Dockerfile.monitor-dashboard" "."
build_image "log-viewer" "deployment/Dockerfile.log-viewer" "."

# 健康检查和监控
build_image "health-check" "deployment/Dockerfile.health-check" "."
build_image "log-monitor" "deployment/Dockerfile.log-monitor" "."
build_image "resource-monitor" "deployment/Dockerfile.resource-monitor" "."

echo ""
echo "========================================"
echo "✅ 所有镜像构建完成!"
echo "========================================"
echo ""
echo "📊 镜像列表:"
docker images | grep "$REGISTRY"

echo ""
echo "💡 使用命令部署到 K8s:"
echo "   kubectl apply -f deployment/k8s-deploy.yaml"
echo ""
echo "📝 环境变量配置:"
echo "   请修改 deployment/k8s-deploy.yaml 中的数据库和 Redis 连接信息"