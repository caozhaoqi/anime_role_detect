#!/bin/bash
set -e

export DOCKER_BUILDKIT=1

REGISTRY="ardc"
TAG=$(git rev-parse --short HEAD 2>/dev/null)

if [ -z "$TAG" ]; then
    TAG=$(date +%Y%m%d%H%M%S)
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

cd "$PROJECT_ROOT"

usage() {
    echo "使用方法: $0 [选项] <服务名>"
    echo ""
    echo "服务名列表:"
    echo "  api-service       API服务"
    echo "  model-service     模型服务"
    echo "  api-gateway       API网关"
    echo "  multimedia-service 多媒体服务"
    echo "  search-service    搜索服务"
    echo "  search-worker     搜索Worker"
    echo "  inference-worker  推理Worker"
    echo "  frontend          前端"
    echo "  monitoring        监控"
    echo ""
    echo "选项:"
    echo "  --push          推送到远程仓库"
    echo "  --registry=xxx  指定镜像仓库前缀"
    echo "  --all           构建所有服务"
    echo ""
    echo "示例:"
    echo "  $0 api-service"
    echo "  $0 --push api-service model-service"
    echo "  $0 --registry=harbor.example.com/ardc api-service"
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

DO_PUSH=false
SERVICES=()

for arg in "$@"; do
    case $arg in
        --push)
            DO_PUSH=true
            ;;
        --registry=*)
            REGISTRY="${arg#*=}"
            ;;
        --all)
            SERVICES=("api-service" "model-service" "api-gateway" "multimedia-service" "search-service" "search-worker" "inference-worker" "frontend" "monitoring")
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            SERVICES+=("$arg")
            ;;
    esac
done

if [ ${#SERVICES[@]} -eq 0 ]; then
    echo "错误: 请指定服务名"
    usage
    exit 1
fi

SERVICE_CONFIGS=(
    "api-service deployment/Dockerfile.api-service base"
    "model-service deployment/Dockerfile.model-service ml-base"
    "api-gateway deployment/Dockerfile.api-gateway base"
    "multimedia-service deployment/Dockerfile.multimedia-service base"
    "search-service deployment/Dockerfile.search-service base"
    "search-worker deployment/Dockerfile.search-worker base"
    "inference-worker deployment/Dockerfile.inference-worker ml-base"
    "frontend deployment/Dockerfile.frontend docker.io"
    "monitoring deployment/Dockerfile.monitoring docker.io"
)

BASE_IMAGE="$REGISTRY/base:latest"
ML_BASE_IMAGE="$REGISTRY/ml-base:latest"

_build_service() {
    local service_name=$1
    local dockerfile=$2
    local base_type=$3
    local image_name="$REGISTRY/$service_name:$TAG"

    echo "🔧 构建 $service_name..."

    local build_args=(--build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" --build-arg BUILD_TAG="$TAG")

    if [ "$base_type" = "base" ]; then
        build_args+=(--build-arg BASE_IMAGE="$BASE_IMAGE")
        cache_from="--cache-from=type=registry,ref=$BASE_IMAGE"
    elif [ "$base_type" = "ml-base" ]; then
        build_args+=(--build-arg BASE_IMAGE="$ML_BASE_IMAGE")
        cache_from="--cache-from=type=registry,ref=$ML_BASE_IMAGE --cache-from=type=registry,ref=$BASE_IMAGE"
    else
        cache_from=""
    fi

    if docker build -t "$image_name" $cache_from "${build_args[@]}" -f "$dockerfile" .; then
        echo "✅ $service_name 构建成功"
        if [ "$DO_PUSH" = true ]; then
            echo "📤 推送 $image_name..."
            if docker push "$image_name"; then
                echo "✅ $service_name 推送成功"
            else
                echo "❌ $service_name 推送失败"
                return 1
            fi
        fi
    else
        echo "❌ $service_name 构建失败"
        return 1
    fi
}

echo "========================================"
echo "  ARD 快速构建脚本"
echo "========================================"
echo "  镜像前缀: $REGISTRY"
echo "  镜像标签: $TAG"
echo "  服务列表: ${SERVICES[*]}"
echo "  推送: $DO_PUSH"
echo "========================================"

FAILED=()

for service_name in "${SERVICES[@]}"; do
    found=false
    for config in "${SERVICE_CONFIGS[@]}"; do
        read -r name dockerfile base_type <<< "$config"
        if [ "$name" = "$service_name" ]; then
            _build_service "$name" "$dockerfile" "$base_type" || FAILED+=("$name")
            found=true
            break
        fi
    done
    if [ "$found" = false ]; then
        echo "⚠️ 未知服务: $service_name"
        FAILED+=("$service_name")
    fi
done

echo ""
echo "========================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✅ 所有服务构建完成!"
    echo ""
    echo "💡 部署命令:"
    echo "   kubectl set image deployment/$service_name $service_name=$REGISTRY/$service_name:$TAG"
else
    echo "❌ 以下服务构建失败: ${FAILED[*]}"
    exit 1
fi