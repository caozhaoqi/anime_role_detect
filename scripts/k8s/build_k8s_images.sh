#!/bin/bash
set -e

export DOCKER_BUILDKIT=1

REGISTRY="ardc"
TAG=$(git rev-parse --short HEAD 2>/dev/null)
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

if [ -z "$TAG" ]; then
    echo "警告: 无法获取 Git commit hash，将使用时间戳作为镜像标签。"
    TAG=$(date +%Y%m%d%H%M%S)
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

SKIP_BASE=false
DO_PUSH=false
PUSH_OSS=false
PARALLEL_JOBS=8
USE_CACHE=true
PRE_DOWNLOAD=false
for arg in "$@"; do
    case $arg in
        --skip-base)
            SKIP_BASE=true
            echo "ℹ️  跳过基础镜像构建（使用已有缓存）"
            ;;
        --push)
            DO_PUSH=true
            ;;
        --push-oss)
            PUSH_OSS=true
            ;;
        --registry=*)
            REGISTRY="${arg#*=}"
            ;;
        --jobs=*)
            PARALLEL_JOBS="${arg#*=}"
            ;;
        --no-cache)
            USE_CACHE=false
            echo "ℹ️  禁用构建缓存"
            ;;
        --pre-download)
            PRE_DOWNLOAD=true
            echo "ℹ️  启用 ML 依赖预下载（加速首次构建）"
            ;;
    esac
done

echo "========================================"
echo "  ARD K8s 容器镜像构建脚本 (自动空间清理版)"
echo "========================================"
echo "  项目根目录: $PROJECT_ROOT"
echo "  镜像前缀: $REGISTRY"
echo "  镜像标签: $TAG"
echo "  构建日期: $BUILD_DATE"
echo "  推送到仓库: $DO_PUSH"
echo "  并行任务数: $PARALLEL_JOBS"
echo "  使用缓存: $USE_CACHE"
echo "  预下载ML依赖: $PRE_DOWNLOAD"
echo "========================================"

# ======================== Phase 0.1: 自动空间清理 ========================
echo ""
echo "🧹 Phase 0.1: 自动清理垃圾并释放物理磁盘空间..."
echo "  [清理前] 磁盘空间状态:"
df -h /var/lib/docker 2>/dev/null || df -h /

# 1. 深度清理未使用的容器、镜像、无用卷和本地构建缓存
echo "  正在执行 Docker 深度清理 (System Prune)..."
docker system prune -a --volumes -f || true
docker buildx prune -f || true

# 2. 清理临时目录下的旧构建日志
echo "  清理历史临时日志..."
rm -f /tmp/build_*.log || true

echo "  [清理后] 磁盘空间状态:"
df -h /var/lib/docker 2>/dev/null || df -h /
echo "========================================"

cd "$PROJECT_ROOT"

if [ "$PRE_DOWNLOAD" = true ]; then
    echo ""
    echo "📥 Phase 0: 预下载 ML 依赖..."
    if bash "$SCRIPT_DIR/pre_download_ml_deps.sh"; then
        echo "✅ ML 依赖预下载完成"
    else
        echo "⚠️  ML 依赖预下载部分失败，将在构建时通过 pip 下载"
    fi
fi

BASE_IMAGE="$REGISTRY/base:$TAG"
ML_BASE_IMAGE="$REGISTRY/ml-base:$TAG"

CACHE_ARGS=""

_build_image() {
    local service_name=$1
    local dockerfile=$2
    local base_image=$3
    local image_name="$REGISTRY/$service_name:$TAG"
    local log_file="/tmp/build_${service_name}_$(date +%s%N).log"

    echo "🔧 开始构建 $service_name 镜像 (日志: $log_file)..."

    local build_args=(
        --build-arg BUILD_DATE="$BUILD_DATE"
        --build-arg BUILD_TAG="$TAG"
        --file "$dockerfile"
    )

    if [ "$USE_CACHE" = true ]; then
        if [ -d "/tmp/docker-cache/$service_name" ]; then
            build_args+=(--cache-from=type=local,src=/tmp/docker-cache/$service_name)
        fi
        build_args+=(--cache-to=type=local,dest=/tmp/docker-cache/$service_name,mode=max)
    fi
    
    if [ "$base_image" != "docker.io" ]; then
        build_args+=(
            --build-arg BASE_IMAGE="$base_image"
        )
    fi

    if docker build --pull=false -t "$image_name" "${build_args[@]}" . > "$log_file" 2>&1; then
        echo "✅ $service_name 构建成功 → $image_name"
        echo "  🏷️  添加 latest 标签..."
        docker tag "$image_name" "$REGISTRY/$service_name:latest" 2>/dev/null || true
        if [ "$DO_PUSH" = true ]; then
            echo "  📤 推送 $image_name ..." >> "$log_file"
            if docker push "$image_name" >> "$log_file" 2>&1; then
                echo "✅ $service_name 推送成功."
            else
                echo "❌ $service_name 推送失败. 详情: $log_file"
                return 1
            fi
            echo "  📤 推送 ${REGISTRY}/${service_name}:latest ..." >> "$log_file"
            docker push "${REGISTRY}/${service_name}:latest" >> "$log_file" 2>&1 || true
        fi
        return 0
    else
        echo "❌ $service_name 构建失败. 详情: $log_file"
        cat "$log_file"
        return 1
    fi
}

export -f _build_image
export REGISTRY TAG BUILD_DATE DO_PUSH USE_CACHE

check_base_images() {
    local base_exists=$(docker images "$BASE_IMAGE" --format "{{.ID}}" 2>/dev/null | head -1)
    local ml_base_exists=$(docker images "$ML_BASE_IMAGE" --format "{{.ID}}" 2>/dev/null | head -1)
    
    if [ -z "$base_exists" ] || [ -z "$ml_base_exists" ]; then
        echo "⚠️  基础镜像不存在，自动回退到构建模式..."
        return 1
    fi
    echo "ℹ️  基础镜像已存在，跳过构建"
    return 0
}

if [ "$SKIP_BASE" = false ] || ! check_base_images; then
    echo ""
    echo "📦 Phase 1: 构建基础镜像..."

    echo "  🔧 构建 base 镜像..."
    log_file="/tmp/build_base_$(date +%s%N).log"
    if docker build --pull=false \
        -t "$BASE_IMAGE" \
        $(if [ "$USE_CACHE" = true ]; then \
            if [ -d "/tmp/docker-cache/base" ]; then \
                echo "--cache-from=type=local,src=/tmp/docker-cache/base"; \
            fi; \
            echo "--cache-to=type=local,dest=/tmp/docker-cache/base,mode=max"; \
        fi) \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg BUILD_TAG="$TAG" \
        -f deployment/Dockerfile.base \
        . > "$log_file" 2>&1; then
        echo "✅ base 镜像构建成功"
        echo "  🏷️  添加 latest 标签..."
        docker tag "$BASE_IMAGE" "${REGISTRY}/base:latest" 2>/dev/null || true
    else
        echo "❌ base 镜像构建失败"
        echo "📜 详细日志:"
        cat "$log_file"
        exit 1
    fi

    echo "  🔧 构建 ml-base 镜像..."
    BASE_NAME="${BASE_IMAGE%%:*}"
    log_file="/tmp/build_ml-base_$(date +%s%N).log"
    if docker build --pull=false \
        -t "$ML_BASE_IMAGE" \
        $(if [ "$USE_CACHE" = true ]; then \
            if [ -d "/tmp/docker-cache/ml-base" ]; then \
                echo "--cache-from=type=local,src=/tmp/docker-cache/ml-base"; \
            fi; \
            echo "--cache-to=type=local,dest=/tmp/docker-cache/ml-base,mode=max"; \
        fi) \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg BUILD_TAG="$TAG" \
        --build-arg BASE_IMAGE="$BASE_IMAGE" \
        -f deployment/Dockerfile.ml-base \
        . > "$log_file" 2>&1; then
        echo "✅ ml-base 镜像构建成功"
        echo "  🏷️  添加 latest 标签..."
        docker tag "$ML_BASE_IMAGE" "${REGISTRY}/ml-base:latest" 2>/dev/null || true
    else
        echo "❌ ml-base 镜像构建失败"
        exit 1
    fi

    echo "✅ 基础镜像构建完成"

    if [ "$DO_PUSH" = true ]; then
        echo "  📤 推送基础镜像..."
        docker push "$BASE_IMAGE" &
        docker push "$ML_BASE_IMAGE" &
        wait
        echo "✅ 基础镜像推送完成"
    fi
else
    echo ""
    echo "ℹ️  跳过基础镜像构建"
fi

echo ""
echo "📋 Phase 2: 并行构建服务镜像..."

BUILD_TASKS_FILE=$(mktemp)
BUILD_STATUS=0

echo "api-service deployment/Dockerfile.api-service $BASE_IMAGE" >> "$BUILD_TASKS_FILE"
echo "model-service deployment/Dockerfile.model-service $ML_BASE_IMAGE" >> "$BUILD_TASKS_FILE"
echo "frontend deployment/Dockerfile.frontend docker.io" >> "$BUILD_TASKS_FILE"
echo "api-gateway deployment/Dockerfile.api-gateway $BASE_IMAGE" >> "$BUILD_TASKS_FILE"
echo "multimedia-service deployment/Dockerfile.multimedia-service $ML_BASE_IMAGE" >> "$BUILD_TASKS_FILE"
echo "search-service deployment/Dockerfile.search-service $ML_BASE_IMAGE" >> "$BUILD_TASKS_FILE"
echo "search-worker deployment/Dockerfile.search-worker $ML_BASE_IMAGE" >> "$BUILD_TASKS_FILE"
echo "inference-worker deployment/Dockerfile.inference-worker $ML_BASE_IMAGE" >> "$BUILD_TASKS_FILE"
echo "monitoring deployment/Dockerfile.monitoring docker.io" >> "$BUILD_TASKS_FILE"

cat "$BUILD_TASKS_FILE" | xargs -P $PARALLEL_JOBS -n 3 bash -c '_build_image "$@"' _

if [ $? -ne 0 ]; then
    BUILD_STATUS=1
fi

rm "$BUILD_TASKS_FILE"

echo ""
echo "========================================"
if [ "$BUILD_STATUS" -eq 0 ]; then
    echo "✅ 所有镜像构建完成!"
else
    echo "❌ 部分镜像构建失败。"
fi
echo "========================================"
echo ""
echo "📊 镜像列表:"
docker images | grep "$REGISTRY"

# Phase 3: 推送到 OSS (可选)
if [ "$PUSH_OSS" = true ] && [ "$BUILD_STATUS" -eq 0 ]; then
    echo ""
    echo "☁️  Phase 3: 推送镜像到阿里云 OSS..."
    python3 "$SCRIPT_DIR/push_images_to_oss.py" --tag "$TAG" --registry "$REGISTRY"
fi

echo ""
echo "💡 部署到 K8s（本地镜像）:"
echo "   kubectl apply -f deployment/k8s-deploy.yaml"
echo "   kubectl apply -f deployment/k8s-volumes.yaml"
echo "   kubectl apply -f deployment/k8s-services.yaml"
echo "   kubectl apply -f deployment/k8s-deployments.yaml"
echo "   kubectl apply -f deployment/k8s-ingress.yaml"
echo ""
echo "💡 推送到远程仓库:"
echo "   $0 --push --registry=your-harbor.com/anime-role-detect"
echo ""
echo "💡 构建后推送到阿里云 OSS:"
echo "   $0 --push-oss"
echo "   python3 $SCRIPT_DIR/push_images_to_oss.py  # 独立运行"
echo ""
echo "💡 下次构建时跳过基础镜像（依赖未变更时）:"
echo "   $0 --skip-base"
echo ""
echo "💡 调整并行任务数:"
echo "   $0 --jobs=8"