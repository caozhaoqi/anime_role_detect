#!/bin/bash
# ============================================================
# ARD K8s 集群故障修复脚本
# 针对三类问题:
#   1. ImagePullBackOff (ardc/monitoring 镜像缺失)
#   2. Pending (PVC storageClassName 缺失 + 资源不足)
#   3. 单节点副本缩减
# ============================================================

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
NAMESPACE="anime-role-detect"

echo "========================================"
echo "  ARD K8s 集群故障修复"
echo "========================================"

# ============================================================
# Step 1: 构建 monitoring 镜像
# ============================================================
echo ""
echo "📦 Step 1: 构建 ardc/monitoring:latest 镜像..."

cd "$PROJECT_ROOT"

TAG=$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

export DOCKER_BUILDKIT=1

if sudo docker build \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg BUILD_TAG="$TAG" \
    -f deployment/Dockerfile.monitoring \
    -t ardc/monitoring:$TAG \
    -t ardc/monitoring:latest \
    . ; then
    echo "✅ ardc/monitoring:latest 构建成功"
else
    echo "❌ ardc/monitoring 构建失败!"
    echo "   请检查 deployment/Dockerfile.monitoring 和 deployment/requirements-monitoring.txt"
    exit 1
fi

# ============================================================
# Step 2: 将 monitoring 镜像导入 K3s containerd
# ============================================================
echo ""
echo "📥 Step 2: 将 monitoring 镜像导入 K3s containerd..."

# K3s 使用自己的 containerd, 需要通过 k3s ctr 导入
sudo docker save ardc/monitoring:latest | sudo k3s ctr images import -

echo "✅ 镜像已导入 K3s"

# ============================================================
# Step 3: 删除旧 PVC 并重新创建 (带 storageClassName)
# ============================================================
echo ""
echo "🔄 Step 3: 重新创建 PVC (添加 storageClassName: local-path)..."

# 先删除旧的未绑定 PVC
for pvc in model-data model-cache api-cache mysql-data redis-data rabbitmq-data log-data; do
    PVC_STATUS=$(sudo kubectl -n $NAMESPACE get pvc $pvc -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
    if [ "$PVC_STATUS" = "Pending" ] || [ "$PVC_STATUS" = "NotFound" ]; then
        echo "  删除未绑定 PVC: $pvc (状态: $PVC_STATUS)"
        sudo kubectl -n $NAMESPACE delete pvc $pvc --ignore-not-found=true
    elif [ "$PVC_STATUS" = "Bound" ]; then
        echo "  ⚠️  PVC $pvc 已绑定, 跳过删除"
    fi
done

# 重新 apply 全部 K8s 资源（权威源：k8s/base/）
# 旧 deployment/k8s-*.yaml 已归档至 deployment/_legacy_backup/
sudo kubectl apply -k k8s/base/

echo "  等待 PVC 绑定..."
sleep 5
sudo kubectl -n $NAMESPACE get pvc

# ============================================================
# Step 4: 重新 apply Deployment (降低资源请求 + 缩减副本)
# ============================================================
echo ""
echo "🔄 Step 4: 重新 apply Deployment 配置..."

sudo kubectl apply -k k8s/base/

# ============================================================
# Step 5: 删除所有 Pending 和 ImagePullBackOff 的 Pod, 触发重新调度
# ============================================================
echo ""
echo "🧹 Step 5: 删除故障 Pod, 触发重新调度..."

# 删除 Pending pods
sudo kubectl -n $NAMESPACE delete pods --field-selector=status.phase=Pending --grace-period=0

# 删除 ImagePullBackOff pods (通过 label 逐个删除)
for pod_name in $(sudo kubectl -n $NAMESPACE get pods -o json | \
    python3 -c "
import json, sys
data = json.load(sys.stdin)
for item in data['items']:
    for cs in item.get('status', {}).get('conditions', []):
        if cs.get('reason') == 'ImagePullBackOff' or cs.get('message', '').find('ImagePullBackOff') >= 0:
            print(item['metadata']['name'])
            break
    for cs in item.get('status', {}).get('containerStatuses', []) or []:
        if cs.get('state', {}).get('waiting', {}).get('reason') == 'ImagePullBackOff':
            print(item['metadata']['name'])
" 2>/dev/null); do
    echo "  删除 ImagePullBackOff Pod: $pod_name"
    sudo kubectl -n $NAMESPACE delete pod $pod_name --grace-period=0 --force
done

# ============================================================
# Step 6: 等待并验证
# ============================================================
echo ""
echo "⏳ Step 6: 等待 30 秒后验证..."

sleep 30

echo ""
echo "========================================"
echo "  修复后 Pod 状态:"
echo "========================================"
sudo kubectl -n $NAMESPACE get pods -o wide

echo ""
echo "========================================"
echo "  PVC 状态:"
echo "========================================"
sudo kubectl -n $NAMESPACE get pvc

echo ""
echo "💡 如果仍有 Pending Pod, 请检查:"
echo "   1. kubectl describe node <node> | grep -A5 'Allocated resources'"
echo "   2. kubectl describe pod <pending-pod> | grep -A10 Events"
echo ""
echo "💡 如果 monitoring Pod 仍然 ImagePullBackOff:"
echo "   sudo k3s ctr images ls | grep monitoring"
echo "   确认镜像已导入到 K3s containerd"
