#!/usr/bin/env bash
# ============================================================================
# 本地 K8s 验证脚本 —— 真机运行（需在 macOS 上启动 Docker Desktop / OrbStack）
#
# 用途：用 kind 单节点集群验证 k8s/overlays/ci manifests 是否可用
# 前置：1) 安装并启动 Docker Desktop 或 OrbStack（提供 docker daemon）
#       2) 已装 kind 与 kubectl（brew install kind kubectl）
#
# 用法：
#   ./scripts/k8s/verify_local.sh            # 完整验证（建集群→apply→等待→探测→清理）
#   ./scripts/k8s/verify_local.sh --no-cleanup   # 保留集群，便于手动排查
# ============================================================================
set -euo pipefail

CLUSTER="ard-ci"
NAMESPACE="anime-role-detect"
OVERLAY="k8s/overlays/ci"
CLEANUP=1
KIND_WAIT="120s"

for arg in "$@"; do
  case "$arg" in
    --no-cleanup) CLEANUP=0 ;;
    *) echo "未知参数: $arg"; exit 2 ;;
  esac
done

log()  { echo -e "\033[32m[INFO]\033[0m  $*"; }
warn() { echo -e "\033[33m[WARN]\033[0m  $*"; }
die()  { echo -e "\033[31m[FAIL]\033[0m  $*" >&2; exit 1; }

# ---- 0. 环境检查 ----
log "检查 docker daemon ..."
docker info >/dev/null 2>&1 || die "docker daemon 未运行，请先启动 Docker Desktop / OrbStack"
command -v kind >/dev/null || die "未安装 kind：brew install kind"
command -v kubectl >/dev/null || die "未安装 kubectl：brew install kubectl"

# ---- 1. kind 集群 ----
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  warn "集群 $CLUSTER 已存在，复用"
else
  log "创建 kind 集群 $CLUSTER ..."
  kind create cluster --name "$CLUSTER" --wait "$KIND_WAIT"
fi

# ---- 2. 校验 manifests ----
log "kubectl kustomize 校验 (${OVERLAY}) ..."
kubectl kustomize "$OVERLAY" >/dev/null || die "kustomize 渲染失败"

# ---- 3. apply ----
log "apply manifests ..."
kubectl apply -k "$OVERLAY"

# ---- 4. 等待资源就绪 ----
log "等待 namespace + 基础组件就绪 (redis/rabbitmq/mysql/services) ..."
kubectl -n "$NAMESPACE" wait --for=condition=Available --timeout=180s \
  $(kubectl -n "$NAMESPACE" get deploy -o name) 2>/dev/null || warn "部分 Deployment 未 Available（镜像/资源原因，见下）"

log "等待 30s 让 Pod 调度/启动 ..."
sleep 30

# ---- 5. 资源汇总 ----
log "=== Pod 状态 ==="
kubectl -n "$NAMESPACE" get pods -o wide
echo
log "=== 未就绪 Pod 事件（便于定位镜像/资源问题）==="
kubectl -n "$NAMESPACE" get pods --field-selector=status.phase!=Running -o name 2>/dev/null | while read p; do
  echo "--- $p ---"
  kubectl -n "$NAMESPACE" describe "$p" | sed -n '/Events:/,$p' | head -20
done

# ---- 6. 探针/向下 API 验证（针对能跑起来的组件）----
log "=== 验证 POD_NAME / NODE_NAME 注入 ==="
for dep in redis rabbitmq api-gateway api-service; do
  if kubectl -n "$NAMESPACE" get deploy "$dep" >/dev/null 2>&1; then
    kubectl -n "$NAMESPACE" exec deploy/"$dep" -c "${dep}" -- sh -c 'echo POD=$POD_NAME NODE=$NODE_NAME' 2>/dev/null \
      && log "$dep: 向下 API OK" || warn "$dep: 暂未运行或容器名不一致，跳过"
  fi
done

# ---- 7. 服务连通性（集群内）----
log "=== 验证基础服务端口可达（redis/rabbitmq/mysql）==="
for svc in redis:6379 rabbitmq:5672 mysql:3306; do
  name=${svc%%:*}; port=${svc##*:}
  kubectl -n "$NAMESPACE" run netcheck-$name --rm -i --restart=Never \
    --image=busybox:1.36 --timeout=60s -- \
    sh -c "nc -zv $name $port" 2>/dev/null \
    && log "$name:$port reachable" || warn "$name:$port 不可达或镜像拉取失败"
done

# ---- 8. 清理 ----
if [ "$CLEANUP" -eq 1 ]; then
  log "清理集群 ..."
  kind delete cluster --name "$CLUSTER"
  log "完成：集群已删除"
else
  warn "保留集群 $CLUSTER（--no-cleanup）。手动删除：kind delete cluster --name $CLUSTER"
fi
