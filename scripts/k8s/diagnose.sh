#!/bin/bash
# =============================================================================
# ARD K8s 自动化诊断脚本
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
err()   { echo -e "${RED}[ERROR]${NC} $1"; }

NAMESPACE="anime-role-detect"
KUBECONFIG="/etc/rancher/k3s/k3s.yaml"

export KUBECONFIG="$KUBECONFIG"

echo ""
echo "========================================="
echo "  ARD K8s 自动化诊断"
echo "========================================="
echo ""

echo "--- 1. 系统环境检查 ---"
info "检查 k3s 服务状态..."
if systemctl is-active --quiet k3s; then
    ok "k3s 服务运行中"
    systemctl status k3s --no-pager | head -20
else
    err "k3s 服务未运行！"
    systemctl status k3s --no-pager | head -30
fi
echo ""

info "检查 Docker 服务状态..."
if systemctl is-active --quiet docker; then
    ok "Docker 服务运行中"
    docker_info=$(docker info 2>/dev/null || true)
    if [[ -n "$docker_info" ]]; then
        echo "$docker_info" | grep -E '(Server Version|Storage Driver|OS|Architecture)'
    else
        warn "Docker info 获取失败"
    fi
else
    err "Docker 服务未运行！"
    systemctl status docker --no-pager | head -20
fi
echo ""

info "检查 Docker 镜像..."
ardc_images=$(docker images 2>/dev/null | grep ardc || true)
if [[ -n "$ardc_images" ]]; then
    ok "已找到 ardc 镜像:"
    echo "$ardc_images"
else
    err "未找到 ardc 镜像！"
    echo "可用镜像:"
    docker images 2>/dev/null | head -20 || true
fi
echo ""

echo "--- 2. K8s 集群状态 ---"
info "检查节点状态..."
nodes=$(kubectl get nodes -o wide 2>/dev/null || true)
if [[ -n "$nodes" ]]; then
    echo "$nodes"
    node_status=$(echo "$nodes" | grep -E 'Ready|NotReady')
    if echo "$node_status" | grep -q 'Ready'; then
        ok "节点状态正常"
    else
        err "节点状态异常！"
    fi
else
    err "获取节点信息失败！"
fi
echo ""

info "检查命名空间..."
ns_exists=$(kubectl get namespace "$NAMESPACE" 2>/dev/null || true)
if [[ -n "$ns_exists" ]]; then
    ok "命名空间 $NAMESPACE 已存在"
else
    err "命名空间 $NAMESPACE 不存在！"
fi
echo ""

echo "--- 3. Pod 状态诊断 ---"
info "获取所有 Pod 状态..."
pods=$(kubectl -n "$NAMESPACE" get pods -o wide 2>/dev/null || true)
if [[ -n "$pods" ]]; then
    echo "$pods"
    echo ""
    
    crash_pods=$(echo "$pods" | grep -E 'CrashLoopBackOff|Error|ImagePullBackOff|ContainerCreating')
    if [[ -n "$crash_pods" ]]; then
        warn "发现异常 Pod:"
        echo "$crash_pods"
        echo ""
        
        while IFS= read -r pod_line; do
            pod_name=$(echo "$pod_line" | awk '{print $1}')
            pod_status=$(echo "$pod_line" | awk '{print $3}')
            
            info "获取 Pod [$pod_name] 的日志..."
            logs=$(kubectl -n "$NAMESPACE" logs "$pod_name" --previous 2>/dev/null | tail -30 || true)
            if [[ -n "$logs" ]]; then
                echo "$logs"
            else
                warn "  无法获取日志，尝试获取当前状态日志..."
                kubectl -n "$NAMESPACE" logs "$pod_name" 2>/dev/null | tail -30 || true
            fi
            echo ""
            
            info "获取 Pod [$pod_name] 的事件..."
            events=$(kubectl -n "$NAMESPACE" describe pod "$pod_name" 2>/dev/null | grep -A 10 'Events')
            if [[ -n "$events" ]]; then
                echo "$events"
            else
                warn "  无法获取事件"
            fi
            echo ""
            echo "-----------------------------------------"
            echo ""
        done <<< "$crash_pods"
    else
        ok "所有 Pod 状态正常"
    fi
else
    err "获取 Pod 信息失败！"
fi
echo ""

echo "--- 4. 服务状态检查 ---"
info "检查 Services..."
services=$(kubectl -n "$NAMESPACE" get svc 2>/dev/null || true)
if [[ -n "$services" ]]; then
    echo "$services"
    
    missing_ip=$(echo "$services" | grep '<none>' | grep -v 'EXTERNAL-IP')
    if [[ -n "$missing_ip" ]]; then
        warn "发现无 CLUSTER-IP 的服务:"
        echo "$missing_ip"
    fi
else
    err "获取 Services 信息失败！"
fi
echo ""

echo "--- 5. 存储状态检查 ---"
info "检查 PVC..."
pvcs=$(kubectl -n "$NAMESPACE" get pvc 2>/dev/null || true)
if [[ -n "$pvcs" ]]; then
    echo "$pvcs"
    
    pending_pvcs=$(echo "$pvcs" | grep 'Pending')
    if [[ -n "$pending_pvcs" ]]; then
        err "发现 Pending 状态的 PVC:"
        echo "$pending_pvcs"
    else
        ok "所有 PVC 状态正常"
    fi
else
    err "获取 PVC 信息失败！"
fi
echo ""

echo "--- 6. 资源使用情况 ---"
info "检查节点资源使用..."
resource=$(kubectl top nodes 2>/dev/null || true)
if [[ -n "$resource" ]]; then
    echo "$resource"
else
    warn "无法获取节点资源信息（metrics-server 可能未部署）"
    free -h || true
fi
echo ""

info "检查 Pod 资源使用..."
pod_resource=$(kubectl -n "$NAMESPACE" top pods 2>/dev/null || true)
if [[ -n "$pod_resource" ]]; then
    echo "$pod_resource"
else
    warn "无法获取 Pod 资源信息"
fi
echo ""

echo "--- 7. 配置检查 ---"
info "检查 ConfigMap..."
configmaps=$(kubectl -n "$NAMESPACE" get configmap 2>/dev/null || true)
if [[ -n "$configmaps" ]]; then
    echo "$configmaps"
else
    warn "未找到 ConfigMap"
fi
echo ""

info "检查 Secrets..."
secrets=$(kubectl -n "$NAMESPACE" get secret 2>/dev/null || true)
if [[ -n "$secrets" ]]; then
    echo "$secrets" | grep -v 'kubernetes.io/service-account-token'
else
    warn "未找到 Secrets"
fi
echo ""

echo "========================================="
echo "  诊断完成"
echo "========================================="
echo ""
echo "建议："
echo "  1. 如果是 ImagePullBackOff：检查 Docker 中是否有对应的镜像"
echo "  2. 如果是 CrashLoopBackOff：查看上面的日志分析具体错误"
echo "  3. 如果是 ContainerCreating：检查 PVC 和网络配置"
echo "  4. 如果是 OOMKilled：增加 Pod 的内存限制"
echo ""
