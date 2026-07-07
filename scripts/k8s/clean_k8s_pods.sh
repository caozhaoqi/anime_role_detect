#!/bin/bash
# =============================================================================
# K8s 异常 Pod 一键清理与重启脚本
#
# 功能：
#   1. 清理已驱逐 (Evicted) 的垃圾 Pod（安全释放节点空间）
#   2. 清理/重置处于 Error、Failed、CrashLoop、ImagePull 异常状态的 Pod
#      (删除后 K8s 控制器会自动用最新本地镜像重建它们！)
#   3. 强制清除长期卡在 Terminating (删除中) 状态的顽固 Pod
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

# 默认命名空间（优先服务您的项目）
NAMESPACE="anime-role-detect"
DRY_RUN=false

# 参数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --all-namespaces)
            NAMESPACE="all"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  -n, --namespace <ns>    指定命名空间 (默认: anime-role-detect)"
            echo "  --all-namespaces        清理所有命名空间下的错误 Pod"
            echo "  --dry-run               只打印将要清理的 Pod，不执行实际删除"
            exit 0
            ;;
        *)
            err "未知参数: $1"
            ;;
    esac
done

if ! command -v kubectl &>/dev/null; then
    err "未找到 kubectl 命令，请确保已安装并配置好 K8s 环境。"
fi

if [[ "$DRY_RUN" == true ]]; then
    warn "⚠️  当前处于 [DRY-RUN] 预览模式，仅打印资源，不会执行实际删除动作"
fi

# ======================== 1. 清理 Evicted (已驱逐) 的 Pod ========================
echo ""
info "--- 1. 检索 Evicted (已驱逐) 状态的 Pod ---"
if [[ "$NAMESPACE" == "all" ]]; then
    EVICTED_PODS=$(kubectl get pods -A --no-headers 2>/dev/null | awk '$4 == "Evicted" {print $1, $2}') || true
else
    EVICTED_PODS=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '$3 == "Evicted" {print "'"$NAMESPACE"'", $1}') || true
fi

if [[ -z "$EVICTED_PODS" ]]; then
    ok "没有发现已驱逐 (Evicted) 的 Pod"
else
    echo "$EVICTED_PODS" | while read -r ns pod; do
        if [[ "$DRY_RUN" == true ]]; then
            echo -e "  ${YELLOW}[预览]${NC} 将删除 Evicted Pod: $pod (Namespace: $ns)"
        else
            info "  正在删除 Evicted Pod: $pod (Namespace: $ns)..."
            kubectl delete pod "$pod" -n "$ns" --wait=false >/dev/null 2>&1 || true
        fi
    done
fi

# ======================== 2. 清理/重置 各种报错状态的 Pod ========================
echo ""
info "--- 2. 检索 各种报错/镜像拉取失败 状态的 Pod ---"
# 过滤：Error, Failed, CrashLoopBackOff, ImagePullBackOff, ErrImagePull, CreateContainerConfigError 等
if [[ "$NAMESPACE" == "all" ]]; then
    ERR_PODS=$(kubectl get pods -A --no-headers 2>/dev/null | awk '$4 == "Error" || $4 == "Failed" || $4 == "CrashLoopBackOff" || $4 == "ImagePullBackOff" || $4 == "ErrImagePull" || $4 == "CreateContainerConfigError" || $4 == "CreateContainerError" {print $1, $2}') || true
else
    ERR_PODS=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '$3 == "Error" || $3 == "Failed" || $3 == "CrashLoopBackOff" || $3 == "ImagePullBackOff" || $3 == "ErrImagePull" || $3 == "CreateContainerConfigError" || $3 == "CreateContainerError" {print "'"$NAMESPACE"'", $1}') || true
fi

if [[ -z "$ERR_PODS" ]]; then
    ok "没有发现处于 Error/Failed/CrashLoop/ImagePull 状态的 Pod"
else
    echo "$ERR_PODS" | while read -r ns pod; do
        if [[ "$DRY_RUN" == true ]]; then
            echo -e "  ${YELLOW}[预览]${NC} 将删除并触发重建: $pod (Namespace: $ns)"
        else
            info "  正在重置异常 Pod: $pod (Namespace: $ns)..."
            # 使用 --wait=false 加快并行处理速度
            kubectl delete pod "$pod" -n "$ns" --wait=false >/dev/null 2>&1 || true
        fi
    done
fi

# ======================== 3. 强力清除卡在 Terminating 的 Pod ========================
echo ""
info "--- 3. 检索长期卡在 Terminating (删除中) 状态的 Pod ---"
if [[ "$NAMESPACE" == "all" ]]; then
    TERM_PODS=$(kubectl get pods -A --no-headers 2>/dev/null | awk '$4 == "Terminating" {print $1, $2}') || true
else
    TERM_PODS=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '$3 == "Terminating" {print "'"$NAMESPACE"'", $1}') || true
fi

if [[ -z "$TERM_PODS" ]]; then
    ok "没有发现卡在 Terminating 状态的 Pod"
else
    echo "$TERM_PODS" | while read -r ns pod; do
        if [[ "$DRY_RUN" == true ]]; then
            echo -e "  ${YELLOW}[预览]${NC} 将强制删除 Terminating Pod: $pod (Namespace: $ns)"
        else
            warn "  正在【强制】删除 Terminating Pod: $pod (Namespace: $ns)..."
            # 使用 --force --grace-period=0 强行摘除
            kubectl delete pod "$pod" -n "$ns" --force --grace-period=0 >/dev/null 2>&1 || true
        fi
    done
fi

echo ""
if [[ "$DRY_RUN" == true ]]; then
    ok "预览结束。如果确认无误，请去掉 '--dry-run' 参数实际运行。"
else
    ok "一键清理/重置完成！Deployment 控制器将自动调度并加载本地新镜像。"
    info "您可运行 'kubectl get pods -n ${NAMESPACE}' 查看重建状态。"
fi