#!/bin/bash
# =============================================================================
# ARD K8s 一键部署脚本 (Ubuntu)
#
# 功能：
#   1. 安装 Docker + k3s（轻量 K8s）
#   2. 从 OSS 下载镜像 tar.gz 并加载到容器运行时
#   3. 按顺序部署 K8s 资源
#
# 用法：
#   ./deploy_ubuntu.sh --tag c8cd26b                        # 从 OSS 下载并部署
#   ./deploy_ubuntu.sh --tag c8cd26b --local               # 使用本地已有 tar.gz
#   ./deploy_ubuntu.sh --tag c8cd26b --skip-k8s            # 跳过 K8s 安装
#   ./deploy_ubuntu.sh --tag c8cd26b --only-infra          # 只部署基础设施
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

# ======================== 参数解析 ========================
TAG=""
REGISTRY="ardc"
LOCAL_MODE=false
SKIP_K8S=false
SKIP_DOCKER=false
ONLY_INFRA=false
DEPLOY_DIR="/opt/ardc"
EXPORT_DIR="/tmp/docker-images-export"
OSS_BUCKET="colllect-zip"
OSS_ENDPOINT="oss-cn-wulanchabu.aliyuncs.com"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --tag=*)        TAG="${1#*=}"; shift ;;
        --tag)          TAG="$2"; shift 2 ;;
        --registry=*)   REGISTRY="${1#*=}"; shift ;;
        --local)        LOCAL_MODE=true; shift ;;
        --skip-k8s)     SKIP_K8S=true; shift ;;
        --skip-docker)  SKIP_DOCKER=true; shift ;;
        --only-infra)   ONLY_INFRA=true; shift ;;
        --deploy-dir=*) DEPLOY_DIR="${1#*=}"; shift ;;
        --export-dir=*) EXPORT_DIR="${1#*=}"; shift ;;
        -h|--help)
            echo "用法: $0 --tag <commit_sha> [选项]"
            echo ""
            echo "选项:"
            echo "  --tag <sha>        镜像标签 (git commit short hash)"
            echo "  --registry <name>  镜像前缀 (默认: ardc)"
            echo "  --local            使用本地已有的 tar.gz，不从 OSS 下载"
            echo "  --skip-k8s         跳过 K8s 安装（已安装时使用）"
            echo "  --skip-docker      跳过 Docker 安装"
            echo "  --only-infra       只部署基础设施 (Redis/MySQL/RabbitMQ)"
            echo "  --deploy-dir <dir> 部署文件目录 (默认: /opt/ardc)"
            echo "  --export-dir <dir> 镜像 tar.gz 目录 (默认: /tmp/docker-images-export)"
            exit 0
            ;;
        *) err "未知参数: $1" ;;
    esac
done

[[ -z "$TAG" ]] && err "请指定 --tag 参数，例如: --tag c8cd26b"

SERVICES=(api-service model-service frontend api-gateway
          multimedia-service search-service search-worker inference-worker monitoring)

# ======================== Phase 1: 安装 Docker ========================
install_docker() {
    if command -v docker &>/dev/null; then
        ok "Docker 已安装: $(docker --version)"
        return
    fi

    info "安装 Docker..."
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl gnupg lsb-release > /dev/null

    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg 2>/dev/null
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list

    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin > /dev/null

    systemctl enable --now docker
    ok "Docker 安装完成: $(docker --version)"
}

# ======================== Phase 2: 安装 k3s ========================
install_k3s() {
    if command -v k3s &>/dev/null; then
        ok "k3s 已安装: $(k3s --version | head -1)"
        check_k3s_status
        setup_kubectl
        return
    fi

    info "检测网络连通性..."
    
    local k3s_url="https://get.k3s.io"
    local github_url="https://github.com"
    local aliyun_url="https://rancher-mirror.oss-cn-beijing.aliyuncs.com/k3s"
    
    local install_url="$k3s_url"
    local mirror_env=""
    
    if curl -fsSL --max-time 5 "$github_url" &>/dev/null; then
        info "github.com 可访问，使用官方源"
        install_url="$k3s_url"
        mirror_env=""
    else
        warn "github.com 不可访问，启用阿里云镜像加速..."
        install_url="$k3s_url"
        mirror_env="INSTALL_K3S_MIRROR=cn"
    fi

    info "安装 k3s (源: ${install_url})..."
    info "安装参数: INSTALL_K3S_EXEC=\"--docker --write-kubeconfig-mode 644\" ${mirror_env}"
    
    local max_retries=3
    local retry_delay=30
    
    for attempt in $(seq 1 $max_retries); do
        info "安装尝试 ${attempt}/${max_retries}..."
        
        if timeout 300 \
           HTTPS_PROXY="${HTTPS_PROXY:-}" HTTP_PROXY="${HTTP_PROXY:-}" NO_PROXY="${NO_PROXY:-}" \
           curl -sfL "$install_url" | \
           INSTALL_K3S_EXEC="--docker --write-kubeconfig-mode 644" \
           ${mirror_env} \
           sh -; then
            
            info "k3s 安装脚本执行完成"
            break
        else
            local exit_code=$?
            warn "安装尝试 ${attempt}/${max_retries} 失败 (退出码: ${exit_code})"
            
            if [[ $attempt -lt $max_retries ]]; then
                warn "等待 ${retry_delay} 秒后重试..."
                sleep $retry_delay
            else
                err "k3s 安装失败，已重试 ${max_retries} 次"
            fi
        fi
    done

    check_k3s_status
    setup_kubectl
    ok "k3s 安装完成: $(k3s --version | head -1)"
}

check_k3s_status() {
    info "等待 k3s 服务就绪..."
    
    if ! systemctl is-active --quiet k3s 2>/dev/null; then
        info "k3s 服务未运行，启动中..."
        systemctl start k3s 2>/dev/null || true
        sleep 10
    fi
    
    local ready_timeout=120
    local ready_interval=2
    local elapsed=0
    
    while [[ $elapsed -lt $ready_timeout ]]; do
        if k3s kubectl get nodes &>/dev/null; then
            ok "k3s 集群就绪"
            return
        fi
        
        if [[ $((elapsed % 10)) -eq 0 ]]; then
            info "等待 k3s 就绪... (已等待 ${elapsed}s)"
        fi
        
        sleep $ready_interval
        elapsed=$((elapsed + ready_interval))
    done
    
    warn "k3s 集群启动超时 (${ready_timeout}s)，检查服务状态..."
    systemctl status k3s 2>/dev/null || true
    journalctl -u k3s --no-pager -n 30 2>/dev/null || true
}

setup_kubectl() {
    if [[ -f /etc/rancher/k3s/k3s.yaml ]]; then
        export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
        mkdir -p "$HOME/.kube"
        cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config" 2>/dev/null || true
        info "kubectl 已配置 (KUBECONFIG=$KUBECONFIG)"
    fi

    if ! command -v kubectl &>/dev/null; then
        if [[ -f /usr/local/bin/k3s ]]; then
            ln -sf /usr/local/bin/k3s /usr/local/bin/kubectl 2>/dev/null || true
        fi
    fi
}

# ======================== Phase 3: 下载/加载镜像 ========================
download_from_oss() {
    info "从 OSS 下载镜像..."
    mkdir -p "$EXPORT_DIR"

    if ! command -v ossutil64 &>/dev/null && ! command -v ossutil &>/dev/null; then
        info "安装 ossutil..."
        curl -fsSL -o /usr/local/bin/ossutil64 \
            "https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64/ossutil64"
        chmod +x /usr/local/bin/ossutil64
    fi

    local oss_cmd="ossutil64"
    command -v ossutil64 &>/dev/null || oss_cmd="ossutil"

    for svc in "${SERVICES[@]}"; do
        local tar_name="${svc}-${TAG}.tar.gz"
        local tar_path="${EXPORT_DIR}/${tar_name}"
        local oss_key="oss://${OSS_BUCKET}/docker-images/${TAG}/${tar_name}"

        if [[ -f "$tar_path" ]]; then
            info "  已存在，跳过: ${tar_name}"
            continue
        fi

        info "  下载 ${tar_name}..."
        if $oss_cmd cp "$oss_key" "$tar_path" -f 2>/dev/null; then
            ok "  ${tar_name} 下载完成"
        else
            warn "  ${tar_name} 下载失败 (可能不存在)"
        fi
    done
}

load_images() {
    info "加载镜像到容器运行时..."
    local existing=0
    local retagged=0
    local imported=0

    local old_errexit=$(set +o | grep errexit | awk '{print $2}')
    set +e

    for svc in "${SERVICES[@]}"; do
        local image_name="${REGISTRY}/${svc}:${TAG}"
        local latest_image="${REGISTRY}/${svc}:latest"

        local tag_count=$(docker images "${REGISTRY}/${svc}" --format "{{.Tag}}" 2>/dev/null | grep -c "^${TAG}$" || echo 0)
        local latest_count=$(docker images "${REGISTRY}/${svc}" --format "{{.Tag}}" 2>/dev/null | grep -c "^latest$" || echo 0)

        if [[ $tag_count -gt 0 ]]; then
            info "  Docker 镜像已存在: ${image_name}"
            ((existing++))
            if [[ $latest_count -eq 0 ]]; then
                info "  添加 latest 标签..."
                docker tag "$image_name" "$latest_image" 2>/dev/null || true
            fi
            
            info "  导入到 containerd..."
            docker save "$image_name" | ctr -n k8s.io images import -
            docker save "$latest_image" | ctr -n k8s.io images import - 2>/dev/null || true
            ((imported++))
            continue
        fi

        if [[ $latest_count -gt 0 ]]; then
            info "  发现 latest 镜像，重新打 ${TAG} 标签: ${latest_image}"
            docker tag "$latest_image" "$image_name" 2>/dev/null || true
            ((retagged++))
            
            info "  导入到 containerd..."
            docker save "$image_name" | ctr -n k8s.io images import -
            docker save "$latest_image" | ctr -n k8s.io images import - 2>/dev/null || true
            ((imported++))
            continue
        fi

        warn "  镜像 ${image_name} 不存在，跳过"
    done

    if [[ "$old_errexit" == "on" ]]; then
        set -e
    fi

    ok "共 ${existing} 个镜像已存在，${retagged} 个重新打标签，${imported} 个导入到 containerd"
}

# ======================== Phase 4: 部署 K8s ========================
deploy_k8s() {
    local manifest_dir="${PROJECT_DIR}/deployment"

    if [[ ! -d "$manifest_dir" ]]; then
        manifest_dir="${DEPLOY_DIR}/deployment"
    fi

    if [[ ! -d "$manifest_dir" ]]; then
        err "部署文件目录不存在: $manifest_dir\n  请先将项目代码复制到 $DEPLOY_DIR 或在项目目录下运行脚本"
    fi

    info "使用部署文件目录: $manifest_dir"
    export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}

    info "创建 Namespace..."
    kubectl create namespace anime-role-detect --dry-run=client -o yaml | kubectl apply -f -

    if [[ "$ONLY_INFRA" == true ]]; then
        info "仅部署基础设施..."
        kubectl apply -f "${manifest_dir}/k8s-deploy.yaml"
        kubectl apply -f "${manifest_dir}/k8s-volumes.yaml"
        kubectl apply -f "${manifest_dir}/k8s-services.yaml"
        deploy_infra_only "$manifest_dir"
        return
    fi

    info "Step 1/5: 基础配置 (Namespace/ConfigMap/Secret)..."
    kubectl apply -f "${manifest_dir}/k8s-deploy.yaml"

    info "Step 2/5: 持久化存储 (PVC)..."
    kubectl apply -f "${manifest_dir}/k8s-volumes.yaml"

    info "Step 3/5: Services..."
    kubectl apply -f "${manifest_dir}/k8s-services.yaml"

    info "Step 4/5: Deployments..."
    kubectl apply -f "${manifest_dir}/k8s-deployments.yaml"

    info "Step 5/5: Ingress + HPA + PDB..."
    kubectl apply -f "${manifest_dir}/k8s-ingress.yaml" 2>/dev/null || true
    kubectl apply -f "${manifest_dir}/k8s-hpa.yaml" 2>/dev/null || true
    kubectl apply -f "${manifest_dir}/k8s-pdb.yaml" 2>/dev/null || true

    ok "K8s 资源已全部提交"
}

deploy_infra_only() {
    local manifest_dir="$1"

    info "等待基础设施 Pod 就绪..."
    kubectl -n anime-role-detect wait --for=condition=ready pod -l app=redis --timeout=120s 2>/dev/null || true
    kubectl -n anime-role-detect wait --for=condition=ready pod -l app=mysql --timeout=120s 2>/dev/null || true
    kubectl -n anime-role-detect wait --for=condition=ready pod -l app=rabbitmq --timeout=120s 2>/dev/null || true
    ok "基础设施就绪"
}

# ======================== Phase 5: 状态检查 ========================
wait_for_pods() {
    export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
    
    info "等待关键服务 Pod 就绪..."
    
    local services=(frontend api-gateway api-service)
    local timeout=300
    local interval=5
    local elapsed=0
    
    while [[ $elapsed -lt $timeout ]]; do
        local ready_count=0
        local total_count=${#services[@]}
        
        for svc in "${services[@]}"; do
            local pod_status=$(kubectl -n anime-role-detect get pods -l app=${svc} -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
            if [[ "$pod_status" == "True" ]]; then
                ((ready_count++))
            fi
        done
        
        if [[ $ready_count -eq $total_count ]]; then
            ok "所有关键服务就绪 (${ready_count}/${total_count})"
            return
        fi
        
        info "等待中... 就绪: ${ready_count}/${total_count} (已等待 ${elapsed}s)"
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    
    warn "等待超时 (${timeout}s)，部分服务可能未就绪"
}

get_node_ip() {
    export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
    
    local node_ip
    node_ip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}' 2>/dev/null)
    
    if [[ -z "$node_ip" ]]; then
        node_ip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null)
    fi
    
    if [[ -z "$node_ip" ]]; then
        node_ip=$(hostname -I | awk '{print $1}')
    fi
    
    echo "$node_ip"
}

show_status() {
    export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}

    echo ""
    echo "========================================="
    echo "  部署状态"
    echo "========================================="
    echo ""

    echo "--- Pods ---"
    kubectl -n anime-role-detect get pods -o wide 2>/dev/null || echo "  (无资源或命名空间未创建)"
    echo ""

    echo "--- Services ---"
    kubectl -n anime-role-detect get svc 2>/dev/null || echo "  (无资源或命名空间未创建)"
    echo ""

    echo "--- PVC ---"
    kubectl -n anime-role-detect get pvc 2>/dev/null || echo "  (无资源或命名空间未创建)"
    echo ""

    local node_ip=$(get_node_ip)
    local frontend_port="30000"
    local gateway_port="30080"

    echo "========================================="
    echo "  🌐 直接访问地址（浏览器打开）"
    echo "========================================="
    echo ""
    echo "  🎨 前端应用:"
    echo "     http://${node_ip}:${frontend_port}"
    echo ""
    echo "  🔌 API 网关:"
    echo "     http://${node_ip}:${gateway_port}"
    echo "     http://${node_ip}:${gateway_port}/docs    (API 文档)"
    echo "     http://${node_ip}:${gateway_port}/redoc   (API 文档)"
    echo ""
    echo "  📝 访问说明:"
    echo "     1. 确保服务器 ${node_ip} 的 ${frontend_port} 和 ${gateway_port} 端口已开放"
    echo "     2. 在浏览器中直接访问上述地址即可"
    echo "     3. 如果无法访问，请检查防火墙规则"
    echo ""
    echo "  🔧 常用命令:"
    echo "     kubectl -n anime-role-detect get pods          # 查看 Pod 状态"
    echo "     kubectl -n anime-role-detect logs <pod-name>   # 查看日志"
    echo "     kubectl -n anime-role-detect port-forward svc/frontend 3000:3000 &  # 端口转发"
    echo "========================================="
}

# ======================== 主流程 ========================
echo ""
echo "========================================="
echo "  ARD K8s 一键部署 (Ubuntu)"
echo "========================================="
echo "  TAG:        ${TAG}"
echo "  REGISTRY:   ${REGISTRY}"
echo "  LOCAL:      ${LOCAL_MODE}"
echo "  DEPLOY_DIR: ${DEPLOY_DIR}"
echo "  EXPORT_DIR: ${EXPORT_DIR}"
echo "========================================="
echo ""

if [[ "$SKIP_DOCKER" == false ]]; then
    echo "--- Phase 1: Docker ---"
    install_docker
    echo ""
fi

if [[ "$SKIP_K8S" == false ]]; then
    echo "--- Phase 2: K8s (k3s) ---"
    install_k3s
    echo ""
fi

echo "--- Phase 3: 镜像 ---"
if [[ "$LOCAL_MODE" == false ]]; then
    download_from_oss
fi
load_images
echo ""

echo "--- Phase 4: K8s 部署 ---"
deploy_k8s
echo ""

echo "--- Phase 5: 状态检查 ---"
wait_for_pods
show_status
