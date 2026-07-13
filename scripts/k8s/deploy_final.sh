#!/bin/bash
# =============================================================================
# ARD K8s 最终部署脚本
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

TAG=$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)
REGISTRY="ardc"
DEPLOY_DIR="/opt/ardc"
EXPORT_DIR="/tmp/docker-images-export"

echo ""
echo "========================================="
echo "  ARD K8s 最终部署"
echo "========================================="
echo ""

echo "--- Step 1: 准备部署目录 ---"
info "创建部署目录..."
mkdir -p /opt/ardc/deployment

info "复制部署文件..."
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
DEPLOY_SRC="${PROJECT_ROOT}/deployment"

if [[ -d "$DEPLOY_SRC" ]]; then
    cp -rf "$DEPLOY_SRC"/* /opt/ardc/deployment/
else
    info "未找到项目部署目录，使用当前目录..."
    cp -rf deployment/* /opt/ardc/deployment/ 2>/dev/null || true
fi

info "验证部署文件..."
ls -la /opt/ardc/deployment/
ok "部署目录准备完成"
echo ""

echo "--- Step 2: 加载镜像 ---"
info "检查并使用 docker images 中的本地镜像..."

loaded=0
existing=0

for svc in base ml-base api-service model-service frontend api-gateway \
           multimedia-service search-service search-worker inference-worker monitoring; do
    image_name="${REGISTRY}/${svc}:${TAG}"
    
    if docker image inspect "$image_name" &>/dev/null; then
        info "  镜像已存在: ${image_name}"
        ((existing++))
    else
        tar_path="${EXPORT_DIR}/${svc}-${TAG}.tar.gz"
        if [[ -f "$tar_path" ]]; then
            info "  加载 ${svc}-${TAG}.tar.gz → ${image_name}..."
            if docker load -i "$tar_path" >/dev/null 2>&1; then
                ok "  ${image_name} 加载成功"
                ((loaded++))
            else
                warn "  ${image_name} 加载失败"
            fi
        else
            warn "  镜像 ${image_name} 不存在，tar.gz 文件也不存在"
        fi
    fi
done

info "添加 latest 标签..."
for svc in base ml-base api-service model-service frontend api-gateway \
           multimedia-service search-service search-worker inference-worker monitoring; do
    image_name="${REGISTRY}/${svc}:${TAG}"
    if docker image inspect "$image_name" &>/dev/null; then
        docker tag "$image_name" "${REGISTRY}/${svc}:latest" &>/dev/null || true
    fi
done

info "验证已加载的镜像..."
docker images | grep "${REGISTRY}/"
ok "镜像加载完成 (${existing} 个已存在, ${loaded} 个从 tar.gz 加载)"
echo ""

echo "--- Step 3: 部署 K8s 资源 ---"
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

info "创建 Namespace..."
kubectl create namespace anime-role-detect --dry-run=client -o yaml | kubectl apply -f -

info "部署 ConfigMap/Secret..."
kubectl apply -f "${DEPLOY_DIR}/deployment/k8s-deploy.yaml"

info "部署 PVC..."
kubectl apply -f "${DEPLOY_DIR}/deployment/k8s-volumes.yaml"

info "部署 Services..."
kubectl apply -f "${DEPLOY_DIR}/deployment/k8s-services.yaml"

info "部署 Deployments..."
kubectl apply -f "${DEPLOY_DIR}/deployment/k8s-deployments.yaml"

info "部署 Ingress/HPA/PDB..."
kubectl apply -f "${DEPLOY_DIR}/deployment/k8s-ingress.yaml" 2>/dev/null || true
kubectl apply -f "${DEPLOY_DIR}/deployment/k8s-hpa.yaml" 2>/dev/null || true
kubectl apply -f "${DEPLOY_DIR}/deployment/k8s-pdb.yaml" 2>/dev/null || true

info "部署 Logging (Elasticsearch + Kibana + Filebeat)..."
info "  设置内核参数 vm.max_map_count..."
sysctl -w vm.max_map_count=262144 2>/dev/null || true
echo "vm.max_map_count=262144" >> /etc/sysctl.conf 2>/dev/null || true
kubectl apply -f "${DEPLOY_DIR}/deployment/k8s-logging.yaml"

ok "K8s 资源部署完成"
echo ""

echo "--- Step 4: 等待 Pod 就绪 ---"
info "等待 Pod 启动 (60秒)..."
sleep 60

echo ""
echo "========================================="
echo "  部署状态"
echo "========================================="
echo ""

echo "--- Pods ---"
kubectl -n anime-role-detect get pods -o wide
echo ""

echo "--- Services ---"
kubectl -n anime-role-detect get svc
echo ""

echo "--- PVC ---"
kubectl -n anime-role-detect get pvc
echo ""

gateway_ip=$(kubectl -n anime-role-detect get svc api-gateway -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "N/A")
frontend_ip=$(kubectl -n anime-role-detect get svc frontend -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "N/A")

echo "========================================="
echo "  访问地址"
echo "========================================="
echo "  API Gateway:  http://${gateway_ip}:8080"
echo "  Frontend:     http://${frontend_ip}:3000"
echo ""
echo "  端口转发到本机:"
echo "    kubectl -n anime-role-detect port-forward svc/api-gateway 8080:8080 &"
echo "    kubectl -n anime-role-detect port-forward svc/frontend 3000:3000 &"
echo ""
echo "  然后访问: http://localhost:3000"
echo "========================================="
