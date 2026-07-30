#!/usr/bin/env bash
# ============================================================================
# 本地 K8s 验证脚本 (kind + colima / OrbStack / Docker Desktop on macOS)
#
# 默认 smoke 模式：
#   - 真实运行 redis / rabbitmq / mysql / fluent-bit 基础设施（验证健康/端口/探针）
#   - 用 nginx:alpine 占位 ardc/* 应用镜像，仅用于验证 manifests 机制
#     （POD_NAME/NODE_NAME 向下 API、探针端点、端口匹配、服务可应用）
#   - 应用 Pod 因占位镜像不会 Ready（探针打 /live 但 nginx 没这路由），属预期
#
# --build 模式：本地构建真实 ardc/* 镜像（含 torch，较重/较慢）后 load 进 kind。
#
# 前置：
#   1) colima start --cpu 6 --memory 8   （或 Docker Desktop / OrbStack，需 >=4 CPU）
#   2) brew install kind kubectl
#   3) 终端已设置 HTTPS_PROXY/HTTP_PROXY（仅供本机 docker pull；脚本建 kind 前会清除，
#      避免 kind 节点继承 127.0.0.1 代理导致镜像拉取失败）
#
# 用法：
#   ./scripts/k8s/verify_local.sh                # smoke（默认）
#   ./scripts/k8s/verify_local.sh --build        # 构建真实 ardc/* 镜像
#   ./scripts/k8s/verify_local.sh --no-cleanup   # 保留集群手动排查
# ============================================================================
set -uo pipefail

CLUSTER="ard-ci"
NAMESPACE="anime-role-detect"
OVERLAY="k8s/overlays/ci"
CLEANUP=1
MODE="smoke"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-cleanup) CLEANUP=0 ;;
    --build)      MODE="build" ;;
    --smoke)      MODE="smoke" ;;
    -h|--help)    sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
  shift
done

log()  { echo -e "\033[32m[INFO]\033[0m  $*"; }
warn() { echo -e "\033[33m[WARN]\033[0m  $*"; }
die()  { echo -e "\033[31m[FAIL]\033[0m  $*" >&2; exit 1; }

docker info >/dev/null 2>&1 || die "docker daemon 未运行，请先 colima start / 启动 Docker Desktop"
command -v kind >/dev/null    || die "未安装 kind：brew install kind"
command -v kubectl >/dev/null || die "未安装 kubectl：brew install kubectl"

# ---- 镜像清单 ----
INFRA_IMAGES=(redis:7-alpine rabbitmq:3-management-alpine mysql:8.0 fluent/fluent-bit:3.0)
HELPER_IMAGES=(nginx:alpine busybox:1.36)
ALL_IMAGES=("${INFRA_IMAGES[@]}" "${HELPER_IMAGES[@]}")

# ---- 本机预拉取（此时代理仍可用）----
# 增加重试：Docker Hub 经代理常遇瞬时 502，重试可恢复
pull_with_retry() {
  local img="$1" tries=3 i
  for ((i=1; i<=tries; i++)); do
    if docker pull "$img" >/dev/null 2>&1; then
      return 0
    fi
    warn "  pull 失败 (尝试 $i/$tries): $img —— 可能是 Docker Hub 瞬时 502，3s 后重试"
    sleep 3
  done
  warn "  pull 最终失败: ${img}（将跳过其 kind load，不阻塞其他镜像）"
  return 1
}
log "本机预拉取基础镜像（使用当前代理 + 重试）..."
for img in "${ALL_IMAGES[@]}"; do
  if docker image inspect "$img" >/dev/null 2>&1; then
    log "  已存在: $img"
  else
    log "  pull: $img"
    pull_with_retry "$img" || true
  fi
done

# ---- --build：本地构建真实应用镜像 ----
if [[ "$MODE" == "build" ]]; then
  warn "build 模式：将构建 ardc/* 镜像（含 torch，较重/较慢）"
  SERVICES=(api-gateway api-service model-service multimedia-service search-service search-worker inference-worker monitoring frontend)
  for svc in "${SERVICES[@]}"; do
    log "  build ardc/$svc ..."
    docker build -f "deployment/Dockerfile.$svc" -t "ardc/$svc:latest" . \
      || warn "构建失败: ${svc}（检查 deployment/DockerFile.${svc}）"
  done
fi

# ---- 关键：清除代理，避免 kind 节点继承 127.0.0.1 代理 ----
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy NO_PROXY no_proxy
log "已清除代理环境变量（kind 节点将不走 127.0.0.1 代理）"

# ---- kind 集群 ----
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  warn "集群 $CLUSTER 已存在，复用"
else
  log "创建 kind 集群 $CLUSTER ..."
  kind create cluster --name "$CLUSTER" --wait 120s
fi

# ---- 离线加载镜像（节点无代理，直接 load）----
# 逐镜像加载：单个镜像缺失/损坏不应阻塞其他镜像
# （之前用整批 `kind load ... ALL_IMAGES` 时，rabbitmq 拉取失败导致整批 load 中止，
#   redis/mysql/fluent-bit/nginx 全部没进 kind，造成所有 Pod ImagePullBackOff）
#
# 重要的坑：colima 的 Docker 守护进程默认用 containerd 快照驱动
#   (driver-type: io.containerd.snapshotter.v1)，此时 `docker save` 产出的 tar 不一致，
#   `kind load docker-image` / `kind load image-archive` 会失败
#   （ctr: content digest ... not found），或只按 digest 导入而丢失 tag，
#   导致 Pod 写 image: redis:7-alpine 仍拉不到。
# 策略：优先原生 `kind load`（Docker Desktop / overlay2 图驱动可用）；
#       失败则回退 `skopeo copy`(从 Docker 守护进程导出一致 tar)
#       -> 经 stdin 管道 `ctr images import`(保留 RepoTags tag)。
DOCKER_SOCK=$(docker context inspect colima --format '{{ .Endpoints.docker.Host }}' 2>/dev/null)
[ -z "$DOCKER_SOCK" ] && DOCKER_SOCK="${DOCKER_HOST:-unix:///var/run/docker.sock}"
HAS_SKOPEO=0; command -v skopeo >/dev/null 2>&1 && HAS_SKOPEO=1
SKOPEO_HINTED=0

# 镜像转完全限定 tag（供 ctr tag 兜底）
fq_tag() {
  local img="$1" name tag
  if [[ "$img" == *:* ]]; then name="${img%:*}"; tag="${img##*:}"; else name="$img"; tag="latest"; fi
  case "$name" in
    */*) echo "docker.io/$name:$tag" ;;
    *)   echo "docker.io/library/$name:$tag" ;;
  esac
}

load_image() {
  local img="$1" tar dig
  if ! docker image inspect "$img" >/dev/null 2>&1; then
    warn "  本地缺失，跳过 load: $img"; return 0
  fi
  # 1) 优先原生 kind load
  if kind --name "$CLUSTER" load docker-image "$img" >/dev/null 2>&1; then
    log "  loaded (kind): $img"; return 0
  fi
  # 2) 回退：skopeo 导出 -> ctr import（保留 tag）
  if [[ "$HAS_SKOPEO" -eq 1 ]]; then
    tar="$(mktemp -t kindload.XXXXXX).tar"
    if skopeo copy --src-daemon-host "$DOCKER_SOCK" "docker-daemon:$img" "docker-archive:$tar" >/dev/null 2>&1; then
      if docker exec -i "$CLUSTER-control-plane" ctr -n k8s.io images import - < "$tar" >/dev/null 2>&1; then
        log "  loaded (skopeo+ctr): $img"
        # 兜底：ctr import 偶发不应用 RepoTags，显式补 tag
        dig=$(tar -xf "$tar" -O manifest.json 2>/dev/null \
              | grep -o '"Config"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 \
              | grep -o '[0-9a-f]\{64\}')
        if [[ -n "$dig" ]]; then
          docker exec "$CLUSTER-control-plane" ctr -n k8s.io images tag "sha256:$dig" "$(fq_tag "$img")" >/dev/null 2>&1 || true
          docker exec "$CLUSTER-control-plane" ctr -n k8s.io images tag "sha256:$dig" "$img" >/dev/null 2>&1 || true
        fi
        rm -f "$tar"; return 0
      else
        warn "  ctr import 失败: $img"
      fi
    else
      warn "  skopeo 导出失败: $img"
    fi
    rm -f "$tar"
  else
    if [[ "$SKOPEO_HINTED" -eq 0 ]]; then
      warn "  提示：kind load 失败且未安装 skopeo；colima 用户请 'brew install skopeo' 以绕过 containerd 快照驱动的 docker save 问题"
      SKOPEO_HINTED=1
    fi
  fi
  warn "  load 最终失败: $img"
}

log "加载镜像到 kind（逐镜像，缺失不阻塞）..."
for img in "${ALL_IMAGES[@]}"; do
  load_image "$img"
done
if [[ "$MODE" == "build" ]]; then
  for svc in "${SERVICES[@]}"; do
    load_image "ardc/$svc:latest"
  done
fi

# ---- 资源预检 ----
AVAIL_CPU=$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.cpu}' 2>/dev/null)
log "kind 节点可分配 CPU: ${AVAIL_CPU:-未知}"
if [[ "$AVAIL_CPU" == *m ]]; then cpu_m=${AVAIL_CPU%m}; else cpu_m=$(( ${AVAIL_CPU:-0} * 1000 )); fi
if (( cpu_m < 4000 )); then
  warn "节点 CPU 偏低（<4 核），部分 Pod 可能因 Insufficient cpu 无法调度。"
  warn "请重启 colima 扩容： colima stop && colima start --cpu 6 --memory 8"
fi

# ---- 渲染 manifest ----
log "kubectl kustomize 校验 (${OVERLAY}) ..."
kubectl kustomize "$OVERLAY" >/tmp/ci_render.yaml || die "kustomize 渲染失败"

if [[ "$MODE" == "smoke" ]]; then
  log "smoke 模式：ardc/* 应用镜像替换为 nginx:alpine（仅验证 manifests 机制）"
  sed -i '' -E 's#image: ardc/[^:]+:latest#image: nginx:alpine#g' /tmp/ci_render.yaml
fi
MANIFEST=/tmp/ci_render.yaml

# ---- apply ----
log "apply manifests ..."
kubectl apply -f "$MANIFEST"

# ---- 等待 ----
log "等待 90s 让 Pod 调度/启动 ..."
sleep 90

log "=== Pod 状态 ==="
kubectl -n "$NAMESPACE" get pods -o wide

log "=== 未就绪 Pod 事件（定位调度/镜像问题）==="
kubectl -n "$NAMESPACE" get pods --field-selector=status.phase!=Running -o name 2>/dev/null | while read p; do
  echo "--- $p ---"
  kubectl -n "$NAMESPACE" describe "$p" 2>/dev/null | awk '/Events:/{f=1} f{print} f&&/^[A-Z].*:/{exit}' | head -12
done

# ---- POD_NAME / NODE_NAME 向下 API 注入（读 deploy spec，无需 Pod 运行）----
log "=== 验证 POD_NAME / NODE_NAME 向下 API 注入 ==="
for dep in redis rabbitmq mysql api-gateway api-service model-service frontend monitoring; do
  if kubectl -n "$NAMESPACE" get deploy "$dep" >/dev/null 2>&1; then
    pv=$(kubectl -n "$NAMESPACE" get deploy "$dep" -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="POD_NAME")].valueFrom.fieldRef.fieldPath}' 2>/dev/null)
    nv=$(kubectl -n "$NAMESPACE" get deploy "$dep" -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="NODE_NAME")].valueFrom.fieldRef.fieldPath}' 2>/dev/null)
    if [[ "$pv" == "metadata.name" && "$nv" == "spec.nodeName" ]]; then
      log "$dep: 向下 API OK (POD_NAME=$pv, NODE_NAME=$nv)"
    else
      warn "$dep: 注入异常 pv=[$pv] nv=[$nv]"
    fi
  fi
done

# ---- 探针端点（来自渲染后的 manifest，不依赖运行）----
log "=== 探针端点（应为 liveness=/live readiness=/ready，frontend 用 /）==="
grep -nE "path: /(live|ready)?" /tmp/ci_render.yaml | head -20

# ---- 服务连通性（仅对真实运行的 infra）----
log "=== 验证基础服务端口可达（redis/rabbitmq/mysql）==="
for svc in redis:6379 rabbitmq:5672 mysql:3306; do
  name=${svc%%:*}; port=${svc##*:}
  pod=$(kubectl -n "$NAMESPACE" get pod -l "app=$name" -o name 2>/dev/null | head -1)
  if [[ -z "$pod" ]]; then
    warn "$name: 无 Pod（镜像/资源原因），跳过端口探测"
    continue
  fi
  ok=0
  for attempt in 1 2; do
    if kubectl -n "$NAMESPACE" run netcheck-$name --rm -i --restart=Never --image=busybox:1.36 --timeout=40s -- \
        sh -c "nc -zv $name $port" 2>/dev/null; then
      ok=1; break
    fi
    [[ $attempt -eq 1 ]] && { warn "  $name:$port 首次不可达，5s 后重试..."; sleep 5; }
  done
  if [[ $ok -eq 1 ]]; then log "$name:$port reachable"; else warn "$name:$port 不可达或尚未就绪"; fi
done

# ---- 验证小结 ----
log "=== 验证小结 ==="
total=$(kubectl -n "$NAMESPACE" get pods 2>/dev/null | awk 'NR>1' | wc -l | tr -d ' ')
running=$(kubectl -n "$NAMESPACE" get pods 2>/dev/null | awk 'NR>1 && $3=="Running"' | wc -l | tr -d ' ')
log "  Pod 运行(Running 阶段): $running / $total"
log "  （smoke 模式下 ardc/* 用 nginx:alpine 占位，探针无 /live 故不会 Ready，属预期）"

# ---- 清理 ----
if [[ "$CLEANUP" -eq 1 ]]; then
  log "清理集群 ..."
  kind delete cluster --name "$CLUSTER"
  log "完成：集群已删除"
else
  warn "保留集群 $CLUSTER。手动删除：kind delete cluster --name $CLUSTER"
fi
