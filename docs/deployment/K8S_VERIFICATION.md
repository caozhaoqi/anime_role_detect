# 本地 K8s 验证报告

生成时间：2026-07-29
仓库：anime_role_detect
验证对象：`k8s/overlays/ci`（CI 测试 overlay，由 `.github/workflows/k8s-deploy-test.yml` apply）

## 0. 验证环境结论（关键）

| 项目 | 状态 |
|---|---|
| 本机 OS | macOS arm64 (Apple Silicon) |
| docker daemon | **未运行**（`/var/run/docker.sock` 不存在，无 Docker.app / OrbStack） |
| Docker Desktop / OrbStack | 未安装 |
| kind | 已通过 brew 安装 v0.32.0 |
| kubectl | v1.36.3（含 kustomize v5.8.1） |
| 真集群 live 验证 | **本沙箱不可行** —— kind 依赖 docker/podman 容器运行时，沙箱内无法启动 |

> 结论：无法在本会话内启动真实 K8s 集群做 live 验证。改用**离线静态校验**（kustomize 渲染 + 结构/语义断言），覆盖本次修改的全部关键变更；并附**真机一键验证脚本**供你在启动 Docker Desktop 后直接跑。

## 1. kustomize 渲染校验（已通过）

| 资源类型 | base | ci overlay |
|---|---|---|
| Namespace | 1 | 1 |
| ConfigMap | 2 | 2 |
| Secret | 1 | 1 |
| Service | 10 | 10 |
| Deployment | 13 | 13 |
| StatefulSet | 1 | 1 |
| DaemonSet | 1 | 1 |
| Ingress | 1 | 1 |
| PVC | 4 | 4 |
| ServiceAccount / ClusterRole / ClusterRoleBinding | 各 1 | 各 1 |
| **合计** | **37** | **37** |

两个集合均 `kubectl kustomize ...` exit 0，无循环引用 / 字段冲突。

## 2. CI overlay 关键变换（全部生效）

- [x] **model-service GPU 亲和已移除**：CI 中无 `nodeSelector` / `tolerations`（base 有 GPU nodeSelector，CI 必须去掉才能跑在普通节点）
- [x] **副本降配**：`api-gateway / api-service / inference-worker / frontend` 在 CI 中 replicas=1
- [x] **emptyDir 替换 PVC**：CI 中 5 个有状态卷改用 `emptyDir`（kind 单节点默认无动态 PVC provisioner），infra PVC 仍声明但不强制绑定
- [x] **POD_NAME / NODE_NAME 向下 API 注入**：全部 14 个工作负载（含 redis/rabbitmq/mysql 基础设施）均已注入，供 Resource Monitor Pod/Node 感知
- [x] **mysql 增加 startupProbe**：CI 中 mysql 有 `startupProbe`（避免慢启动被 liveness 误杀）

## 3. 服务/探针/配置一致性（全部通过）

- [x] **Service 端口一致**：10 个 Service 的 `targetPort` 全部能在对应工作负载容器 `containerPort` 中找到，0 mismatch
- [x] **探针端点规范**：8 个后端服务 `liveness=/live` + `readiness=/ready`（与 P0 探针改造一致）；frontend 探针为 `/`（Next.js 无 `/live` 路由，用根路径合理，**不改**）
- [x] **Secret 安全**：`ard-secret` 使用 `CHANGE_ME_*` 占位符，未泄露真实凭据（旧 `ardc-secrets` 明文 JWT 已被归档）
- [x] **imagePullPolicy**：`IfNotPresent`（11）/ 未指定（3），CI 不强制 Always，符合本地镜像策略
- [x] **CI workflow 指向正确**：`.github/workflows/k8s-deploy-test.yml` 已改为 `kubectl apply -k k8s/overlays/ci/`，触发路径含 `k8s/**`；其引用的 11 个 `deployment/Dockerfile.*` 全部存在（未被备份误删）

## 4. 活跃引用修复（已验证无残留）

- `README.md`、`scripts/k8s/deploy_final.sh`、`build_k8s_images.sh`、`create_pod.sh`、`fix_cluster.sh` 内的 `kubectl apply -f deployment/k8s-*.yaml` 已全部改为 `kubectl apply -k k8s/base`（或 CI overlay），无活跃文件指向已归档的旧 manifests（仅 `egg-info`/`logs`/`deliverables` 等生成物残留旧路径，无关）。

## 5. 真机 live 验证步骤（本机启动 Docker 后）

```bash
# 1. 启动 Docker Desktop（或 OrbStack），确认 docker info 正常
# 2. 赋予执行权限并运行一键脚本：
chmod +x scripts/k8s/verify_local.sh
./scripts/k8s/verify_local.sh            # 建集群→apply→等就绪→探测→自动清理
# 或保留集群排查：
./scripts/k8s/verify_local.sh --no-cleanup
```

脚本会验证：所有 Pod 调度、POD_NAME/NODE_NAME 注入、redis/rabbitmq/mysql 端口可达。

## 6. 已知限制 / 后续

- [ ] 本会话未做 live 集群验证（硬约束：无容器运行时）。建议在启用了 Docker 的开发机或 CI（GitHub Actions 自带 ubuntu + 可起 kind）跑一次完整验证。
- [ ] `ard-secret` 占位符需在真实部署前替换为真实凭据（JWT/Redis/MySQL/RabbitMQ）。
- [ ] 生产 `k8s/base` 依赖 GPU 节点（model-service 的 nodeSelector），CI overlay 已正确去 GPU 化。
- [ ] infra PVC（redis/rabbitmq/mysql）在 base 中声明；CI overlay 用 emptyDir 替换，生产环境需保留 PVC 或对接 StorageClass。
