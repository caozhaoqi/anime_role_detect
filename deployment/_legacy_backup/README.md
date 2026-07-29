# Legacy Backup — 旧部署配置归档

本目录保存**已被取代**的旧部署配置文件，统一归档于此，便于追溯历史，不再被任何 CI / 部署流程引用。

## 迁移背景
- 项目原先有两套并行 K8s manifests：`deployment/`（本目录下的 `k8s-*.yaml` + `kustomization.yaml`）与新建的 `k8s/`。
- 两套都创建同 namespace `anime-role-detect` 与同名资源，且 Secret 命名不同（`ardc-secrets` vs `ard-secret`），互相 `apply` 会冲突/覆盖。
- 2026-07-29 决定以 **`k8s/` 为唯一权威部署源**，本目录内容归档备份。

## 被归档的文件
### K8s manifests（旧）
- `k8s-deploy.yaml` / `k8s-deployments.yaml` / `k8s-services.yaml` / `k8s-volumes.yaml`
- `k8s-ingress.yaml` / `k8s-hpa.yaml` / `k8s-pdb.yaml` / `k8s-logging.yaml`
- `kustomization.yaml`（旧 CI overlay：降配 + emptyDir）

### 未引用的 Dockerfile（无人引用，仅旧文档提及）
- `Dockerfile.backend` / `Dockerfile.health-check` / `Dockerfile.log-monitor`
- `Dockerfile.log-viewer` / `Dockerfile.monitor-dashboard` / `Dockerfile.resource-monitor`

## 当前权威部署源
- 生产/本地：`k8s/`（kustomize 根，`kubectl apply -k k8s/`）
- CI 测试：`k8s/overlays/ci/`（kind 集群降配 overlay）
- 镜像构建：仍使用 `deployment/Dockerfile.*`（base / ml-base / 各服务）

> 注意：`deploy/logging/`（fluent-bit 配置）仍被 `docker-compose.yml` 引用，**未**移入此备份目录。
