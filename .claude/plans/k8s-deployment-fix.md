# K8s 部署修复计划

## 背景分析

通过对比 `docker-compose.yml`（已验证可用）、`k8s-deploy.yaml`、`src/core/config/service_config.py`、`.env.example` 以及各 Dockerfile，确定了 20 个问题。服务间依赖关系为：

```
frontend → api-gateway → api-service → model-service
                                  → search-service → search-worker
                                  → multimedia-service
基础设施：Redis, MySQL/PostgreSQL, RabbitMQ
```

各服务通过 `ServiceConfig` (pydantic-settings) 从环境变量读取 host/port，拼成 URL 进行服务发现。

## 修复方案

### P0 — 致命问题（部署无法运行）

#### 1. 合并并统一 K8s 配置

**操作**：
- 删除旧版 `backend-deployment.yaml`、`frontend-deployment.yaml`、`services.yaml`、`hpa.yaml`、`ingress.yaml`（使用旧命名规范，与 k8s-deploy.yaml 冲突）
- 将 `k8s-deploy.yaml` 拆分为多个文件便于维护：
  - `k8s-deploy.yaml` — Namespace + ConfigMap + Secret
  - `k8s-services.yaml` — 所有 Service 定义
  - `k8s-deployments.yaml` — 所有 Deployment 定义
  - `k8s-ingress.yaml` — Ingress 定义
  - `k8s-hpa.yaml` — HPA 定义
  - `k8s-pdb.yaml` — PodDisruptionBudget 定义
  - `k8s-volumes.yaml` — PVC 定义

**原因**：两套配置并存会导致误用，拆分后更容易维护和 review。

#### 2. 补全基础设施服务（Redis + MySQL + RabbitMQ）

在 `k8s-deployments.yaml` 中添加：
- **Redis** Deployment + Service（端口 6379，512Mi 内存限制）
- **MySQL** Deployment + Service（端口 3306，2Gi 内存限制，PVC 持久化）
- **RabbitMQ** Deployment + Service（端口 5672 + 15672，512Mi 内存限制）

使用合理的资源限制，与 `docker-compose.yml` 保持一致。healthcheck 用 `redis-cli ping` / `mysqladmin ping` / `rabbitmq-diagnostics ping`。

#### 3. 补全服务发现环境变量

在 ConfigMap 中添加所有必需的环境变量（参照 `ServiceConfig` 的字段和 `docker-compose.yml` 的环境变量）：

```yaml
# 服务发现
MODEL_SERVICE_HOST: "model-service"
MODEL_SERVICE_PORT: "8000"
CORE_API_HOST: "api-service"
CORE_API_PORT: "8000"
MULTIMEDIA_SERVICE_HOST: "multimedia-service"
MULTIMEDIA_SERVICE_PORT: "8000"
SEARCH_SERVICE_HOST: "search-service"
SEARCH_SERVICE_PORT: "8000"
API_GATEWAY_HOST: "0.0.0.0"
API_GATEWAY_PORT: "8080"

# 基础设施
REDIS_HOST: "redis"
REDIS_PORT: "6379"
USE_REDIS: "true"

# 数据库
DATABASE_MODE: "mysql"
MYSQL_HOST: "mysql"
MYSQL_PORT: "3306"

# 消息队列
RABBITMQ_HOST: "rabbitmq"
RABBITMQ_PORT: "5672"

# 缓存
CACHE_ENABLED: "true"
HF_CACHE_DIR: "/app/cache/huggingface"
KERAS_CACHE_DIR: "/app/cache/keras"

# 性能
OMP_NUM_THREADS: "1"      # 保留用于 ML 服务避免线程爆炸
MKL_NUM_THREADS: "1"
KMP_DUPLICATE_LIB_OK: "TRUE"
MKL_THREADING_LAYER: "GNU"
```

在 Secret 中补全 JWT、数据库密码、RabbitMQ 密码等敏感信息。

#### 4. 模型文件 Volume 挂载

模型目录 271MB，不适合打入镜像。

**方案**：使用 PVC + initContainer 从 OSS/S3 下载模型，或使用 hostPath 用于单节点。

推荐方案：创建 PVC `model-data`（10Gi），为 `model-service` 和 `inference-worker` 添加 initContainer 从对象存储（如 OSS）下载模型文件。Dockerfile 中新增一个 model-sync 脚本。

短期简单方案：使用 hostPath（适合单节点测试），或者将模型 COPY 到镜像但单独做一个只含模型的基础镜像层用于开发。

最终选择：PVC（ReadOnlyMany 取决于存储类，否则 ReadWriteOnce + 每个节点一份）。实际更通用的方案是用 initContainer 下载，或者用 hostPath + 外部脚本同步。

#### 5. 镜像仓库地址 + imagePullSecrets

在 `k8s-deploy.yaml` 中添加默认的 `imagePullSecrets` 引用。提供清晰的文档注释说明如何替换 `harbor.example.com` 为真实地址。

### P1 — 严重问题（能启动但出故障）

#### 6. 修复 Ingress rewrite-target

当前所有路径都 rewrite 到 `/`，这会导致 `/api/classify` → `/classify`，后端路由匹配不上。

修复方案：移除 `rewrite-target` annotation，使用 nginx ingress 默认的路径转发（保留完整路径）：
```yaml
annotations:
  nginx.ingress.kubernetes.io/rewrite-target: /$2  # 不需要 rewrite
```

或者完全去掉 rewrite-target，因为 nginx ingress 默认就会把 /api/xxx 原样转发到后端。

**正确做法**：nginx-frontend 容器的 `nginx.conf` 已经配置了 `/api/` → `api-gateway:8000/api/` 的代理，所以在 K8s Ingress 层面不需要做 rewrite，透传即可。

#### 7. 修复 health check 超时

`model-service` 需要加载多个大型模型（EfficientNet + CLIP + YOLO + WD Tagger），`initialDelaySeconds: 60` 不够。

改为 `initialDelaySeconds: 120`，配合 `failureThreshold: 5` 给予充足启动时间。

同时 `inference-worker` 添加 livenessProbe（通过检查进程存活）和 readinessProbe。

#### 8. 修复 search-worker Dockerfile 引用 .venv

`src/run/sh/start_search_worker.sh` 硬编码了 `.venv/bin/python3`。在 Docker 容器中 Python 直接可用。

修复：修改 `start_search_worker.sh`，优先使用系统 python3，fallback 到 .venv。同时 Dockerfile CMD 直接调用 python 而不是 shell 脚本更合理。

#### 9. 修复 nginx.conf 端口

nginx.conf 中 `proxy_pass http://api-gateway:8000/api/` — 在 K8s 中，Service 对外暴露 port 80（映射到容器的 8000），但 nginx 容器直接用 Service DNS 名，K8s Service 的 `port: 80` 会转发到容器的 `targetPort: 8000`。两种写法都可以，但直接用 Service port 80 更标准：
```
proxy_pass http://api-gateway/api/;
```

### P2 — 中等问题（影响稳定性）

#### 10. 持久化存储

为以下目录创建 PVC：
- `model-data`：模型文件（10Gi）
- `log-data`：日志文件（5Gi）
- `cache-data`：HF/Keras 缓存（5Gi）

model-service、api-service、inference-worker 挂载对应 PVC。

#### 11. 反亲和性 + PDB

- 为 `api-service`、`api-gateway`（2 副本）添加 `podAntiAffinity` (preferredDuringScheduling)
- 为 `model-service`（1 副本）添加 PDB `maxUnavailable: 0`（自愿中断时保护）
- 为 `api-service` 添加 PDB `minAvailable: 1`

#### 12. GPU 支持

在 ConfigMap 添加 `MODEL_DEVICE=cuda`。为 model-service 和 inference-worker 添加 GPU 资源请求注释（非必需，默认 CPU 模式）。

#### 13. worker 服务健康检查

search-worker、log-monitor、resource-monitor 等服务缺少 livenessProbe。worker 类型服务用 exec 方式检查进程存活：
```yaml
livenessProbe:
  exec:
    command: ["pgrep", "-f", "search_worker.py"]
  initialDelaySeconds: 30
  periodSeconds: 10
```

### P3 — 次要问题

#### 14. 升级监控版本
- Prometheus: v2.30.0 → v2.52.0
- Grafana: 8.2.0 → 10.4.0

#### 15. Secret 安全提示
在 YAML 中添加注释说明生产环境应使用 Sealed Secrets / External Secrets Operator。

---

## 实现步骤（按顺序）

### Step 1：清理旧配置文件
- 删除 `backend-deployment.yaml`、`frontend-deployment.yaml`、`services.yaml`、`hpa.yaml`、`ingress.yaml`
- 保留 `nginx.conf`、`grafana_dashboard.json`、所有 Dockerfile

### Step 2：重写 k8s-deploy.yaml → 拆分为多个文件
- `k8s-deploy.yaml`：Namespace + ConfigMap（完整环境变量） + Secret
- `k8s-services.yaml`：所有 Service（含基础设施）
- `k8s-deployments.yaml`：所有 Deployment（含基础设施：Redis/MySQL/RabbitMQ）
- `k8s-ingress.yaml`：修复后的 Ingress
- `k8s-hpa.yaml`：新的 HPA（指向正确 Deployment）
- `k8s-pdb.yaml`：PodDisruptionBudget
- `k8s-volumes.yaml`：PVC 定义

### Step 3：修复 Dockerfile 问题
- `Dockerfile.search-worker`：CMD 改为直接调用 python
- `start_search_worker.sh`：优先使用系统 python3

### Step 4：修复 nginx.conf
- 端口改为 K8s Service port 80（可选，当前写法也可用）

### Step 5：修复 build 脚本
- `build_k8s_images.sh`：保持不变（已覆盖所有服务）
- `build_from_source.sh`：补充缺失服务，或添加注释指向正确脚本

### Step 6：更新 deploy_monitoring.sh
- 升级 Prometheus 和 Grafana 版本
- 添加工厂化注释

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `deployment/backend-deployment.yaml` | 删除 | 旧配置 |
| `deployment/frontend-deployment.yaml` | 删除 | 旧配置 |
| `deployment/services.yaml` | 删除 | 旧配置 |
| `deployment/hpa.yaml` | 删除 | 旧配置，HPA 目标不匹配 |
| `deployment/ingress.yaml` | 删除 | 旧配置 |
| `deployment/k8s-deploy.yaml` | **重写** | 仅保留 Namespace + ConfigMap + Secret |
| `deployment/k8s-services.yaml` | **新建** | 所有 Service（含基础设施） |
| `deployment/k8s-deployments.yaml` | **新建** | 所有 Deployment（含基础设施） |
| `deployment/k8s-ingress.yaml` | **新建** | 修复 rewrite-target 的 Ingress |
| `deployment/k8s-hpa.yaml` | **新建** | 指向正确 Deployment 的 HPA |
| `deployment/k8s-pdb.yaml` | **新建** | PodDisruptionBudget |
| `deployment/k8s-volumes.yaml` | **新建** | PVC 定义 |
| `deployment/nginx.conf` | 修改 | 端口对齐 |
| `deployment/Dockerfile.search-worker` | 修改 | CMD 改用 python3 |
| `src/run/sh/start_search_worker.sh` | 修改 | 不硬编码 .venv |
| `scripts/k8s/build_from_source.sh` | 修改 | 补充缺失服务 |
| `deployment/deploy_monitoring.sh` | 不存在，应改 scripts/... | 升级版本 |
| `scripts/monitoring/deploy_monitoring.sh` | 修改 | 升级 Prometheus/Grafana 版本 |