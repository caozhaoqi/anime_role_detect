# K8s 部署指南

> 本目录（`k8s/`）是项目**唯一的权威 K8s 部署源**。旧的 `deployment/k8s-*.yaml` 已归档至 `deployment/_legacy_backup/`，请勿再使用。

## 目录结构

```
k8s/
├── base/            # 生产/本地基础清单（kubectl apply -k k8s/base/）
├── overlays/
│   └── ci/          # kind 单节点 CI 测试 overlay（降配 + emptyDir + 去 GPU 亲和）
└── README.md
```

## 快速部署

```bash
# 生产 / 本地集群（命名空间 anime-role-detect）
kubectl apply -k k8s/base/

# CI 测试（kind 单节点，资源降配）
kubectl apply -k k8s/overlays/ci/
```

## 前置条件

1. 一个运行中的 K8s 集群 (1.24+)
2. `kubectl` 已配置
3. 集群中有 GPU 节点 (给 model-service 打标签：`kubectl label node <node> workload=ml-gpu`)
4. Ingress Controller 已安装 (nginx-ingress)
5. 镜像已推送到镜像仓库

## 快速部署

```bash
# 1. 替换 Secret 中的 CHANGE_ME 占位
kubectl create secret generic ard-secret \
  --namespace=anime-role-detect \
  --from-literal=MYSQL_ROOT_PASSWORD='your_root_pwd' \
  --from-literal=MYSQL_PASSWORD='your_anime_pwd' \
  --from-literal=RABBITMQ_USER='ard_admin' \
  --from-literal=RABBITMQ_PASSWORD='your_rabbit_pwd' \
  --from-literal=JWT_SECRET_KEY='your_jwt_secret' \
  --from-literal=INTERNAL_SERVICE_TOKEN='your_internal_token' \
  --from-literal=SUPERVISOR_ADMIN='your_admin' \
  --from-literal=SUPERVISOR_PWD='your_pwd' \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. 给 GPU 节点打标签
kubectl label node <gpu-node> workload=ml-gpu

# 3. 一键部署
kubectl apply -k k8s/

# 4. 查看部署状态
kubectl -n anime-role-detect get pods -w
```

## 资源清单

| 服务 | 副本 | 内存 limit | CPU limit | 说明 |
|------|------|-----------|-----------|------|
| redis | 1 | 256Mi | 200m | 缓存(db0)+队列(db1) |
| mysql | 1 (StatefulSet) | 1Gi | 500m | 三层降级保留 |
| rabbitmq | 1 | 512Mi | 250m | 异步 broker |
| api-gateway | 2 | 512Mi | 250m | 入口网关 |
| api-service | 2 | 1Gi | 500m | 核心 API |
| model-service | 1 | **4Gi** | **4** | GPU节点，防OOM |
| multimedia-service | 1 | 512Mi | 250m | 图片处理 |
| search-service | 1 | 512Mi | 250m | CLIP/FAISS |
| monitoring | 1 | 256Mi | 100m | 监控面板 |
| frontend | 2 | 512Mi | 250m | Next.js |
| search-worker | 1 | 1.5Gi | 500m | 索引构建 |
| inference-worker | 2 | 1.5Gi | 500m | 推理队列 |
| celery-worker | 1 | 512Mi | 250m | 通用异步 |
| celery-beat | 1 | 256Mi | 100m | 定时调度 |

## 探针映射

| 服务 | liveness | readiness |
|------|----------|-----------|
| 应用服务 | `/live` | `/ready` (model-aware) |
| Workers | `pgrep` exec | — |
| Redis | `redis-cli ping` | `redis-cli ping` |
| MySQL | `mysqladmin ping` | `mysqladmin ping` |
| RabbitMQ | TCP 5672 | TCP 5672 |

## 关键设计决策

1. **model-service 4Gi 内存 limit** — 参考 K8s 文档中 worker-2 OOM 162% 事故，EfficientNet-B3 (~795MB) + WD ViT + EasyOCR 懒加载峰值需足够 headroom
2. **MySQL StatefulSet + PVC** — 保留三层降级 (MySQL → SQLite → Memory)，PVC 确保持久化
3. **Redis 同实例不同 DB** — db0=cache / db1=queue (P0-2 已实现)
4. **推理队列优先级** — high/low 双队列 (P1 已实现)
5. **Prometheus 注解** — api-gateway/api-service/model-service 已加 scrape 注解

## 卸载

```bash
kubectl delete -k k8s/
```
