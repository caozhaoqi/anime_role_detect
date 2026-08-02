# 项目优化计划

> 版本：v1.0  
> 日期：2026-08-02  
> 基于：架构分析报告 + 近期 K8s 部署问题 + 代码审查

---

## 一、P0 — 立即修复（本周）

### 1.1 硬编码路径清零

**现状**：`supervisord.conf` 中 3 处写死开发机器绝对路径：

```ini
# 第 12 行 — 硬编码
command=/Users/caozhaoqi/PycharmProjects/anime_role_detect/.venv/bin/python3 ...

# 第 29 行 — 硬编码
command=/Users/caozhaoqi/PycharmProjects/anime_role_detect/.venv/bin/python3 ...

# 第 46 行 — 硬编码
command=/Users/caozhaoqi/PycharmProjects/anime_role_detect/.venv/bin/python3 ...
```

**方案**：统一改为 `%(here)s` 变量（与其他 8 个 program 一致）

| 文件 | 改动 |
|------|------|
| `supervisord.conf` | L12: `command=%(here)s/.venv/bin/python3 %(here)s/src/services/model_service/app.py ...` |
| `supervisord.conf` | L29: `command=%(here)s/.venv/bin/python3 %(here)s/src/api/app.py` |
| `supervisord.conf` | L46: `command=%(here)s/.venv/bin/python3 %(here)s/src/services/api_gateway/app.py` |

**验收**：`supervisorctl reread && supervisorctl update` 后所有服务正常启动。

---

### 1.2 K8s 端口一致性校验

**现状**：近期发现 4 处端口映射错误（multimedia-service 8002→8000、search-service 8003→8000 等），手动排查不可靠。

**方案**：新增 CI 校验脚本 `scripts/ci/check_k8s_ports.sh`

```bash
#!/bin/bash
# 对照 docker-compose.yml 的端口定义，校验 K8s 配置中的 targetPort

set -e

echo "=== Checking K8s port consistency ==="

# 从 docker-compose.yml 提取服务端口映射
# 格式: service_name:container_port
declare -A EXPECTED_PORTS=(
  ["api-service"]="8000"
  ["model-service"]="8000"
  ["api-gateway"]="8080"
  ["multimedia-service"]="8002"
  ["search-service"]="8003"
  ["frontend"]="3000"
)

for svc in "${!EXPECTED_PORTS[@]}"; do
  expected="${EXPECTED_PORTS[$svc]}"
  actual=$(grep -A1 "name: $svc" deployment/k8s-services.yaml | grep targetPort | awk '{print $2}')
  if [ "$actual" != "$expected" ]; then
    echo "ERROR: $svc targetPort=$actual, expected=$expected"
    exit 1
  fi
  echo "OK: $svc -> $expected"
done

echo "All port checks passed."
```

添加到 `.github/workflows/docker-image.yml` 的 lint 步骤中。

**验收**：CI 通过，端口不一致时 CI 失败。

---

### 1.3 统一数据采集入口

**现状**：三套数据采集代码并存

```
src/data/collection/          ← 新实现（保留）
  ├── general/
  │   ├── data_collection.py
  │   └── keyword_based_collector.py
  └── spider/
      └── fetch_characters.py

src/data_collection/           ← 旧实现（废弃）
  └── keyword_based_collector.py  ← 与上面同名但实现不同

src/data_pipeline/             ← 清洗流水线（保留）
  ├── cleaners/
  └── pipeline.py
```

**方案**：

| 步骤 | 操作 |
|------|------|
| 1 | 确认 `src/data_collection/` 的无外部引用（grep 全项目） |
| 2 | 将 `src/data_collection/` 中仍有用的逻辑迁移到 `src/data/collection/` |
| 3 | 删除 `src/data_collection/` 目录 |
| 4 | 全项目 `sys.path.insert` 或 import 引用的迁移 |

**验收**：`grep -r "data_collection" src/` 无残留引用，CI 通过。

---

## 二、P1 — 两周内完成

### 2.1 API 版本化

**现状**：所有路由无版本前缀，例如 `/classify`、`/models`、`/auth/refresh`。

**方案**：

```
src/api/
├── app.py
├── routes/
│   ├── __init__.py
│   ├── v1/                          ← 新增
│   │   ├── __init__.py
│   │   ├── classify.py              ← 从 routes/classify.py 迁移
│   │   ├── auth.py
│   │   ├── collector.py
│   │   ├── cleaning.py
│   │   ├── tracing.py
│   │   └── ...
│   └── classify.py                  ← 保留为兼容路由，重定向到 /v1
```

`app.py` 改动：

```python
from src.api.routes.v1 import router as v1_router
from src.api.routes import router as legacy_router

app.include_router(v1_router, prefix="/api/v1")
app.include_router(legacy_router, prefix="/api")  # 兼容期 2 个版本
```

前端 `apiClient` 改动（`src/frontend/app/api/client.ts`）：

```typescript
// 第 22 行
baseURL: options.baseURL || '/api/v1',  // 原 '/api'
```

**验收**：`/api/v1/classify` 可正常调用，`/api/classify` 仍可用（兼容期）。

---

### 2.2 前端状态管理

**现状**：`src/frontend/app/hooks/` 仅 `useAuth.ts` 和 `useRecognition.ts`，无全局状态管理，无请求缓存。

**方案**：

```
src/frontend/app/
├── stores/                          ← 新增
│   ├── useAuthStore.ts              ← Zustand，替代 useAuth
│   └── useRecognitionStore.ts       ← Zustand，替代 useRecognition
├── hooks/
│   ├── useAuth.ts                   ← 标记 @deprecated，保留兼容
│   └── useRecognition.ts            ← 标记 @deprecated，保留兼容
├── api/
│   ├── client.ts                    ← 保留
│   ├── queryClient.ts               ← 新增：TanStack Query 配置
│   └── services/
│       ├── RecognitionService.ts    ← 保留，新增 hooks 包装
│       ├── AuthService.ts
│       └── HistoryService.ts
```

`stores/useAuthStore.ts` 示例：

```typescript
import { create } from 'zustand';

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (token: string) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: localStorage.getItem('accessToken'),
  isAuthenticated: !!localStorage.getItem('accessToken'),
  login: (token) => {
    localStorage.setItem('accessToken', token);
    set({ token, isAuthenticated: true });
  },
  logout: () => {
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');
    set({ token: null, user: null, isAuthenticated: false });
  },
}));
```

`api/queryClient.ts`：

```typescript
import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,   // 5 分钟
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});
```

**依赖安装**：

```bash
cd src/frontend && npm install zustand @tanstack/react-query
```

**验收**：图片分类结果缓存 5 分钟内不重复请求，刷新页面后登录状态保持。

---

### 2.3 业务健康检查

**现状**：各服务健康检查仅 `curl /ready` 返回 200，不检查依赖是否就绪。

**方案**：每个服务新增 `/health` 端点，返回结构化状态：

```json
{
  "status": "healthy",       // healthy | degraded | unhealthy
  "version": "2.3.0",
  "checks": {
    "database": {"status": "ok", "latency_ms": 2},
    "redis": {"status": "ok", "latency_ms": 1},
    "rabbitmq": {"status": "ok"},
    "model_loaded": {"status": "ok", "models": ["ViT-B/32", "efficientnet_b3"]}
  }
}
```

| 服务 | 检查项 |
|------|--------|
| api-service | DB、Redis、Model Service 可达 |
| model-service | 模型文件存在、模型加载成功 |
| search-service | Redis、RabbitMQ、FAISS 索引 |
| multimedia-service | Redis、RabbitMQ、OSS 可达 |
| inference-worker | 模型加载、Redis 连接 |
| search-worker | Redis、RabbitMQ、FAISS 索引 |

`supervisord.conf` 中 healthcheck 改为 `curl /health`。

**验收**：K8s 就绪探针返回真实状态，`kubectl describe pod` 能看到健康检查结果。

---

### 2.4 API Gateway 增强

**现状**：`src/services/api_gateway/app.py` 仅做文档聚合，无网关核心能力。

**方案**：

```python
# src/services/api_gateway/middleware/rate_limit.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# src/services/api_gateway/middleware/circuit_breaker.py
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def call_backend(url: str, **kwargs):
    ...
```

| 功能 | 实现 |
|------|------|
| 限流 | slowapi，默认 100 req/min，按 IP |
| 熔断 | tenacity，3 次失败后断路 30 秒 |
| 统一鉴权 | JWT 验证中间件，401 统一返回 |
| 请求日志 | 中间件记录 method/path/status/latency |

**验收**：同一 IP 超过 100 req/min 返回 429，后端不可用时返回 503 而非超时。

---

### 2.5 日志结构化

**现状**：Loguru 输出文本格式，ES 无法高效检索。

**方案**：`src/config/config.py` 中统一配置：

```python
from loguru import logger
import sys
import json

logger.remove()
logger.add(
    sys.stdout,
    format=lambda record: json.dumps({
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "service": "api-service",
        "trace_id": record["extra"].get("trace_id", ""),
        "message": record["message"],
        "module": record["name"],
        "line": record["line"],
    }),
    level="INFO",
)
```

**验收**：`docker logs` 输出 JSON 行，Kibana 可按 `service`、`level`、`trace_id` 过滤。

---

## 三、P2 — 一个月内完成

### 3.1 拆分大函数

**现状**：`src/api/app.py` 中 `process_single_image` 超过 200 行，职责混杂。

**方案**：拆分为独立模块

```
src/api/
├── processors/
│   ├── __init__.py
│   ├── image_validator.py       # validate_image(file) -> bool
│   ├── feature_extractor.py     # extract_features(image) -> vector
│   ├── tag_generator.py         # generate_tags(image) -> list[Tag]
│   ├── keypoint_detector.py     # detect_keypoints(image) -> list[Keypoint]
│   └── image_classifier.py      # classify(vector, tags) -> ClassificationResult
```

**验收**：每个模块函数不超过 50 行，有独立单元测试。

---

### 3.2 分类器缓存优化

**现状**：`src/api/app.py` 中分类器缓存使用完整路径做键，命中率低。

**方案**：

```python
# src/api/cache/classifier_cache.py
from functools import lru_cache
import time

class ClassifierCache:
    def __init__(self, max_size: int = 3, ttl: int = 600):
        self._cache: dict[str, tuple[object, float]] = {}  # model_name -> (instance, expire_at)
        self.max_size = max_size
        self.ttl = ttl

    def get(self, model_name: str):
        if model_name in self._cache:
            instance, expire_at = self._cache[model_name]
            if time.time() < expire_at:
                return instance
            del self._cache[model_name]
        return None

    def set(self, model_name: str, instance):
        if len(self._cache) >= self.max_size:
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        self._cache[model_name] = (instance, time.time() + self.ttl)
```

**验收**：同一模型名请求不重复初始化，内存占用稳定。

---

### 3.3 清理遗留代码

**待清理项**：

| 路径 | 大小 | 处理方式 |
|------|------|---------|
| `archived/broken_modules/` | ~1MB | 删除（已确认无引用） |
| `scripts/skillhub/` | 88MB | 移出仓库或删除（无外部引用） |
| `src/run/start_all.py` | — | 标记 `@deprecated`，保留为开发工具 |
| `src/run/start_all_stable.py` | — | 同上 |
| `src/run/start_core.py` | — | 同上 |
| `src/application.py` | — | 同上 |

**验收**：仓库体积减少 ~90MB，`grep -r` 无残留引用。

---

### 3.4 依赖升级

**现状**：torch 2.1.0（2023 Q4）、transformers 4.34.0（2023 Q3）。

**升级目标**：

| 依赖 | 当前版本 | 目标版本 | 收益 |
|------|---------|---------|------|
| torch | 2.1.0 | 2.4.0 | Flash Attention 2、torch.compile |
| torchvision | 0.16.0 | 0.19.0 | 配套升级 |
| transformers | 4.34.0 | 4.44.0 | 更快分词器、模型支持 |
| sentence-transformers | 2.2.2 | 3.0.0 | 性能提升 |
| fastapi | 0.115.0 | 0.115.6 | 安全补丁 |
| onnxruntime | 1.18.0 | 1.19.0 | 性能提升 |

**注意**：升级后在 development 环境验证 3 天再合并。

**验收**：`pytest` 全部通过，模型推理结果一致。

---

### 3.5 K8s 原生化（废弃 Supervisor）

**现状**：11 个 program 由 Supervisor 统一管理，K8s 下是反模式。

**方案**：每个服务独立 Deployment

```
deployment/
├── k8s-deployments.yaml          ← 已有，每个服务独立 Deployment
├── k8s-services.yaml             ← 已有
├── k8s-volumes.yaml              ← 已有
├── k8s-logging.yaml              ← 已有
└── k8s-ingress.yaml              ← 新增
```

Dockerfile 改动：ENTRYPOINT 直接启动服务，不再用 supervisord：

```dockerfile
# Dockerfile.api-service
CMD ["python", "-m", "src.api.app"]
```

移除 `supervisord.conf` 文件，Docker镜像中不再安装 supervisor。

**验收**：K8s 下每个服务独立 Pod，`kubectl get pods` 显示 8+ 个服务 Pod，HPA 可配置。

---

## 四、P3 — 三个月内完成

### 4.1 GPU 资源调度

- 引入 K8s Device Plugin 管理 GPU
- Volcano 调度器处理推理任务优先级
- 模型服务按 GPU 类型（T4/A10）做 nodeSelector

### 4.2 模型量化部署

- CLIP 模型 ONNX 量化（INT8）
- YOLOv8 模型 TensorRT 加速
- 推理延迟目标：< 100ms（当前 ~300ms）

### 4.3 动态批处理推理

- 推理请求合并窗口（50ms）
- 批量推理 GPU 利用率提升 3-5x

### 4.4 前后端分离部署

- 前端独立部署到 OSS + CDN
- `src/frontend/` 保留开发源码，构建产物不再打包进 Docker 镜像

### 4.5 CI/CD 门禁完善

- 测试覆盖率阈值 80%
- E2E 测试（Playwright）
- 金丝雀发布（10% 流量 → 50% → 100%）
- 自动回滚（错误率 > 5% 触发）

### 4.6 可观测性看板

- Grafana 统一 Dashboard
- RED 指标（Rate / Error / Duration）
- 模型推理延迟 P50/P95/P99

---

## 五、执行路线图

```
Week 1-2 (P0)
├── 硬编码路径清零
├── K8s 端口校验 CI
└── 统一数据采集入口

Week 3-4 (P1)
├── API 版本化
├── 前端状态管理
├── 业务健康检查
├── API Gateway 增强
└── 日志结构化

Week 5-8 (P2)
├── 拆分大函数
├── 分类器缓存优化
├── 清理遗留代码
├── 依赖升级
└── K8s 原生化

Week 9-12 (P3)
├── GPU 资源调度
├── 模型量化部署
├── 动态批处理推理
├── 前后端分离部署
├── CI/CD 门禁完善
└── 可观测性看板
```