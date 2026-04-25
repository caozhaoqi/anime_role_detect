# anime_role_detect 项目优化计划

## 一、项目现状分析

### 1.1 当前架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│  API Gateway │────▶│ Backend API │
│  (Next.js)  │     │   (8000)    │     │   (8001)   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                                           ┌────▼────┐
                                           │Model Svc │
                                           │  (8888)  │
                                           └─────────┘
```

### 1.2 存在的问题

| 问题 | 描述 | 影响 |
|------|------|------|
| 同步阻塞 | 图片分类同步处理，大图/批量请求阻塞 | 响应时间长，用户体验差 |
| 无缓存层 | 相同图片重复计算 | 资源浪费，延迟高 |
| 数据无持久化 | 识别记录存储在内存/SQLite | 数据丢失，无分析能力 |
| 监控缺失 | 无关键指标监控 | 问题难定位 |
| 临时文件问题 | temp 目录需手动创建 | 部署繁琐 |

---

## 二、优化计划

### 阶段一：基础设施完善（1-2周）

#### 1.1 引入 Redis 缓存服务

**目标**：减少重复计算，提升响应速度

**实现方案**：
```python
# src/services/cache_service/redis_cache.py
import redis
from functools import wraps

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db)

    def get_image_result(self, image_hash: str) -> dict:
        """获取缓存的识别结果"""
        key = f"classify:result:{image_hash}"
        result = self.client.get(key)
        return json.loads(result) if result else None

    def set_image_result(self, image_hash: str, result: dict, ttl=3600):
        """缓存识别结果，默认1小时过期"""
        key = f"classify:result:{image_hash}"
        self.client.setex(key, ttl, json.dumps(result))
```

**改造点**：
- 新增 `src/services/cache_service/redis_cache.py`
- 修改 `src/api/app.py` 分类接口，增加缓存查询
- 配置项添加到 `src/core/config/service_config.py`

#### 1.2 完善 temp 目录自动创建

**目标**：简化部署流程

**实现方案**：
```python
# src/core/utils/utils.py 或新建 src/core/utils/temp_manager.py
import os
from pathlib import Path

def ensure_temp_dir():
    """确保 temp 目录存在"""
    temp_dir = Path(project_root) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir
```

**改造点**：
- 在 `src/api/app.py` 启动时调用
- 在 `src/services/processor/model_processor.py` 启动时调用
- 移除手动 mkdir 步骤

#### 1.3 引入结构化日志

**目标**：统一日志格式，便于排查

**实现方案**：
```python
# src/core/logging/formatters.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)
```

---

### 阶段二：任务队列引入（2-3周）

#### 2.1 引入 Celery 任务队列

**目标**：异步处理图片分类，支持高并发

**实现方案**：
```python
# src/tasks/celery_app.py
from celery import Celery
from celery.signals import worker_ready

celery_app = Celery(
    'anime_role_detect',
    broker='redis://localhost:6379/1',
    backend='redis://localhost:6379/2'
)

# 任务定义
@celery_app.task(bind=True, max_retries=3)
def classify_image_task(self, image_path: str, model_name: str, options: dict):
    """异步图片分类任务"""
    try:
        # 调用现有的分类逻辑
        result = classify_image(image_path, model_name, **options)
        return {"status": "success", "result": result}
    except Exception as e:
        self.retry(exc=e, countdown=60)

@celery_app.task
def batch_classify_task(image_paths: list, model_name: str):
    """批量分类任务"""
    results = []
    for path in image_paths:
        result = classify_image_task.delay(path, model_name)
        results.append(result)
    return results
```

**API 改造**：
```python
# src/api/app.py
@app.post("/api/classify/async")
async def classify_image_async(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """异步分类接口"""
    # 保存文件
    temp_path = save_temp_file(file)

    # 直接返回任务ID
    task = classify_image_task.delay(temp_path, model_name)

    return {
        "task_id": task.id,
        "status": "pending",
        "message": "任务已提交，请通过 /api/task/{task_id} 查询结果"
    }

@app.get("/api/task/{task_id}")
async def get_task_result(task_id: str):
    """获取异步任务结果"""
    task = celery_app.AsyncResult(task_id)
    if task.ready():
        return {"status": "completed", "result": task.result}
    else:
        return {"status": "pending", "progress": task.info}
```

#### 2.2 任务状态监控接口

**目标**：跟踪异步任务状态

```python
@app.get("/api/tasks")
async def list_tasks(status: str = None):
    """列出任务"""
    # 使用 Flower 或自定义接口
    pass
```

---

### 阶段三：数据库持久化（2-3周）

#### 3.1 引入 SQLAlchemy ORM

**目标**：结构化存储识别记录

**数据模型**：
```python
# src/models/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class RecognitionRecord(Base):
    __tablename__ = 'recognition_records'

    id = Column(String, primary_key=True)
    image_filename = Column(String)
    image_hash = Column(String, index=True)
    model_used = Column(String)
    processing_time = Column(Float)
    is_multi_role = Column(Boolean)
    nsfw_status = Column(Boolean)
    detected_text = Column(Boolean)
    recognition_result = Column(JSON)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    user_id = Column(Integer, ForeignKey('users.id'))

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    role = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    records = relationship("RecognitionRecord", back_populates="user")
```

#### 3.2 历史记录 API 增强

```python
@app.get("/api/history")
async def get_history(
    skip: int = 0,
    limit: int = 20,
    model: str = None,
    start_date: datetime = None,
    end_date: datetime = None
):
    """获取识别历史（支持分页和筛选）"""
    query = session.query(RecognitionRecord)

    if model:
        query = query.filter(RecognitionRecord.model_used == model)
    if start_date:
        query = query.filter(RecognitionRecord.created_at >= start_date)
    if end_date:
        query = query.filter(RecognitionRecord.created_at <= end_date)

    total = query.count()
    records = query.order_by(desc(RecognitionRecord.created_at)).offset(skip).limit(limit).all()

    return {"total": total, "records": records}

@app.get("/api/statistics")
async def get_statistics():
    """获取统计数据"""
    stats = {
        "total_recognitions": session.query(RecognitionRecord).count(),
        "models_usage": {},  # 各模型使用次数
        "daily_recognitions": {},  # 每日识别量
        "avg_processing_time": {},  # 各模型平均处理时间
    }
    return stats
```

---

### 阶段四：监控体系完善（1-2周）

#### 4.1 引入 Prometheus 指标

**目标**：量化系统性能

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# 请求指标
REQUEST_COUNT = Counter(
    'anime_detect_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'anime_detect_request_latency_seconds',
    'Request latency',
    ['method', 'endpoint']
)

# 模型指标
MODEL_INFERENCE_TIME = Histogram(
    'anime_detect_model_inference_seconds',
    'Model inference time',
    ['model_name']
)

CACHE_HIT_RATE = Gauge(
    'anime_detect_cache_hit_rate',
    'Cache hit rate'
)

# 在 API 中使用
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    return response
```

#### 4.2 健康检查增强

```python
@app.get("/api/health/detailed")
async def detailed_health_check():
    """详细健康检查"""
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api_gateway": {"status": "up"},
            "backend_api": {"status": "up"},
            "model_service": {"status": "up"},
            "redis": {"status": "up" if redis_client.ping() else "down"},
            "database": {"status": "up"}
        },
        "metrics": {
            "requests_per_minute": get_request_rate(),
            "avg_response_time": get_avg_response_time(),
            "cache_hit_rate": get_cache_hit_rate()
        }
    }
    return health
```

---

### 阶段五：安全与认证增强（1-2周）

#### 5.1 JWT 令牌刷新机制

```python
# src/services/auth_service.py
class AuthService:
    def refresh_token(self, refresh_token: str) -> dict:
        """刷新访问令牌"""
        payload = decode_token(refresh_token)

        if payload.get("type") != "refresh":
            raise HTTPException(401, "Invalid refresh token")

        # 生成新的访问令牌
        new_access_token = create_access_token(payload["sub"])

        return {
            "access_token": new_access_token,
            "expires_in": 3600  # 1小时
        }
```

#### 5.2 API 限流

```python
# src/middleware/rate_limit.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/classify")
@limiter.limit("10/minute")  # 每分钟10次
async def classify_image(request: Request):
    pass
```

---

## 三、时间规划

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        优化计划时间线                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ 第1-2周   │ 阶段一：基础设施完善                                       │
│           │ ├── 引入 Redis 缓存                                        │
│           │ ├── 完善 temp 目录自动创建                                  │
│           │ └── 结构化日志                                             │
│           │                                                            │
│ 第3-5周   │ 阶段二：任务队列引入                                       │
│           │ ├── 引入 Celery + Redis                                    │
│           │ ├── 异步分类接口                                            │
│           │ └── 任务状态监控                                            │
│           │                                                            │
│ 第6-8周   │ 阶段三：数据库持久化                                       │
│           │ ├── 引入 SQLAlchemy ORM                                     │
│           │ ├── 数据模型设计                                            │
│           │ └── 历史记录 API 增强                                       │
│           │                                                            │
│ 第9-10周  │ 阶段四：监控体系完善                                       │
│           │ ├── Prometheus 指标                                         │
│           │ └── 详细健康检查                                            │
│           │                                                            │
│ 第11-12周 │ 阶段五：安全与认证增强                                     │
│           │ ├── JWT 令牌刷新                                            │
│           │ └── API 限流                                               │
│           │                                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 四、里程碑

| 阶段 | 里程碑 | 验收标准 |
|------|--------|----------|
| 阶段一 | MVP 完成 | 缓存命中率 > 50%，日志统一格式 |
| 阶段二 | 异步处理上线 | 支持 100+ 并发，任务不丢失 |
| 阶段三 | 数据持久化 | 历史记录可查询，统计数据准确 |
| 阶段四 | 监控可用 | Grafana 面板展示，告警正常 |
| 阶段五 | 安全合规 | 令牌刷新正常，限流生效 |

---

## 五、技术债务清理

| 问题 | 解决方案 | 优先级 |
|------|----------|--------|
| 多余的 `proxies=None` 参数 | 移除（httpx 版本不支持） | 🔴 高 |
| temp 目录硬编码 | 改为配置项 | 🔴 高 |
| 前端 `cache_bypass` 传时间戳 | 修复为布尔值 | 🔴 高 |
| 重复的认证头处理逻辑 | 抽取为中间件 | 🟡 中 |
| 模型服务无健康检查端点 | 补充 `/api/health` | 🟡 中 |
| 缺少单元测试 | 补充基础测试 | 🟢 低 |

---

## 六、风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| Redis 连接失败 | 缓存不可用 | 降级到内存缓存 |
| Celery 任务丢失 | 分类失败 | 开启任务持久化，确认机制 |
| 数据库迁移复杂 | 数据丢失风险 | 先在测试环境验证，准备回滚脚本 |
| 监控数据量大 | 存储成本高 | 设置数据保留期限，定期清理 |

---

## 七、预期收益

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 平均响应时间 | 500ms | 50ms (缓存命中) | 10x |
| 并发支持 | 10 | 100+ | 10x |
| 重复计算 | 100% | <50% | 50% 节省 |
| 问题定位时间 | 30min | 5min | 6x |
| 数据可用性 | 无 | 90天+ | - |