# 【技术难点】Docker Compose 部署方案

> 在多服务架构中，手动逐个启动服务非常繁琐。本文介绍如何使用 Docker Compose 一键部署整个系统。

---

## 🔍 问题背景

系统包含多个独立服务：

| 服务 | 端口 | 依赖 |
|------|------|------|
| API Gateway | 8000 | 无 |
| Backend API | 8001 | Redis, PostgreSQL |
| Model Service | 8888 | Redis |
| Frontend | 3001 | API Gateway |
| Redis | 6379 | 无 |
| PostgreSQL | 5432 | 无 |

**核心挑战**：如何简化部署流程，确保各服务正确启动并相互通信？

---

## 💡 解决方案：Docker Compose

### docker-compose.yaml

```yaml
version: '3.8'

services:
  # Redis 缓存服务
  redis:
    image: redis:7-alpine
    container_name: anime_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL 数据库服务
  postgres:
    image: postgres:15-alpine
    container_name: anime_postgres
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: anime_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_password_here
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Model Service
  model_service:
    build:
      context: .
      dockerfile: Dockerfile.model
    container_name: anime_model_service
    ports:
      - "8888:8888"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - MODEL_DIR=/models
    volumes:
      - ./models:/models
    depends_on:
      redis:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Backend API
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: anime_backend
    ports:
      - "8001:8001"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_NAME=anime_db
      - DB_USER=postgres
      - DB_PASSWORD=your_password_here
      - MODEL_SERVICE_URL=http://model_service:8888
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy

  # API Gateway
  gateway:
    build:
      context: .
      dockerfile: Dockerfile.gateway
    container_name: anime_gateway
    ports:
      - "8000:8000"
    environment:
      - BACKEND_URL=http://backend:8001
      - MODEL_SERVICE_URL=http://model_service:8888
    depends_on:
      backend:
        condition: service_started

  # Frontend
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    container_name: anime_frontend
    ports:
      - "3001:3001"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      gateway:
        condition: service_started

volumes:
  redis_data:
  postgres_data:
```

### Dockerfile 示例

#### Dockerfile.model（模型服务）

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 安装依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制代码
COPY src/services/model_service /app/src/services/model_service
COPY src/core /app/src/core
COPY src/utils /app/src/utils

# 设置环境变量
ENV PYTHONPATH=/app

# 启动服务
CMD ["uvicorn", "src.services.model_service.app:app", "--host", "0.0.0.0", "--port", "8888"]
```

#### Dockerfile.backend（后端服务）

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制代码
COPY src/api /app/src/api
COPY src/core /app/src/core
COPY src/utils /app/src/utils

# 设置环境变量
ENV PYTHONPATH=/app

# 启动服务
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8001"]
```

#### Dockerfile.gateway（API网关）

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制代码
COPY src/services/api_gateway /app/src/services/api_gateway

# 设置环境变量
ENV PYTHONPATH=/app

# 启动服务
CMD ["uvicorn", "src.services.api_gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🚀 使用示例

### 启动服务

```bash
# 启动所有服务（后台运行）
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 停止服务并删除数据卷
docker-compose down -v
```

### 服务通信测试

```bash
# 测试 API Gateway
curl http://localhost:8000/api/health

# 测试 Backend API
curl http://localhost:8001/api/health

# 测试 Model Service
curl http://localhost:8888/api/model/health
```

### 环境变量配置

创建 `.env` 文件：

```env
# 数据库配置
DB_HOST=postgres
DB_PORT=5432
DB_NAME=anime_db
DB_USER=postgres
DB_PASSWORD=your_secure_password

# Redis 配置
REDIS_HOST=redis
REDIS_PORT=6379

# 服务地址（容器内部通信）
BACKEND_URL=http://backend:8001
MODEL_SERVICE_URL=http://model_service:8888

# 前端配置
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## ⚡ 部署架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Network                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Redis   │    │PostgreSQL│    │  Model   │              │
│  │  (6379)  │    │  (5432)  │    │ Service  │              │
│  │          │    │          │    │  (8888)  │              │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘              │
│       │               │               │                     │
│       │               │               │                     │
│       └──────┬────────┴───────┬───────┘                     │
│              │                │                             │
│       ┌──────▼──────┐         │                             │
│       │   Backend   │─────────┘                             │
│       │   (8001)    │                                       │
│       └──────┬──────┘                                       │
│              │                                              │
│       ┌──────▼──────┐                                       │
│       │   Gateway   │                                       │
│       │   (8000)    │                                       │
│       └──────┬──────┘                                       │
│              │                                              │
│       ┌──────▼──────┐                                       │
│       │  Frontend   │                                       │
│       │   (3001)    │                                       │
│       └─────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 关键要点

1. **服务编排**：使用 Docker Compose 定义所有服务及其依赖关系
2. **健康检查**：确保依赖服务启动后再启动当前服务
3. **环境变量**：通过 `.env` 文件管理配置，避免硬编码
4. **数据持久化**：使用 Docker 卷保存 Redis 和 PostgreSQL 数据
5. **GPU 支持**：模型服务配置 GPU 资源预留
6. **网络隔离**：容器间通过服务名通信，无需暴露端口到宿主机

---

## 📚 系列文章汇总

| 文章 | 主题 | 文件 |
|------|------|------|
| 第1篇 | 多模型集成与性能优化 | `01_multi_model_management.md` |
| 第2篇 | API Gateway 设计与实现 | `02_api_gateway.md` |
| 第3篇 | 分布式服务协调 | `03_distributed_coordination.md` |
| 第4篇 | 图像预处理与特征提取 | `04_image_preprocessing.md` |
| 第5篇 | NSFW 内容过滤 | `05_nsfw_detection.md` |
| 第6篇 | 爬虫反爬机制突破 | `06_anti_crawler.md` |
| 第7篇 | 数据持久化与缓存层 | `07_storage_and_cache.md` |
| 第8篇 | Docker Compose 部署 | `08_docker_deployment.md` |

---

*感谢阅读！如有问题欢迎留言讨论。*
