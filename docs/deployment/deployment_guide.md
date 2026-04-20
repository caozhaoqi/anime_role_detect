# 分层部署指南

## 1. 环境要求

### 1.1 系统要求

- **操作系统**：Linux (Ubuntu 18.04+)、macOS 10.14+
- **内存**：
  - 前端服务：2GB+ RAM
  - 后端API服务：4GB+ RAM
  - 模型服务：8GB+ RAM (推荐16GB+)
  - 数据服务：4GB+ RAM
- **CPU**：
  - 前端服务：2核+ CPU
  - 后端API服务：4核+ CPU
  - 模型服务：4核+ CPU (推荐8核+)
  - 数据服务：2核+ CPU
- **存储**：
  - 前端服务：1GB+ 磁盘空间
  - 后端API服务：5GB+ 磁盘空间
  - 模型服务：20GB+ 磁盘空间（用于存储模型文件）
  - 数据服务：50GB+ 磁盘空间（用于存储采集的数据）

### 1.2 软件要求

- **Python**：3.8+（后端服务和模型服务）
- **Node.js**：16+（前端服务）
- **pip**：20.0+（Python包管理）
- **npm**：8.0+（Node.js包管理）
- **Docker**（可选，用于容器化部署）
- **Git**：2.0+（代码版本控制）

## 2. 部署步骤

### 2.1 克隆代码仓库

```bash
git clone https://github.com/caozhaoqi/anime_role_detect.git
cd anime_role_detect
```

### 2.2 前端服务部署

#### 2.2.1 安装依赖

```bash
cd src/frontend
npm install
```

#### 2.2.2 配置环境变量

创建 `.env` 文件：

```env
# 后端API服务地址
NEXT_PUBLIC_API_URL=http://localhost:8000
```

#### 2.2.3 构建和部署

```bash
# 构建
npm run build

# 启动（开发模式）
npm run dev

# 启动（生产模式）
npm start
```

### 2.3 后端API服务部署

#### 2.3.1 安装依赖

```bash
cd src/backend
pip install -r requirements.txt
```

#### 2.3.2 配置环境变量

创建 `.env` 文件：

```env
# 模型服务地址
MODEL_SERVICE_URL=http://localhost:8001

# 是否使用模型服务
USE_MODEL_SERVICE=true

# 其他配置
MAX_MEMORY_USAGE=6000
```

#### 2.3.3 启动服务

```bash
cd api
python app.py
```

### 2.4 模型服务部署

#### 2.4.1 安装依赖

```bash
cd src/backend/services/model_service
pip install -r requirements.txt
```

#### 2.4.2 配置环境变量

创建 `.env` 文件：

```env
# 模型路径
MODEL_PATH=./models

# 是否启用GPU
GPU_ENABLED=true
```

#### 2.4.3 启动服务

```bash
python app.py
```

### 2.5 数据服务部署

#### 2.5.1 安装依赖

```bash
cd spider_image_system
pip install -r requirements.txt
```

#### 2.5.2 配置环境变量

创建 `.env` 文件：

```env
# 数据存储路径
DATA_PATH=./data

# 采集配置
SPIDER_THREADS=5
```

#### 2.5.3 启动服务

```bash
python main.py
```

## 3. 配置说明

### 3.1 前端服务配置

| 配置项 | 说明 | 默认值 |
|-------|------|-------|
| `NEXT_PUBLIC_API_URL` | 后端API服务地址 | `http://localhost:8000` |
| `NEXT_PUBLIC_APP_TITLE` | 应用标题 | `Anime Role Detector` |
| `NEXT_PUBLIC_APP_VERSION` | 应用版本 | `1.0.0` |

### 3.2 后端API服务配置

| 配置项 | 说明 | 默认值 |
|-------|------|-------|
| `MODEL_SERVICE_URL` | 模型服务地址 | `http://localhost:8001` |
| `USE_MODEL_SERVICE` | 是否使用模型服务 | `false` |
| `MAX_MEMORY_USAGE` | 最大内存使用阈值（MB） | `6000` |
| `CACHE_MAX_SIZE` | 缓存最大大小 | `200` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

### 3.3 模型服务配置

| 配置项 | 说明 | 默认值 |
|-------|------|-------|
| `MODEL_PATH` | 模型文件路径 | `./models` |
| `GPU_ENABLED` | 是否启用GPU | `true` |
| `BATCH_SIZE` | 批处理大小 | `1` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

### 3.4 数据服务配置

| 配置项 | 说明 | 默认值 |
|-------|------|-------|
| `DATA_PATH` | 数据存储路径 | `./data` |
| `SPIDER_THREADS` | 采集线程数 | `5` |
| `DOWNLOAD_TIMEOUT` | 下载超时时间（秒） | `30` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

## 4. 服务间通信

### 4.1 前端服务 ↔ 后端API服务

- **协议**：HTTP/HTTPS
- **端口**：8000
- **接口**：
  - `POST /api/classify`：分类图片
  - `GET /api/health`：健康检查

### 4.2 后端API服务 ↔ 模型服务

- **协议**：HTTP/HTTPS
- **端口**：8001
- **接口**：
  - `POST /api/model/predict`：模型预测
  - `POST /api/model/extract`：特征提取
  - `GET /api/health`：健康检查

### 4.3 后端API服务 ↔ 数据服务

- **协议**：HTTP/HTTPS
- **端口**：8002
- **接口**：
  - `POST /api/data/collect`：数据采集
  - `GET /api/data/health`：健康检查

## 5. 容器化部署

### 5.1 Docker Compose 配置

创建 `docker-compose.yml` 文件：

```yaml
version: '3.8'
services:
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - MODEL_SERVICE_URL=http://model:8001
      - USE_MODEL_SERVICE=true
    depends_on:
      - model

  model:
    build:
      context: .
      dockerfile: Dockerfile.model
    ports:
      - "8001:8001"
    environment:
      - GPU_ENABLED=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  data:
    build:
      context: .
      dockerfile: Dockerfile.data
    ports:
      - "8002:8002"
    volumes:
      - ./data:/app/data
```

### 5.2 构建和运行

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 停止服务
docker-compose down
```

## 6. 监控和维护

### 6.1 健康检查

- **前端服务**：`http://frontend-server:3000/api/health`
- **后端API服务**：`http://backend-server:8000/api/health`
- **模型服务**：`http://model-server:8001/api/health`
- **数据服务**：`http://data-server:8002/api/health`

### 6.2 日志管理

- **前端服务**：`src/frontend/logs/`
- **后端API服务**：`src/backend/api/logs/`
- **模型服务**：`src/backend/services/model_service/logs/`
- **数据服务**：`spider_image_system/log_dir/`

### 6.3 性能监控

- **内存使用**：使用 `top`、`htop` 或 Prometheus + Grafana
- **CPU使用**：使用 `top`、`htop` 或 Prometheus + Grafana
- **API响应时间**：使用 Prometheus + Grafana 或 New Relic

## 7. 故障排除

### 7.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 前端无法连接到后端 | 网络连接问题 | 检查网络连接和后端服务状态 |
| 后端无法连接到模型服务 | 模型服务未启动 | 启动模型服务并检查配置 |
| 模型服务内存不足 | 模型太大或并发请求过多 | 增加内存或减少并发请求 |
| 数据服务采集失败 | 网络问题或配置错误 | 检查网络连接和采集配置 |

### 7.2 调试技巧

- **检查日志**：查看各个服务的日志文件
- **测试API**：使用 `curl` 或 Postman 测试API接口
- **监控资源**：使用 `top`、`htop` 监控系统资源使用情况
- **网络测试**：使用 `ping`、`traceroute` 测试网络连接

## 8. 扩展和升级

### 8.1 水平扩展

- **前端服务**：使用负载均衡器（如 Nginx）
- **后端API服务**：使用负载均衡器和多实例
- **模型服务**：使用模型服务集群
- **数据服务**：使用分布式采集

### 8.2 垂直扩展

- **增加内存**：为模型服务增加更多内存
- **增加CPU**：为计算密集型服务增加更多CPU
- **使用GPU**：为模型服务使用GPU加速

### 8.3 升级步骤

1. **备份数据**：备份所有数据和配置
2. **更新代码**：`git pull` 获取最新代码
3. **更新依赖**：重新安装依赖
4. **重启服务**：按顺序重启服务
5. **验证服务**：检查所有服务是否正常运行

## 9. 安全考虑

### 9.1 认证和授权

- **API认证**：使用API密钥或JWT进行认证
- **访问控制**：限制API访问权限
- **HTTPS**：使用HTTPS保护数据传输

### 9.2 数据安全

- **数据加密**：对敏感数据进行加密
- **数据备份**：定期备份数据
- **数据清理**：定期清理过期数据

### 9.3 网络安全

- **防火墙**：配置防火墙规则
- **网络隔离**：使用网络隔离保护服务
- **漏洞扫描**：定期进行漏洞扫描

## 10. 总结

通过分层部署架构，系统可以更好地适应不同的部署环境和需求，提高系统的可扩展性、可靠性和可维护性。各个服务可以独立部署和扩展，根据实际需求分配资源，从而提高系统的整体性能和效率。

在部署过程中，需要注意配置环境变量、监控服务状态、及时处理故障，确保系统的稳定运行。同时，要注意安全问题，保护系统和数据的安全。