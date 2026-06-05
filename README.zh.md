# 角色分类系统

![GitHub Actions](https://img.shields.io/github/actions/workflow/status/ard-team/anime_role_detect/docker-image.yml?branch=main)
![Python 版本](https://img.shields.io/badge/python-3.9%2B-blue)
![许可证](https://img.shields.io/badge/license-MIT-green)

基于人工智能的图片识别系统，专门用于识别游戏和动漫中的角色。

## ✨ 核心功能

- **多格式识别**: 支持图片和视频
- **多角色检测**: 识别单张图片中的多个角色（集成YOLOv8/v10）
- **高准确率**: 基于 MobileNetV2、EfficientNet-B0/B3、ResNet50
- **DeepDanbooru集成**: 增强标签生成能力
- **属性预测**: 发色、瞳色、服装等属性
- **RESTful API**: 支持批量处理
- **日志融合**: 从分类日志构建新模型
- **分层架构**: 支持分布式部署
- **模型预热**: 减少首次请求延迟
- **请求防抖/节流**: 防止重复提交
- **图片压缩上传**: 优化带宽占用
- **Redis缓存**: 减少重复计算
- **飞书通知**: 实时进度同步
- **Token自动刷新**: 无缝认证体验

## 🚀 快速开始

### 环境要求
- Python 3.9+
- 16GB+ 内存（模型加载必需）
- NVIDIA GPU（推荐用于推理加速）
- Redis 服务器（用于缓存）
- Docker & Docker Compose（用于容器化部署）

### 安装

```bash
# 克隆仓库
git clone https://github.com/ard-team/anime_role_detect.git
cd anime_role_detect

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements-base.txt
pip install -r requirements-ml.txt   # 用于模型训练/推理
pip install -r requirements-dev.txt  # 用于开发
pip install supervisor               # 用于进程管理

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件配置
```

### 使用 Supervisor 运行（推荐）

```bash
# 启动 Redis（缓存必需）
redis-server &

# 使用 supervisord 启动所有服务
supervisord -c supervisord.conf

# 检查服务状态
supervisorctl status

# 停止所有服务
supervisorctl stop all
```

### Docker 部署

```bash
# 构建并启动所有服务
docker-compose up --build -d

# 检查容器状态
docker-compose ps

# 查看日志
docker-compose logs -f <service_name>

# 停止服务
docker-compose down
```

### 服务访问

| 服务 | URL | 端口 |
|------|-----|------|
| 前端 | http://localhost:3000 | 3000 |
| API 网关 | http://localhost:8080 | 8080 |
| 模型服务 | http://localhost:8000 | 8000 |
| API 服务 | http://localhost:8001 | 8001 |
| 多媒体服务 | http://localhost:8002 | 8002 |
| 搜索服务 | http://localhost:8003 | 8003 |
| Supervisor 管理面板 | http://localhost:9001 | 9001 |

### API 文档
- **Swagger 文档**: `http://localhost:8080/docs`
- **Redoc 文档**: `http://localhost:8080/redoc`

### 默认账号
- **用户名**: `admin` / `user`
- **密码**: `admin123` / `user123`

## 📁 项目结构

```
anime_role_detect/
├── src/                    # 源代码
│   ├── api/                # 后端API（端口8001）
│   ├── services/           # 微服务
│   │   ├── api_gateway/    # API网关（端口8080）
│   │   ├── model_service/  # 模型服务（端口8000）
│   │   ├── multimedia_service/  # 多媒体服务（端口8002）
│   │   ├── search_service/ # 搜索服务（端口8003）
│   │   ├── cache_service/  # Redis缓存服务
│   │   └── video_service/  # 视频识别服务
│   ├── core/               # 核心功能
│   ├── frontend/           # 前端（Next.js）
│   └── run/                # 服务管理脚本
├── models/                 # 模型权重
├── tests/                  # 测试套件
├── docs/                   # 文档
├── skillhub/               # 技能仓库模块
├── scripts/                # 工具脚本（爬虫、数据采集）
├── deployment/             # Kubernetes 部署文件
├── supervisord.conf        # 进程管理器配置
├── docker-compose.yml      # Docker Compose 配置
├── Dockerfile              # 后端 Dockerfile
├── Dockerfile.model        # 模型服务 Dockerfile
├── requirements-base.txt   # 基础依赖
├── requirements-ml.txt     # ML依赖
├── requirements-dev.txt    # 开发依赖
├── pyproject.toml          # 项目配置
└── .env.example           # 环境变量模板
```

## 🌐 API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/classify` | POST | 图片分类 |
| `/api/classify/multi-role` | POST | 多角色检测（YOLO） |
| `/api/search/image` | POST | 图片搜索 |
| `/api/video/recognize` | POST | 视频识别 |
| `/api/health` | GET | 健康检查 |
| `/api/services` | GET | 服务状态 |
| `/api/auth/login` | POST | 用户登录 |
| `/api/auth/refresh` | POST | 刷新Token |

## 🔧 配置

### 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `REDIS_URL` | Redis连接地址 | redis://localhost:6379 |
| `JWT_SECRET` | JWT密钥 | (必需) |
| `JWT_EXPIRE_MINUTES` | Token过期时间（分钟） | 1440（24小时） |
| `MAX_IMAGE_SIZE` | 最大上传大小（MB） | 10 |
| `DEVICE` | 计算设备（cpu/cuda/mps） | auto |

### Docker 配置

项目包含完整的 Docker 支持：

- **docker-compose.yml**: 多服务编排，包含 Redis、MySQL、RabbitMQ 和所有应用服务
- **Dockerfile**: 后端服务多阶段构建
- **Dockerfile.model**: 优化的模型服务镜像
- **deployment/Dockerfile.frontend**: 前端 Next.js 部署（带 Nginx）
- **docker_manager.py**: Docker 操作辅助脚本

## 📊 模型性能

### 最新基准测试结果（2026年6月）

**测试数据集**: 1,480 张图片，覆盖74个角色类别

| 指标 | 数值 |
|------|------|
| Top-1 准确率 | **93.92%** |
| Top-3 准确率 | **96.15%** |
| Top-5 准确率 | **96.89%** |
| 推理速度 | **85.74 FPS** |
| 单图耗时 | **11.66ms** |
| 首次请求延迟 | **< 500ms**（带预热） |
| 登录 API 响应 | **< 220ms** |

### 模型对比

| 模型 | 准确率 | FPS |
|------|--------|-----|
| MobileNetV2 | 94.00% | 379 |
| EfficientNet-B0 | 95.20% | 298 |
| EfficientNet-B3 (优化版) | **93.92%** | **85.74** |
| ResNet50 | 94.80% | 257 |

**当前生产模型**: `efficientnet_b3_loli_optimized_v2_20260529_133654`

## 🔒 安全

- JWT 认证与密钥轮换
- bcrypt/sha256 密码哈希
- 请求速率限制
- 输入验证与清理
- HttpOnly Cookie 存储
- Content Security Policy (CSP) 防护XSS
- Token自动刷新机制

## 🧪 测试

### 自动化测试

```bash
# 运行单元测试
python -m pytest tests/ -v

# 运行集成测试
python -m pytest tests/integration/ -v

# 运行性能测试
python performance_test.py
```

## 📚 文档

详细文档请参考：
- `docs/technical_guide.md` - 技术规格
- `docs/deployment/` - 部署指南（Kubernetes、Ubuntu）
- `docs/training/` - 模型训练指南
- `skillhub/docs/` - 技能仓库文档
- `docs/blog/` - 技术博客

## 🤝 贡献

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解：
- 如何提交 Bug 报告和功能请求
- 代码风格规范
- Pull Request 流程

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

**版本**: v2.2 | **最后更新**: 2026年6月 | **维护者**: ARD Team

---

**关键词**: anime, character-recognition, image-classification, deep-learning, python-api, computer-vision, yolov8, nextjs, docker, microservices