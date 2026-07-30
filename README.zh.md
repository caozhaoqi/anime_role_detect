# 角色分类系统

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
- **密码**: 通过环境变量 `ADMIN_PASSWORD` 和 `USER_PASSWORD` 设置
- **说明**: 如果未设置，首次启动时会自动生成随机密码

## 📁 项目结构

```
anime_role_detect/
├── src/                    # 源代码（可编辑安装：pip install -e .）
│   ├── api/                # 后端API服务（FastAPI，端口8001）
│   │   └── routes/         # API路由（classify、auth、collector 等）
│   ├── services/           # 微服务
│   │   ├── api_gateway/    # API网关（端口8080）
│   │   ├── model_service/  # 模型服务（端口8000）
│   │   ├── multimedia_service/  # 多媒体服务（端口8002）
│   │   ├── search_service/ # 搜索服务 + worker（端口8003）
│   │   ├── inference_worker/   # CLIP 推理 worker
│   │   ├── cache_service/  # Redis缓存服务
│   │   ├── video_service/  # 视频识别服务
│   │   ├── messaging/      # 消息队列（aio_pika）
│   │   └── processor/      # 模型加载器 / 图像处理器
│   ├── core/               # 核心功能（分类、检测、打标、识别、ocr、日志、配置、缓存…）
│   ├── data/               # 数据采集与预处理流水线
│   ├── data_pipeline/      # 数据清洗 / 构建 / webui 流水线
│   ├── data_collection/    # （遗留）关键词采集入口
│   ├── models/             # 训练 / 评估 / 预测模块
│   ├── tasks/              # Celery 任务（分类、图像、视频、模型、清理）
│   ├── utils/              # 公共工具（图像、http、并发、监控）
│   ├── middleware/         # HTTP 中间件
│   ├── frontend/           # 前端（Next.js 15 + React 18 + TypeScript）
│   └── run/                # 服务管理与监控面板
├── models/                 # 模型权重（git 忽略）
├── tests/                  # 测试套件（unit / integration / model / workflow）
├── docs/                   # 文档（架构、部署、训练、博客）
├── scripts/                # 工具脚本（k8s、监控、data_*、评估…）
│   └── skillhub/           # ⚠️ 归档的实验子项目（88MB，无引用）
├── archived/               # 历史 / 损坏模块（spider_image_system、arona…）
├── deployment/             # Kubernetes 与 Docker 部署文件
├── k8s/                    # Kustomize 覆盖（base / ci）+ 本地 registry 辅助
├── config/                 # 配置模板
├── supervisord.conf        # 进程管理器配置（11 个程序）
├── docker-compose.yml      # Docker Compose 配置
├── Dockerfile              # 后端 Dockerfile
├── Dockerfile.model        # 模型服务 Dockerfile
├── requirements-base.txt   # 基础依赖（base 镜像用）
├── requirements-ml.txt     # ML 依赖
├── requirements-model-service.txt
├── requirements-scripts.txt
├── requirements-dev.txt    # 开发依赖
├── pyproject.toml          # 项目配置（v2.3.0）
└── .env.example            # 环境变量模板
```

> **未使用 / 遗留代码说明（2026-07-30 清理）**
> - 死亡模块 `src/models/training/convert_model_format.py`（语法损坏、无引用）→ 移至 `archived/broken_modules/`。
> - `scripts/skillhub/` 为遗留实验子项目（自带 venv，无任何引用），保留作历史，不纳入构建。
> - `src/run/start_all.py`、`start_all_stable.py`、`application.py`、`start_core.py` 为遗留/备用启动器（supervisord/k8s/docker 未使用），保留为开发工具。
> - 清理了 `src/` 内约 30 处未使用导入 / 未使用变量（低风险 lint 清理）。
> - 运行时产物（`logs/`、`data/`、`models/`、`*.db`、`dump.rdb`、各类缓存）均已 git 忽略。

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

## 📊 模型性能

### 最新基准测试结果（2026-07-28，`scripts/model_evaluation/benchmark_results.json`）

> ⚠️ **数据说明**：测试集（`final_dataset`，1,275 张图，51 类 × 25 张）目前**与训练集同源采样**。所报准确率为训练态表现，独立数据集上的真实泛化精度**尚未验证**，预计会偏低。详见 `docs/architecture/PROJECT_STRUCTURE.md` 已知问题。

**生产模型**：`efficientnet_b3`（`models/efficientnet_b3/model_best.pth`），51 类，256×256 输入，45.99 MB，11.9M 参数，于 Apple MPS 评测。

| 指标 | 数值 |
|------|------|
| Top-1 准确率 | **84.00%** |
| Top-5 准确率 | **93.96%** |
| Macro-F1 | **0.8401** |
| 单图延迟 | **29.04 ms**（34.44 FPS） |
| 批量(32)吞吐 | **31.11 FPS** |
| 首次请求延迟 | **< 500ms**（带预热） |

**最弱类别**（准确率）：`silver_wolf` 64%，`Klee` / `aglaea` / `clorinde` / `kafka` 68%。
**最易混淆对**：`clorinde` → `Furina`。

### 多角色检测（YOLOv8n）

`yolov8n.pt` 为 COCO 预训练基线（6.25 MB，3.15M 参数），**未**在动漫角色上微调 —— 平均置信度 0.444，MPS 下约 4 FPS。微调待完成（见已知问题）。

### 模型对比（参考，训练态）

| 模型 | 类别数 | Top-1 | 说明 |
|------|--------|-------|------|
| EfficientNet-B3（生产） | 51 | **84.00%** | 当前模型 |
| EfficientNet-B0 / MobileNetV2 / ResNet50 | — | — | 早期实验，见 `docs/blog/10_training_and_evaluation.md` |

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

# 运行模型基准（生成 scripts/model_evaluation/benchmark_results.json）
python scripts/model_evaluation/run_benchmark.py
```

## 📚 文档

详细文档请参考：
- `docs/architecture/PROJECT_STRUCTURE.md` - 项目结构与已知问题
- `docs/deployment/` - 部署指南（Kubernetes、Ubuntu）
- `docs/training/` - 模型训练指南
- `docs/blog/` - 技术博客

## 🤝 贡献

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解：
- 如何提交 Bug 报告和功能请求
- 代码风格规范
- Pull Request 流程

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

**版本**: v2.3.0 | **最后更新**: 2026年7月 | **维护者**: ARD Team

---

**关键词**: anime, character-recognition, image-classification, deep-learning, python-api, computer-vision, yolov8, nextjs, docker, microservices