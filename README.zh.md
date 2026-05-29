# 角色分类系统

![GitHub Actions](https://img.shields.io/github/actions/workflow/status/ard-team/anime_role_detect/ci-cd.yml?branch=main)
![Python 版本](https://img.shields.io/badge/python-3.9%2B-blue)
![许可证](https://img.shields.io/badge/license-MIT-green)
![代码覆盖率](https://img.shields.io/codecov/c/github/ard-team/anime_role_detect)
![最后提交](https://img.shields.io/github/last-commit/ard-team/anime_role_detect)

基于人工智能的图片识别系统，专门用于识别游戏和动漫中的角色。

## ✨ 核心功能

- **多格式识别**: 支持图片和视频
- **多角色检测**: 识别单张图片中的多个角色
- **高准确率**: 基于 MobileNetV2、EfficientNet-B0/B3、ResNet50
- **DeepDanbooru集成**: 增强标签生成能力
- **属性预测**: 发色、瞳色、服装等属性
- **RESTful API**: 支持批量处理
- **日志融合**: 从分类日志构建新模型
- **分层架构**: 支持分布式部署

## 🚀 快速开始

### 环境要求
- Python 3.9+
- 16GB+ 内存（模型加载必需）
- NVIDIA GPU（推荐用于推理加速）

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

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件配置

# 启动服务
python3 src/application.py start --core
python3 src/application.py start --services gateway
```

### Docker 部署

```bash
docker-compose up --build -d
```

### API 网关（主入口）
- **网关**: `http://localhost:8080`
- **API 文档**: `http://localhost:8080/docs`

## 📁 项目结构

```
anime_role_detect/
├── src/                    # 源代码
│   ├── api/                # 后端API（端口8001）
│   ├── services/           # 微服务
│   │   ├── api_gateway/    # API网关（端口8080）
│   │   ├── model_service/  # 模型服务（端口8000）
│   │   └── multimedia/     # 多媒体服务（端口8002）
│   ├── core/               # 核心功能
│   └── frontend/           # 前端（Next.js）
├── models/                 # 模型权重
├── tests/                  # 测试套件
├── docs/                   # 文档
├── skillhub/               # 技能仓库模块
├── scripts/                # 工具脚本
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
| `/api/classify/multi-role` | POST | 多角色检测 |
| `/api/search/image` | POST | 图片搜索 |
| `/api/video/recognize` | POST | 视频识别 |
| `/api/health` | GET | 健康检查 |
| `/api/services` | GET | 服务状态 |

## 📊 模型性能

### 最新基准测试结果（2026年5月）

**测试数据集**: 1,480 张图片，覆盖74个角色类别

| 指标 | 数值 |
|------|------|
| Top-1 准确率 | **93.92%** |
| Top-3 准确率 | **96.15%** |
| Top-5 准确率 | **96.89%** |
| 推理速度 | **85.74 FPS** |
| 单图耗时 | **11.66ms** |

### 模型对比

| 模型 | 准确率 | FPS |
|------|--------|-----|
| MobileNetV2 | 94.00% | 379 |
| EfficientNet-B0 | 95.20% | 298 |
| EfficientNet-B3 (优化版) | **93.92%** | **85.74** |
| ResNet50 | 94.80% | 257 |

**当前生产模型**: `efficientnet_b3_loli_optimized_v2_20260529_133654`

## 📚 文档

详细文档请参考：
- `docs/technical_guide.md` - 技术规格
- `docs/deployment/` - 部署指南
- `docs/training/` - 模型训练指南
- `skillhub/docs/` - 技能仓库文档

## 🤝 贡献

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解：
- 如何提交 Bug 报告和功能请求
- 代码风格规范
- Pull Request 流程

## 🔒 安全

- JWT 认证与密钥轮换
- bcrypt/sha256 密码哈希
- 请求速率限制
- 输入验证与清理

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

**版本**: v2.0 | **最后更新**: 2026年5月 | **维护者**: ARD Team

---

**关键词**: anime, character-recognition, image-classification, deep-learning, python-api, computer-vision
