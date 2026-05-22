# 角色分类系统

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
# 安装依赖
pip3 install -r requirements.txt

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
└── skillhub/               # 技能仓库模块
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

| 模型 | 准确率 | FPS |
|------|--------|-----|
| MobileNetV2 | 94.00% | 379 |
| EfficientNet-B0 | 95.20% | 298 |
| EfficientNet-B3 | 96.80% | 188 |
| ResNet50 | 94.80% | 257 |

## 📚 文档

详细文档请参考：
- `docs/technical_guide.md` - 技术规格
- `docs/deployment/` - 部署指南
- `docs/training/` - 模型训练指南
- `skillhub/docs/` - 技能仓库文档

## 📄 许可证

MIT License

---

**版本**: v2.0 | **最后更新**: 2026年5月