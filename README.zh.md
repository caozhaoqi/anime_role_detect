# 角色分类系统

## 🎯 系统简介

角色分类系统是一个基于人工智能的图片识别工具，专门用于识别游戏和动漫中的角色。系统使用先进的深度学习技术，能够快速准确地识别上传图片中的角色，支持端到端角色检测工作流。

## ✨ 核心功能

- **图片/视频识别**：支持多种格式上传
- **高准确率**：使用CLIP模型和Faiss索引
- **DeepDanbooru集成**：通过动漫标签识别提高分类准确率
- **属性预测**：预测角色属性（发色、瞳色、服装等）
- **实时反馈**：提供识别置信度和详细结果
- **用户友好界面**：直观的Web界面
- **API支持**：RESTful API接口，支持批量处理
- **日志融合**：从分类日志中融合特征，构建新模型
- **端到端工作流**：完整的从数据收集到模型训练的流程

## 🚀 快速开始

### 环境要求

- Python 3.7+
- Flask
- PyTorch
- Transformers
- Ultralytics (YOLOv8)
- Faiss
- EfficientNet-B0
- Requests (用于DeepDanbooru API集成)

### 安装依赖

```bash
# 安装Flask
pip3 install flask

# 安装其他依赖
pip3 install torch torchvision transformers ultralytics faiss-cpu Pillow efficientnet_pytorch requests
```

### 启动系统

#### 1. 启动后端服务

```bash
# 启动后端应用
python3 src/backend/api/app.py
```

后端服务将在 `http://127.0.0.1:8000` 上运行。

#### 2. 启动前端服务

```bash
# 进入前端目录
cd frontend

# 安装依赖（首次运行）
npm install

# 启动Next.js前端应用
npm run dev
```

前端服务将在 `http://localhost:3000` 上运行。

## 📁 项目结构

```
anime_role_detect/
├── data/                  # 数据集目录
├── models/                # 模型存储目录
├── src/                   # 源代码
│   ├── backend/           # 后端代码
│   ├── core/              # 核心功能
│   ├── data/              # 数据相关代码
│   ├── frontend/          # 前端代码
│   ├── models/            # 模型相关代码
│   ├── config/            # 配置文件
│   ├── scripts/           # 实用脚本
│   └── utils/             # 工具函数
├── docs/                  # 详细文档
├── cache/                 # 缓存目录
├── auto_spider_img/       # 自动爬虫图片
├── README.md              # 英文文档
└── README.zh.md           # 中文文档
```

## 🌐 使用方法

### Web界面使用

1. 打开浏览器，访问 `http://localhost:3000`
2. 上传要识别的角色图片
3. 等待系统分析图片
4. 查看识别结果和置信度

### API调用

```bash
# 基本使用
curl -X POST -F "file=@path/to/image.jpg" http://127.0.0.1:8000/api/classify

# 使用模型和属性预测
curl -X POST -F "file=@path/to/image.jpg" -F "use_model=true" -F "use_attributes=true" http://127.0.0.1:8000/api/classify
```

## � 文档

详细技术文档请参考 `docs/` 目录：

- **docs/technical_guide.md**：完整技术文档

## 🤝 贡献

欢迎提交Issue和Pull Request，共同改进系统性能和功能。

## 📄 许可证

本项目基于MIT许可证开源。

## 📞 联系

- Email: zhaoqi.cao@icloud.com
- GitHub: https://github.com/caozhaoqi/anime-role-detect

---

**© 2026 角色分类系统** - 让角色识别变得简单！
