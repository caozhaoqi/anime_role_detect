# 角色分类系统使用说明

## 🎯 系统简介

角色分类系统是一个基于人工智能的图片识别工具，专门用于识别游戏和动漫中的角色。系统使用先进的深度学习技术，能够快速准确地识别上传图片中的角色，现已支持端到端角色检测工作流。

## ✨ 核心功能

- **图片/视频上传识别**：支持多种图片和视频格式上传，自动识别图片和视频中的游戏角色
- **高准确率**：使用CLIP模型和Faiss索引，识别准确率高
- **实时反馈**：提供识别置信度和详细结果
- **用户友好界面**：直观的Web界面，操作简单
- **API支持**：提供RESTful API接口，支持批量处理
- **日志融合**：支持从分类日志中融合特征，构建新模型
- **端到端工作流**：完整的角色检测工作流，从数据收集到模型训练
- **EfficientNet-B0模型**：使用先进的EfficientNet-B0进行角色分类
- **数据收集**：支持通过Bing Image Search API自动收集角色图片
- **数据集分割**：自动将收集的数据分割为训练集和验证集

## 🚀 快速开始

### 环境要求

- Python 3.7+
- Flask
- PyTorch
- Transformers
- Ultralytics (YOLOv8)
- Faiss
- EfficientNet-B0

### 安装依赖

```bash
# 安装Flask
pip3 install flask

# 安装其他依赖（如果尚未安装）
pip3 install torch torchvision transformers ultralytics faiss-cpu Pillow efficientnet_pytorch
```

### 启动系统

#### 1. 启动后端服务

```bash
# 启动Flask后端应用
python3 src/web/web_app.py
```

后端服务将在 `http://127.0.0.1:5001` 上运行。

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
├── data/                   # 数据目录
│   ├── all_characters/            # 所有角色图片（包括BangDream MyGo!）
│   ├── blue_archive_optimized/    # 优化后的蔚蓝档案数据
│   ├── blue_archive_optimized_v2/ # 增强版蔚蓝档案数据
│   ├── augmented_characters/      # 增强后的角色数据
│   └── split_dataset/             # 分割后的训练/验证数据
├── src/                    # 源代码
│   ├── core/               # 核心模块
│   │   ├── classification/         # 分类模块
│   │   ├── feature_extraction/     # 特征提取模块
│   │   ├── preprocessing/          # 预处理模块
│   │   ├── general_classification.py  # 通用分类模块
│   │   └── log_fusion/              # 日志融合模块
│   └── web/                # 网页应用
│       ├── templates/      # HTML模板
│       ├── static/         # 静态文件
│       └── web_app.py      # Flask应用
├── scripts/                # 辅助脚本
│   ├── data_collection/    # 数据收集脚本
│   ├── data_processing/    # 数据处理脚本
│   ├── model_training/     # 模型训练脚本
│   └── workflow/           # 端到端工作流脚本
├── tests/                  # 测试代码
├── README.md               # 英文文档
└── README.zh.md            # 中文文档
```

## 🎮 支持的角色

### 蔚蓝档案 (Blue Archive)

- **星野** (Hoshino)
- **白子** (Shiroko)
- **阿罗娜** (Arona)
- **宫子** (Miyako)
- **日奈** (Hina)
- **优花梨** (Yuuka)

### BangDream MyGo!

- **千早爱音** (Aimi Kanazawa)
- **椎名立希** (Ritsuki)
- **高松灯** (Touko Takamatsu)
- **长崎素世** (Soyo Nagasaki)
- **丰川祥子** (Sakiko Tamagawa)
- **若叶睦** (Mutsumi Wakaba)


### 其他角色

- **原神** (Genshin Impact) 角色
- **间谍过家家** (Spy × Family) 角色
- **进击的巨人** (Attack on Titan) 角色
- **海贼王** (One Piece) 角色
- **火影忍者** (Naruto) 角色
- **东京复仇者** (Tokyo Revengers) 角色

## 🌐 使用方法

### Web界面使用

1. 打开浏览器，访问 `http://127.0.0.1:5001`
2. 点击「选择图片文件」按钮，选择要识别的图片
3. 点击「识别角色」按钮，系统将自动分析图片
4. 等待分析完成，查看识别结果和置信度

### 角色检测工作流

1. 打开浏览器，访问 `http://127.0.0.1:5001/workflow`
2. 以JSON格式输入角色信息（例如：`[{"name": "千早爱音", "series": "bangdream_mygo"}]`）
3. 输入测试图片路径
4. 调整训练参数（批量大小、轮数、学习率等）
5. 点击「开始工作流」按钮开始端到端流程
6. 在终端中监控进度

### API调用

```bash
# 使用curl上传图片并识别
curl -X POST -F "file=@path/to/image.jpg" http://127.0.0.1:5001/api/classify
```

API返回结果示例：

```json
{
  "filename": "image.jpg",
  "role": "蔚蓝档案_星野",
  "similarity": 0.98,
  "boxes": []
}
```

## 📊 系统性能

### 识别准确率

| 角色 | 准确率 |
|------|--------|
| 优花梨 | 100% |
| 阿罗娜 | 83.33% |
| 宫子 | 60% |
| 星野 | 40% |
| 白子 | 37.50% |
| 日奈 | 30% |

### 平均处理时间

- 图片上传：~1秒
- 预处理：~0.5秒
- 特征提取：~0.3秒
- 分类：~0.1秒
- 总时间：~2秒

### 模型训练性能

- **训练速度**：在MPS上约2.05 batch/s
- **每轮耗时**：约1小时8分钟
- **总训练时间**：50轮约54小时
- **初始损失**：4.79
- **当前损失**：1.06（第1轮后）
- **最佳验证准确率**：0.9386（第18轮后）

## 🔧 技术实现

### 核心技术栈

| 技术 | 用途 |
|------|------|
| Python | 主要开发语言 |
| Flask | Web应用框架 |
| PyTorch | 深度学习框架 |
| CLIP | 图像特征提取 |
| YOLOv8 | 角色检测 |
| Faiss | 相似性搜索 |
| EfficientNet-B0 | 角色分类 |
| HTML/CSS | 前端界面 |

### 端到端工作流

1. **数据收集**：通过Bing Image Search API收集角色图片
2. **数据集分割**：将收集的数据分割为训练集（80%）和验证集（20%）
3. **模型训练**：使用数据增强训练EfficientNet-B0模型
4. **模型评估**：在验证集上评估模型性能
5. **角色检测**：使用训练好的模型检测新图片中的角色

### 模型训练流水线

- **数据增强**：随机大小裁剪、水平/垂直翻转、旋转、颜色抖动
- **优化器**：AdamW带权重衰减
- **学习率调度**：余弦退火
- **批量大小**：16
- **训练轮数**：50
- **初始学习率**：5e-5

### 分布式训练

系统支持跨多个GPU的分布式训练，以加快训练速度：

```bash
# 启动分布式训练
python scripts/model_training/train_model_distributed.py --batch_size 8 --num_epochs 50 --learning_rate 5e-5 --weight_decay 1e-4 --num_workers 4
```

**主要特性：**
- 自动检测可用GPU
- DistributedDataParallel (DDP) 实现
- 多GPU同步训练
- 自动批量大小缩放（每个GPU的批量大小）
- 如果只有一个GPU，自动回退到单GPU模式

**预期加速效果：**
- 2个GPU：约2倍训练速度
- 4个GPU：约4倍训练速度
- 8个GPU：约8倍训练速度

**注意：** 分布式训练需要至少2个GPU才能有效。

## 📈 系统优化

### 性能优化

- **单例模式**：避免重复初始化模型，减少内存使用
- **缓存机制**：缓存已处理的结果，提高响应速度
- **批量处理**：支持批量图像分类，提高处理效率
- **异步加载**：模型懒加载，加快系统启动速度
- **MPS加速**：使用Apple Silicon GPU加快训练和推理速度

### 用户体验优化

- **加载动画**：添加上传和处理时的加载动画
- **响应式设计**：适配不同屏幕尺寸
- **实时反馈**：提供详细的识别结果和置信度
- **错误处理**：友好的错误提示
- **工作流界面**：专门的端到端角色检测工作流界面

## 🔄 日志融合功能

### 功能介绍

日志融合功能是系统的一个重要特性，它能够：

- **收集分类日志**：自动收集每次分类的结果和特征
- **融合特征**：将多个分类结果的特征融合成一个新的模型
- **持续学习**：从历史分类数据中学习，不断提高分类准确率
- **模型更新**：定期更新模型，保持系统性能

### 使用方法

#### 1. 收集分类日志

系统会自动收集每次分类的结果，包括：
- 上传的图片
- 提取的特征向量
- 分类结果
- 置信度

#### 2. 融合特征构建新模型

```bash
# 运行日志融合脚本
python3 src/core/log_fusion/log_fusion.py --log_dir ./logs --output_model ./models/fused_model
```

#### 3. 使用新模型进行分类

系统会自动使用最新构建的模型进行分类，无需额外配置。

## 🤝 贡献指南

欢迎提交Issue和Pull Request，共同改进系统性能和功能。

## 📄 许可证

本项目基于MIT许可证开源。

## 📞 联系我们

如有问题或建议，请通过以下方式联系：

- Email: zhaoqi.cao@icloud.com
- GitHub: https://github.com/caozhaoqi/anime-role-detect

---

**© 2026 角色分类系统** - 让角色识别变得简单！

## 📋 最新更新

### 2026年2月

- **新增BangDream MyGo!角色**，每个角色50-102张图片
- **实现端到端角色检测工作流**，从数据收集到模型训练
- **添加EfficientNet-B0模型**用于角色分类
- **改进API文档**，支持GET请求
- **增强数据处理流水线**，自动数据集分割
- **添加MPS加速**，在Apple Silicon上更快训练
- **自动化数据集扩充**：从多个源自动收集角色图片，支持21个角色的625张图像
- **模型蒸馏技术**：使用EfficientNet-B0作为教师模型，MobileNetV2作为学生模型，压缩率1.79x
- **在线学习系统**：支持增量学习和新角色添加，自动扩展分类器
- **多模态融合系统**：结合图像和文本信息进行角色识别，提高准确性
- **综合评估框架**：评估所有改进的性能，提供客观的评估指标
- **新增视频检测支持**：支持动漫角色视频识别
- **实现视频帧分析**：逐帧检测视频中的角色
- **更新前端界面**：支持视频上传和播放
- **增强API**：同时处理图片和视频文件的检测

