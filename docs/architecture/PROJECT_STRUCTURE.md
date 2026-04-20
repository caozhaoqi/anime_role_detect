# 项目结构说明

## 1. 项目概览

本项目是一个动漫角色检测系统，用于识别和分类不同动漫和游戏中的角色。项目采用深度学习技术，通过数据采集、处理、模型训练和部署的完整流程，实现角色的自动识别。

## 2. 目录结构

```
anime_role_detect/
├── data/                          # 数据集目录（需要创建）
│   ├── raw/                       # 原始数据
│   │   ├── characters/           # 原始角色图片
│   │   └── annotations/       # 原始标注数据
│   ├── processed/                # 处理后的数据
│   │   ├── augmented_dataset/    # 增强后的数据集
│   │   └── split_dataset/    # 分割后的训练/验证数据
│   └── external/                # 外部数据
├── models/                        # 模型存储目录（需要创建）
│   ├── checkpoints/           # 训练检查点
│   ├── pretrained/             # 预训练模型
│   └── exported/               # 导出的生产模型
├── src/                           # 主源代码目录
│   ├── core/                    # 核心功能模块
│   │   ├── classification/      # 分类模块
│   │   ├── feature_extraction/ # 特征提取模块
│   │   ├── preprocessing/   # 预处理模块
│   │   ├── tagging/         # 标签模块
│   │   ├── keypoint/        # 关键点检测模块
│   │   ├── log_fusion/      # 日志融合模块
│   │   └── logging/          # 日志模块
│   ├── data/                    # 数据相关代码
│   │   ├── collection/        # 数据收集模块
│   │   │   ├── spider/    # 爬虫模块
│   │   │   └── annotation/ # 标注模块
│   │   ├── preprocessing/   # 数据预处理脚本
│   │   └── augmentation/    # 数据增强脚本
│   ├── models/                  # 模型相关代码
│   │   ├── training/        # 模型训练脚本
│   │   ├── evaluation/      # 模型评估脚本
│   │   └── deployment/     # 模型部署脚本
│   ├── backend/                 # 后端代码
│   │   ├── api/             # API实现
│   │   └── web/             # Web界面
│   ├── utils/                   # 工具函数
│   ├── config/                  # 配置文件
│   └── scripts/                # 实用脚本
├── frontend/                      # 前端代码（Next.js）
│   ├── app/
│   ├── public/
│   └── ...
├── spider_image_system/           # 独立爬虫系统（作为子项目保留）
├── tests/                       # 测试目录
│   ├── unit/                 # 单元测试
│   ├── integration/          # 集成测试
│   └── e2e/                # 端到端测试
├── docs/                        # 文档目录
│   ├── PROJECT_STRUCTURE.md   # 项目结构说明
│   ├── TECHNICAL_ARCHITECTURE.md
│   ├── TRAINING_GUIDE.md
│   └── ...
├── cache/                       # 缓存目录（需要创建）
├── logs/                        # 日志目录（需要创建）
├── .gitignore
├── README.md
├── README.zh.md
├── requirements.txt
└── main.py
```

## 3. 架构说明

### 3.1 核心原则

1. **单一源代码入口**：所有核心代码统一在 `src/` 目录下
2. **模块化设计**：按功能模块清晰划分
3. **数据与代码分离**：数据、模型、代码各自独立
4. **测试独立**：测试代码与源代码分离

### 3.2 模块说明

#### 3.2.1 数据模块 (data/)

- **raw/**: 存储原始采集的数据，保持原始状态
- **processed/**: 经过处理的数据，用于模型训练
- **external/**: 外部来源的第三方数据

#### 3.2.2 核心模块 (src/core/)

- **classification/**: 角色分类核心逻辑
- **feature_extraction/**: 特征提取
- **preprocessing/**: 图像预处理
- **tagging/**: 图像标签生成
- **keypoint/**: 关键点检测
- **log_fusion/**: 日志融合与模型更新
- **logging/**: 全局日志管理

#### 3.2.3 数据模块 (src/data/)

- **collection/spider/**: 统一的爬虫模块（整合原 auto_spider_img 和 spider_image_system 核心功能）
- **collection/annotation/**: 数据标注工具
- **preprocessing/**: 数据预处理脚本
- **augmentation/**: 数据增强脚本

#### 3.2.4 模型模块 (src/models/)

- **training/**: 模型训练脚本
- **evaluation/**: 模型评估脚本
- **deployment/**: 模型部署脚本

#### 3.2.5 后端模块 (src/backend/)

- **api/**: RESTful API 实现
- **web/**: Web 界面

#### 3.2.6 前端模块 (frontend/)

- Next.js 前端应用，独立于 src/ 目录

#### 3.2.7 测试模块 (tests/)

- **unit/**: 单元测试
- **integration/**: 集成测试
- **e2e/**: 端到端测试

### 3.3 遗留系统处理

#### spider_image_system/

作为独立子项目保留，原因：
1. 是一个完整的独立应用，有自己的 UI 和功能
2. 可以通过 API 或命令行与主系统集成
3. 便于独立开发和维护

## 4. 迁移建议

### 4.1 第一阶段：目录结构创建

1. 创建缺失的目录：
   - `data/` 及其子目录
   - `models/` 及其子目录
   - `cache/`
   - `logs/`

### 4.2 第二阶段：代码整合

1. 将 `arona/` 下的模块按功能迁移到 `src/` 对应目录
2. 将 `auto_spider_img/` 的核心爬虫逻辑整合到 `src/data/collection/spider/`
3. 保留 `spider_image_system/` 作为独立子项目

### 4.3 第三阶段：清理

1. 逐步移除 `arona/` 目录
2. 更新导入路径
3. 更新文档和配置

## 5. 核心功能

### 5.1 数据采集

- 统一的爬虫系统
- 基于系列的角色图像采集
- 基于关键词的图像搜索
- 图像质量控制和过滤

### 5.2 数据处理

- 数据质量评估
- 数据增强
- 数据集分割
- 数据格式转换

### 5.3 模型训练

- 基础模型训练
- 模型优化和调优
- 模型集成
- 模型格式转换

### 5.4 模型推理

- 角色识别
- 置信度评估
- 结果输出

## 6. 使用流程

1. **配置项目**: 修改配置参数
2. **数据采集**: 运行数据采集脚本，获取角色图像
3. **数据处理**: 运行数据处理脚本，准备训练数据
4. **模型训练**: 运行模型训练脚本，训练角色识别模型
5. **模型评估**: 运行模型评估脚本，评估模型性能
6. **模型部署**: 转换模型为生产格式，部署到生产环境

## 7. 技术栈

- **编程语言**: Python 3.9+
- **深度学习框架**: PyTorch
- **前端框架**: Next.js
- **图像处理**: OpenCV, PIL
- **数据存储**: 本地文件系统
- **模型格式**: PyTorch, ONNX

## 8. 维护说明

- **数据更新**: 定期更新角色列表和图像数据
- **模型更新**: 定期重新训练模型，以适应新数据
- **配置管理**: 集中管理配置参数，避免硬编码
- **代码规范**: 遵循 Python 代码规范，保持代码可读性

## 9. 总结

本项目采用模块化设计，结构清晰，功能完整。通过合理的目录组织和模块划分，提高了代码的可维护性和可扩展性。同时，通过集中的配置管理，使得项目参数的调整更加方便。

项目的核心价值在于实现了从数据采集到模型部署的完整流程，为动漫角色的自动识别提供了可行的解决方案。