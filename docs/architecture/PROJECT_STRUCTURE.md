# 项目代码结构

## 概述

本项目采用模块化架构设计，将功能划分为多个独立的模块，便于维护和扩展。

## 目录结构

```
anime_role_detect/
├── .github/                    # GitHub配置
│   └── workflows/              # CI/CD工作流
├── auto_spider_img/            # 自动爬虫配置
│   ├── html/                   # HTML模板
│   ├── keywords/               # 爬虫关键词
│   └── tests/                  # 爬虫测试
├── config/                     # 配置文件
├── deployment/                 # 部署配置
│   ├── Dockerfile.*            # Docker镜像构建
│   └── *.yaml                  # Kubernetes部署文件
├── docs/                       # 项目文档
│   ├── architecture/           # 架构文档
│   ├── blog/                   # 技术博客
│   ├── deployment/             # 部署指南
│   ├── testing/                # 测试报告
│   └── training/               # 训练指南
├── monitoring/                 # 监控配置
├── nsfw_model_img/             # NSFW模型（Java）
├── scripts/                    # Python脚本模块
│   ├── classification/         # 分类模块
│   ├── common/                 # 公共工具模块
│   │   ├── __init__.py
│   │   ├── database_utils.py   # 数据库工具
│   │   ├── download_utils.py   # 下载工具
│   │   └── notification_utils.py # 通知工具
│   ├── data_cleaning/          # 数据清理模块
│   ├── data_collection/        # 数据采集模块
│   │   ├── batch_collectors/   # 批量采集器
│   │   ├── database/           # 数据库操作
│   │   ├── downloaders/        # 图片下载器
│   │   ├── single_collectors/  # 单角色采集器
│   │   ├── tests/              # 测试用例
│   │   └── utils/              # 辅助工具
│   ├── model_testing/          # 模型测试模块
│   ├── model_training/         # 模型训练模块
│   ├── __init__.py             # 模块初始化
│   └── main.py                 # 项目主入口
├── spider_image_system/        # 图片爬虫系统
├── .gitignore                  # Git忽略配置
├── requirements.txt            # Python依赖
├── README.md                   # 项目说明
└── README.zh.md                # 中文说明
```

## 模块说明

### 1. 公共工具模块 (scripts/common/)

| 文件 | 功能说明 |
|------|----------|
| `download_utils.py` | 图片下载相关工具函数 |
| `notification_utils.py` | 通知服务（飞书、Telegram） |
| `database_utils.py` | SQLite数据库操作 |

### 2. 数据采集模块 (scripts/data_collection/)

| 子目录 | 功能说明 |
|--------|----------|
| `batch_collectors/` | 批量角色数据采集 |
| `single_collectors/` | 单个角色数据采集 |
| `downloaders/` | 图片下载器 |
| `database/` | 数据库初始化和操作 |
| `utils/` | 数据采集辅助工具 |

### 3. 数据清理模块 (scripts/data_cleaning/)

提供数据清洗、去重、质量检查等功能。

### 4. 模型训练模块 (scripts/model_training/)

提供模型训练、优化和导出功能。

### 5. 模型测试模块 (scripts/model_testing/)

提供模型测试、性能评估和压力测试功能。

## 使用方式

### 命令行入口

```bash
# 使用主入口脚本
python scripts/main.py <command> [options]

# 命令列表
- download    # 下载图片
- collect     # 数据采集
- clean       # 数据清理
- stats       # 统计分析
- help        # 显示帮助
```

### 模块导入

```python
# 导入公共模块
from scripts.common import download_image, setup_logger, ImageDatabase

# 使用下载工具
logger = setup_logger('my_module')
success, message = download_image(url, save_dir)

# 使用数据库
db = ImageDatabase('path/to/db')
```

## 编码规范

- Python版本：3.8+
- 代码风格：PEP 8
- 文件编码：UTF-8
- 命名规范：
  - 模块名：snake_case
  - 类名：PascalCase
  - 函数/变量名：snake_case

## 扩展开发

### 添加新的下载器

1. 在 `scripts/data_collection/downloaders/` 目录下创建新文件
2. 继承或使用 `download_utils.py` 中的工具函数
3. 在 `__init__.py` 中导出模块

### 添加新的通知渠道

1. 在 `scripts/common/notification_utils.py` 中添加新的通知器类
2. 在 `CompositeNotifier` 中注册新的通知器
3. 在 `__init__.py` 中导出

## 依赖管理

项目依赖通过 `requirements.txt` 管理：

```bash
# 安装依赖
pip install -r requirements.txt

# 添加新依赖
pip install <package>
pip freeze > requirements.txt
```
