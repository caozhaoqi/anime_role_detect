# 项目结构规划

## 目标
整理项目结构，创建清晰、合理的代码组织，提高代码可维护性和可扩展性。

## 新的项目结构

```
anime_role_detect/
├── README.md                 # 项目说明文档
├── README.zh.md              # 中文项目说明文档
├── requirements.txt          # 项目依赖
├── .gitignore                # Git忽略文件
├──
├── data/                     # 数据目录
│   ├── train/                # 训练数据
│   ├── test/                 # 测试数据
│   └── raw/                  # 原始数据
├──
├── models/                   # 模型目录
│   ├── augmented_training/   # 增强训练模型
│   ├── arona_plana/          # 阿罗娜普拉娜模型
│   ├── optimized/            # 优化模型
│   └── role_index.faiss      # 角色索引文件
├──
├── src/                      # 源代码目录
│   ├── backend/              # 后端代码
│   │   ├── api/              # API接口
│   │   │   ├── app.py        # FastAPI应用
│   │   │   └── run_api.py    # API启动脚本
│   │   ├── services/         # 后端服务
│   │   │   ├── classification_service.py  # 分类服务
│   │   │   └── tagging_service.py         # 标签服务
│   │   └── utils/            # 后端工具
│   ├──
│   ├── core/                 # 核心算法
│   │   ├── classification/   # 分类模块
│   │   ├── feature_extraction/  # 特征提取
│   │   ├── preprocessing/    # 预处理
│   │   ├── tagging/          # 标签生成
│   │   └── keypoint/         # 关键点检测
│   ├──
│   ├── frontend/             # 前端代码
│   │   ├── app/              # 前端应用
│   │   ├── public/           # 静态资源
│   │   └── package.json      # 前端依赖
│   ├──
│   ├── scripts/              # 脚本工具
│   │   ├── data_preparation/ # 数据准备
│   │   ├── model_training/   # 模型训练
│   │   └── evaluation/       # 评估脚本
│   ├──
│   └── utils/                # 通用工具
│       ├── cache_manager.py  # 缓存管理
│       ├── monitoring_system.py  # 监控系统
│       └── distributed_manager.py  # 分布式管理
├──
├── docs/                     # 文档目录
│   ├── technical_architecture.md  # 技术架构
│   ├── training_guide.md     # 训练指南
│   └── usage.md              # 使用说明
├──
├── tests/                    # 测试目录
│   ├── unit/                 # 单元测试
│   └── integration/          # 集成测试
└──
    └── test_results/         # 测试结果
```

## 结构说明

1. **根目录**：包含项目的基本配置文件和说明文档

2. **data/**：存放数据相关文件
   - `train/`：训练数据
   - `test/`：测试数据
   - `raw/`：原始数据

3. **models/**：存放模型文件
   - 按模型类型组织子目录
   - 包含角色索引文件

4. **src/**：源代码目录
   - `backend/`：后端代码
     - `api/`：API接口实现
     - `services/`：后端服务逻辑
     - `utils/`：后端工具类
   - `core/`：核心算法模块
     - `classification/`：分类算法
     - `feature_extraction/`：特征提取
     - `preprocessing/`：图像预处理
     - `tagging/`：标签生成
     - `keypoint/`：关键点检测
   - `frontend/`：前端代码
     - 统一管理前端资源
   - `scripts/`：脚本工具
     - 按功能组织子目录
   - `utils/`：通用工具类

5. **docs/**：文档目录
   - 技术架构、训练指南等文档

6. **tests/**：测试目录
   - 单元测试和集成测试

7. **test_results/**：测试结果
   - 存储测试报告和结果

## 调整步骤

1. **清理根目录**：
   - 将临时脚本移至scripts目录
   - 移除不必要的文件

2. **整理src目录**：
   - 重新组织backend目录结构
   - 统一前端代码到frontend目录
   - 清理重复的模块和文件

3. **优化models目录**：
   - 移除不可用的模型
   - 按类型组织模型文件

4. **完善文档**：
   - 更新技术架构文档
   - 添加使用说明

5. **测试验证**：
   - 确保调整后的代码能够正常运行
   - 验证API接口功能

## 预期效果

- 代码结构清晰，易于理解和维护
- 模块职责明确，降低耦合度
- 便于新功能的扩展和集成
- 提高开发效率和代码质量
