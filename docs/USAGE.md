# 项目使用说明

## 1. 环境准备

### 1.1 系统要求

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.7+ (推荐，用于GPU加速)
- 至少8GB内存
- 至少100GB磁盘空间

### 1.2 依赖安装

```bash
# 克隆项目
git clone <repository_url>
cd anime_role_detect

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 2. 数据采集

### 2.1 配置角色列表

1. **编辑系列配置文件**: `auto_spider_img/anime_set.txt`
   ```
   # 格式: 系列名称, 系列代码
   原神, genshin_impact
   崩坏 星穹铁道, honkai_star_rail
   间谍过家家, spy_x_family
   ```

2. **创建角色列表文件**: 在 `auto_spider_img/characters/` 目录下创建对应系列的角色列表文件
   ```bash
   # 示例: 创建原神角色列表文件
   touch auto_spider_img/characters/原神.txt
   ```

3. **添加角色名称**: 在角色列表文件中添加角色名称，每行一个
   ```
   迪卢克
   可莉
   胡桃
   宵宫
   ```

### 2.2 运行数据采集

#### 2.2.1 基于系列的采集

```bash
# 运行系列基于采集器
python -m src.data_collection.series_based_collector --mode priority
```

**参数说明**:
- `--mode`: 采集模式，可选值：
  - `priority`: 优先采集数据不足的角色
  - `all`: 采集所有角色
  - `specific`: 采集指定角色

#### 2.2.2 基于关键词的采集

```bash
# 运行关键词基于采集器
python -m src.data_collection.keyword_based_collector --keywords "原神 迪卢克" --max-images 50
```

**参数说明**:
- `--keywords`: 搜索关键词
- `--max-images`: 最大图像数量

### 2.3 数据质量控制

```bash
# 运行数据质量控制器
python -m src.data_processing.data_quality_controller --input-dir data/train
```

**功能**:
- 过滤低质量图像
- 检查图像尺寸和宽高比
- 生成数据质量报告

## 3. 数据处理

### 3.1 数据增强

```bash
# 运行数据增强
python -m src.data_processing.data_augmentation --input-dir data/train --output-dir data/train_augmented
```

**参数说明**:
- `--input-dir`: 输入目录
- `--output-dir`: 输出目录

### 3.2 数据集分割

```bash
# 运行数据集分割
python -m src.data_processing.dataset_splitter --input-dir data/all --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
```

**参数说明**:
- `--input-dir`: 输入目录
- `--train-ratio`: 训练集比例
- `--val-ratio`: 验证集比例
- `--test-ratio`: 测试集比例

## 4. 模型训练

### 4.1 基础模型训练

```bash
# 运行基础模型训练
python -m src.model_training.train_model --train-dir data/train --val-dir data/val
```

**参数说明**:
- `--train-dir`: 训练数据目录
- `--val-dir`: 验证数据目录

### 4.2 改进模型训练

```bash
# 运行改进模型训练
python -m src.model_training.train_improved_model --train-dir data/train --val-dir data/val
```

### 4.3 超参数调优

```bash
# 运行超参数调优
python -m src.model_training.hyperparameter_tuning --train-dir data/train --val-dir data/val
```

### 4.4 模型集成

```bash
# 运行模型集成
python -m src.model_training.model_ensemble --models models/checkpoints/model1.pth models/checkpoints/model2.pth
```

**参数说明**:
- `--models`: 模型文件列表

## 5. 模型评估

### 5.1 评估模型性能

```bash
# 运行模型评估
python -m src.model_training.evaluate_model --model models/checkpoints/best_model.pth --test-dir data/test
```

**参数说明**:
- `--model`: 模型文件路径
- `--test-dir`: 测试数据目录

### 5.2 转换模型格式

```bash
# 转换为ONNX格式
python -m src.model_training.convert_to_onnx --model models/checkpoints/best_model.pth --output models/onnx/model.onnx
```

**参数说明**:
- `--model`: 模型文件路径
- `--output`: 输出ONNX文件路径

## 6. 模型推理

### 6.1 单张图像推理

```bash
# 运行单张图像推理
python -m src.inference.infer_single --model models/onnx/model.onnx --image test_image.jpg
```

**参数说明**:
- `--model`: 模型文件路径
- `--image`: 图像文件路径

### 6.2 批量图像推理

```bash
# 运行批量图像推理
python -m src.inference.infer_batch --model models/onnx/model.onnx --input-dir test_images --output-dir results
```

**参数说明**:
- `--model`: 模型文件路径
- `--input-dir`: 输入图像目录
- `--output-dir`: 输出结果目录

## 7. 常见问题

### 7.1 数据采集问题

**问题**: 数据采集速度慢
**解决方案**: 调整采集参数，减少并发请求数

**问题**: 采集的图像质量差
**解决方案**: 调整 `config.py` 中的图像质量阈值

### 7.2 模型训练问题

**问题**: 模型过拟合
**解决方案**: 增加数据增强，添加正则化，减少模型复杂度

**问题**: 模型精度低
**解决方案**: 增加训练数据，调整超参数，尝试不同的模型架构

### 7.3 模型推理问题

**问题**: 推理速度慢
**解决方案**: 使用ONNX格式模型，启用GPU推理，优化输入图像尺寸

**问题**: 识别错误
**解决方案**: 增加训练数据，调整置信度阈值，检查模型是否适合目标场景

## 8. 高级用法

### 8.1 自定义模型

1. **创建自定义模型文件**: 在 `src/model_training/` 目录下创建新的模型训练脚本

2. **实现模型架构**: 继承基础模型类，实现自定义架构

3. **运行自定义训练**: 使用创建的脚本运行训练

### 8.2 扩展数据来源

1. **创建新的数据采集器**: 在 `src/data_collection/` 目录下创建新的数据采集器

2. **实现采集逻辑**: 继承基础采集器类，实现新的数据源采集逻辑

3. **集成到主流程**: 在配置文件中添加新的数据源配置

### 8.3 部署到生产环境

1. **转换模型格式**: 将模型转换为ONNX格式

2. **创建部署脚本**: 在 `scripts/deployment/` 目录下创建部署脚本

3. **配置部署环境**: 根据目标环境配置依赖和参数

4. **运行部署**: 执行部署脚本，将模型部署到生产环境

## 9. 示例工作流

### 9.1 完整工作流示例

1. **配置环境**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **准备角色列表**
   ```bash
   # 创建间谍过家家角色列表
   echo "劳埃德·福杰
约尔·福杰
阿尼亚·福杰" > auto_spider_img/characters/间谍过家家.txt
   ```

3. **运行数据采集**
   ```bash
   python -m src.data_collection.series_based_collector --mode priority
   ```

4. **数据质量控制**
   ```bash
   python -m src.data_processing.data_quality_controller --input-dir data/train
   ```

5. **数据增强**
   ```bash
   python -m src.data_processing.data_augmentation --input-dir data/train --output-dir data/train_augmented
   ```

6. **数据集分割**
   ```bash
   python -m src.data_processing.dataset_splitter --input-dir data/train_augmented --train-ratio 0.8 --val-ratio 0.2
   ```

7. **模型训练**
   ```bash
   python -m src.model_training.train_improved_model --train-dir data/train --val-dir data/val
   ```

8. **模型评估**
   ```bash
   python -m src.model_training.evaluate_model --model models/checkpoints/best_model.pth --test-dir data/test
   ```

9. **模型转换**
   ```bash
   python -m src.model_training.convert_to_onnx --model models/checkpoints/best_model.pth --output models/onnx/model.onnx
   ```

10. **模型推理**
    ```bash
    python -m src.inference.infer_single --model models/onnx/model.onnx --image test_image.jpg
    ```

## 10. 总结

本项目提供了一套完整的动漫角色检测解决方案，从数据采集到模型部署的全流程支持。通过按照本说明文档的步骤操作，您可以快速构建和部署自己的角色检测系统。

如果您在使用过程中遇到任何问题，请参考常见问题部分，或查阅项目文档的其他部分获取更多信息。