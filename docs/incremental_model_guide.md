# 增量训练模型文档

## 一、模型概述

本项目实现了一个基于增量学习的动漫角色识别模型，支持在已有模型基础上继续训练新数据，保留原有知识并适应新数据。

### 1.1 模型配置

| 参数 | 值 |
|------|-----|
| 模型架构 | MobileNetV2 |
| 输入尺寸 | 224x224 |
| 训练轮数 | 48 epochs (早停) |
| 批量大小 | 32 |
| 学习率 | 0.001 |
| 验证准确率 | 88.10% |
| 测试准确率 | 94.00% |
| 推理速度 | 379.34 FPS |

### 1.2 支持的角色

| 角色名称 | 训练样本数 | 测试精确率 | 测试召回率 |
|---------|-----------|-----------|-----------|
| 阿罗娜 | 253 | 96.18% | 99.60% |
| 日奈 | 55 | 86.44% | 92.73% |
| 普拉娜 | 47 | 97.50% | 82.98% |
| 千夏 | 25 | 89.47% | 68.00% |
| 亚子 | 28 | 89.29% | 89.29% |
| 枫香 | 6 | 100.00% | 83.33% |
| 伊织 | 3 | 75.00% | 100.00% |

---

## 二、训练模型

### 2.1 基础训练

```bash
python3 arona/training/train_incremental.py \
    --model-type mobilenet_v2 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --data-dir data/downloaded_images \
    --output-dir models/incremental
```

### 2.2 参数说明

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `--model-type` | 模型类型 (mobilenet_v2, efficientnet_b0, resnet50) | mobilenet_v2 |
| `--epochs` | 训练轮数 | 50 |
| `--batch-size` | 批量大小 | 32 |
| `--lr` | 学习率 | 0.001 |
| `--data-dir` | 数据目录 | data/downloaded_images |
| `--output-dir` | 输出目录 | models/incremental |
| `--checkpoint` | 继续训练的检查点路径 | None |
| `--weight-decay` | 权重衰减 | 0.0001 |
| `--dropout` | Dropout率 | 0.2 |
| `--label-smoothing` | 标签平滑 | 0.1 |
| `--use-mixup` | 使用Mixup数据增强 | False |

### 2.3 增量训练

在已有模型基础上继续训练新角色：

```bash
python3 arona/training/train_incremental.py \
    --model-type mobilenet_v2 \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.0001 \
    --checkpoint models/incremental/model_best.pth \
    --data-dir data/new_characters \
    --output-dir models/incremental_v2
```

**增量训练注意事项**：
1. 学习率应设置为较小的值（如0.0001）
2. 可以冻结部分层，只训练新添加的分类器
3. 确保新数据的类别映射正确

---

## 三、评估模型

### 3.1 完整评估

```bash
python3 arona/evaluation/test_classification_model.py \
    --model-path models/incremental/model_best.pth \
    --model-type mobilenet_v2 \
    --data-dir data/downloaded_images \
    --output-dir models/evaluation
```

### 3.2 基准测试

```bash
python3 tests/benchmark/benchmark_incremental_model.py \
    --model-path models/incremental/model_best.pth \
    --model-type mobilenet_v2 \
    --data-dir data/downloaded_images \
    --output-dir models/benchmark
```

### 3.3 单图测试

```bash
python3 arona/evaluation/test_classification_model.py \
    --model-path models/incremental/model_best.pth \
    --model-type mobilenet_v2 \
    --test-image path/to/image.jpg
```

---

## 四、模型性能

### 4.1 整体指标

| 指标 | 值 |
|------|-----|
| 准确率 (Accuracy) | 94.00% |
| 精确率 (Precision) | 90.55% |
| 召回率 (Recall) | 87.99% |
| F1分数 (F1-Score) | 88.60% |
| 推理速度 (FPS) | 379.34 |

### 4.2 性能分析

1. **阿罗娜表现最佳**：作为训练数据最多的类别（253张），精确率和召回率均达到96%以上
2. **样本量影响**：样本较少的类别（如伊织、枫香）性能波动较大
3. **推理速度**：379 FPS 的推理速度满足实时应用需求

### 4.3 混淆矩阵分析

主要混淆情况：
- 日奈 → 阿罗娜：3张
- 千夏 → 阿罗娜：5张
- 普拉娜 → 日奈：4张
- 普拉娜 → 阿罗娜：4张

---

## 五、数据准备

### 5.1 数据目录结构

```
data/downloaded_images/
├── 阿罗娜/
│   ├── 阿罗娜_1.jpg
│   ├── 阿罗娜_2.jpg
│   └── ...
├── 普拉娜/
│   ├── 普拉娜_1.jpg
│   └── ...
└── 其他角色/
```

### 5.2 数据清洗

运行数据清洗脚本：

```bash
# 删除SVG格式图片
find data/downloaded_images -name "*.svg" -type f -delete

# 检查数据统计
for dir in data/downloaded_images/*; do
    echo "$(basename "$dir"): $(find "$dir" -type f | wc -l) 张"
done
```

### 5.3 从API采集数据

```bash
# 启动API服务
cd spider_image_system
python3 src/run/sis_main_process.py

# 采集角色数据
python3 tests/data/batch_spider_roles.py
```

---

## 六、模型部署

### 6.1 导出模型

```python
import torch

# 加载模型
checkpoint = torch.load('models/incremental/model_best.pth')
model = checkpoint['model_state_dict']

# 导出为ONNX格式
torch.onnx.export(model, dummy_input, 'model.onnx')
```

### 6.2 使用模型进行推理

```python
import torch
from torchvision import transforms
from PIL import Image

# 加载模型
model = torch.load('models/incremental/model_best.pth')
model.eval()

# 预处理图像
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('test.jpg')
input_tensor = transform(image).unsqueeze(0)

# 推理
with torch.no_grad():
    output = model(input_tensor)
    prediction = output.argmax(dim=1)
```

---

## 七、常见问题

### 7.1 训练问题

1. **内存不足**：减小batch_size
2. **过拟合**：增加dropout，使用数据增强
3. **收敛慢**：调整学习率，使用学习率调度器

### 7.2 评估问题

1. **模型加载失败**：检查模型路径和类型是否匹配
2. **类别不匹配**：确保训练和评估使用相同的类别映射

### 7.3 数据问题

1. **图片格式错误**：使用PIL/OpenCV转换图片格式
2. **图片损坏**：检查并删除损坏的图片文件

---

## 八、模型文件说明

### 8.1 文件列表

| 文件名 | 说明 |
|--------|------|
| `model_best.pth` | 最佳验证准确率模型 |
| `training_results.json` | 训练结果记录 |
| `benchmark_results.json` | 基准测试结果 |

### 8.2 检查点内容

```python
{
    'epoch': 48,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_to_idx': {'阿罗娜': 6, '普拉娜': 4, ...},
    'val_acc': 0.8810,
    'train_acc': 0.9850
}
```

---

## 九、后续扩展

### 9.1 数据扩展

1. 继续采集更多角色数据
2. 使用API接口批量采集
3. 数据增强提升模型泛化能力

### 9.2 模型优化

1. 尝试其他模型架构（EfficientNet、ResNet）
2. 调整超参数
3. 使用更高级的数据增强技术

### 9.3 功能扩展

1. 添加属性识别功能
2. 支持多标签分类
3. 模型量化优化推理速度
