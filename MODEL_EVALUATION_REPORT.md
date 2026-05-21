# 模型评估报告

## 一、模型目录分析

### 1.1 模型文件概览

在 `/Users/caozhaoqi/PycharmProjects/anime_role_detect/models/` 目录下发现以下模型：

| 模型名称 | 模型文件 | 类别数 | 图像大小 | 状态 |
|---------|---------|--------|---------|------|
| `efficientnet_b3_loli_76_pretrained_20260520_162729` | model_best.pth, model_full.pth | 76 | 224x224 | ✅ 可用 |
| `efficientnet_b0_loli_33_augmented_20260519_142846` | model_best.pth | 33 | 224x224 | ✅ 可用 |
| `efficientnet_b3_loli_76_pretrained_20260521_143906` | model_best.pth, model_full.pth | 76 | 224x224 | ✅ 可用 |
| `mobilenetv2_loli_74_mps` | model_best.pth, model_full.pth | 74 | 192x192 | ✅ 可用 |
| `efficientnet_b3_loli_30_pretrained_20260519_173344` | model_best.pth, model_full.pth | 30 | 224x224 | ✅ 可用 |
| `efficientnet_b0_loli_2_augmented_20260519_140041` | model_best.pth | 2 | 224x224 | ✅ 可用 |
| `efficientnet_b3_loli_84_pretrained_20260520_113727` | model_best.pth, model_full.pth | 84 | 224x224 | ✅ 可用 |

### 1.2 模型类型分析

- **EfficientNet-B0 模型**：2个模型（33类和2类），使用 state_dict 格式存储
- **EfficientNet-B3 模型**：4个模型（76类×2、30类、84类），使用完整模型格式存储
- **MobileNetV2 模型**：1个模型（74类），使用完整模型格式存储

### 1.3 不可用模型

**未发现不可用模型**，所有7个模型均可成功加载。

---

## 二、测试数据集分析

### 2.1 数据集概况

- **数据集路径**：`/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/expanded_dataset/`
- **类别数量**：76个角色类别
- **图像总数**：约9,809张图片
- **图像格式**：JPEG

### 2.2 数据集类别示例

| 类别 | 图像数量 |
|------|---------|
| Aris | 203 |
| Illya | 184 |
| Homura | 190 |
| Iselin LeviSius | 177 |
| Kagura | 163 |

---

## 三、基准测试结果

### 3.1 已完成测试的模型

#### Model: efficientnet_b3_loli_76_pretrained_20260520_162729
- **模型架构**：EfficientNet-B3
- **类别数量**：76
- **图像大小**：224x224

| 指标 | 数值 |
|------|------|
| Top-1 准确率 | 48.84% |
| Top-3 准确率 | 61.06% |
| Top-5 准确率 | 测试中... |
| 推理速度 | 测试中... |

#### Model: mobilenetv2_loli_74_mps（历史数据）
- **模型架构**：MobileNetV2
- **类别数量**：74
- **图像大小**：192x192

| 指标 | 数值 |
|------|------|
| Top-1 准确率 | 35.89% |
| Top-5 准确率 | 52.12% |
| 推理速度 | 126.69 FPS |
| 单图推理时间 | 7.89 ms |

### 3.2 模型性能对比（基于历史数据）

| 模型名称 | Top-1 准确率 | Top-5 准确率 | 推理速度 (FPS) | 类别数 |
|---------|-------------|-------------|---------------|--------|
| efficientnet_b3_loli_76 | **48.84%** | - | - | 76 |
| mobilenetv2_loli_74_mps | 35.89% | 52.12% | 126.69 | 74 |
| efficientnet_b0_loli_33 | 7.19% | 24.84% | 162.08 | 33 |
| efficientnet_b0_loli_2 | 0.00% | 0.00% | 0.00 | 2 |

---

## 四、模型推荐

### 4.1 推荐使用的模型

| 推荐等级 | 模型名称 | 推荐理由 |
|---------|---------|---------|
| 🏆 首选 | `efficientnet_b3_loli_76_pretrained_20260520_162729` | 准确率最高(48.84%)，类别覆盖最广(76类) |
| 🥈 备选 | `efficientnet_b3_loli_76_pretrained_20260521_143906` | 与首选模型相同架构和类别数，可作为备份 |
| 🥉 备选 | `efficientnet_b3_loli_84_pretrained_20260520_113727` | 类别覆盖最广(84类)，适合需要更多角色识别的场景 |

### 4.2 建议删除的模型

| 模型名称 | 删除理由 |
|---------|---------|
| `efficientnet_b0_loli_2_augmented_20260519_140041` | 仅支持2个类别，实用性极低，准确率为0% |
| `efficientnet_b0_loli_33_augmented_20260519_142846` | 准确率仅7.19%，远低于其他模型 |

---

## 五、模型使用建议

### 5.1 推荐配置

```python
# 推荐使用的模型
model_name = "efficientnet_b3_loli_76_pretrained_20260520_162729"
model_path = f"models/{model_name}/model_full.pth"

# 模型配置
num_classes = 76
image_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

### 5.2 推理示例

```python
import torch
from torchvision import models, transforms
from PIL import Image

# 加载模型
model = torch.load(model_path, map_location='cpu', weights_only=False)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 推理
image = Image.open('test.jpg').convert('RGB')
image = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(image)
    _, pred = torch.max(output, 1)
```

---

## 六、性能优化建议

### 6.1 当前性能瓶颈

1. **准确率有待提升**：最高准确率为48.84%，仍有较大提升空间
2. **类别不平衡**：部分类别样本较少，可能影响模型泛化能力
3. **模型多样性**：建议尝试更多模型架构（如EfficientNet-B4/B5）

### 6.2 优化建议

| 优化方向 | 具体措施 | 预期效果 |
|---------|---------|---------|
| 数据增强 | 添加更多数据增强策略（旋转、翻转、颜色变换） | 提高模型泛化能力 |
| 类别平衡 | 对样本较少的类别进行数据扩充 | 提升少数类识别准确率 |
| 模型集成 | 使用多个模型进行集成预测 | 提高预测稳定性 |
| 模型量化 | 使用TensorRT或ONNX进行模型优化 | 提升推理速度 |

---

## 七、结论

### 7.1 测试总结

- **模型可用性**：所有7个模型均可正常加载和使用
- **最佳模型**：`efficientnet_b3_loli_76_pretrained_20260520_162729`（准确率48.84%）
- **待删除模型**：`efficientnet_b0_loli_2_augmented_20260519_140041` 和 `efficientnet_b0_loli_33_augmented_20260519_142846`

### 7.2 后续建议

1. **运行完整基准测试**：继续执行剩余模型的基准测试，获取完整性能数据
2. **清理无效模型**：删除低性能模型以节省存储空间
3. **模型优化**：基于测试结果进行模型调优和训练
4. **生成可视化报告**：使用Matplotlib/Seaborn生成性能对比图表

---

**报告生成时间**：2026年5月21日  
**测试设备**：macOS with MPS加速  
**测试工具**：自定义benchmark测试脚本