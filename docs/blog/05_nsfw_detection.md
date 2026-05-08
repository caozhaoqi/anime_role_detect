# 【技术难点】NSFW 内容过滤

> 在动漫角色数据采集中，如何识别并过滤敏感内容是保障数据合规性的关键。

---

## 🔍 问题背景

采集到的图片可能包含：

1. **色情内容**：暴露的人物图像
2. **暴力内容**：血腥、暴力场景
3. **低俗内容**：不雅姿势、表情

**核心挑战**：如何准确识别敏感内容，同时减少误判？

---

## 💡 解决方案：深度学习检测模型

### NSFW 检测器

```python
import torch
from torchvision import transforms
from PIL import Image

class NSFWDetector:
    def __init__(self, model_path='models/nsfw_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self, model_path):
        """加载预训练模型"""
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(num_classes=2)  # NSFW / Safe
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def detect(self, image_path, threshold=0.8):
        """检测图片是否为NSFW内容"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image)
                prob = torch.sigmoid(output[:, 1]).item()  # NSFW 概率
            
            return {
                'is_nsfw': prob > threshold,
                'confidence': prob,
                'threshold': threshold
            }
        except Exception as e:
            return {'is_nsfw': False, 'confidence': 0, 'error': str(e)}
```

### 皮肤比例分析（辅助检测）

```python
import cv2
import numpy as np

def analyze_skin_ratio(image_path) -> float:
    """分析图片中皮肤区域比例"""
    image = cv2.imread(image_path)
    if image is None:
        return 0.0
    
    # 转换为 YCrCb 颜色空间
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # 皮肤颜色范围（YCrCb）
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    # 创建皮肤掩码
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    skin_pixels = np.sum(mask > 0)
    total_pixels = image.shape[0] * image.shape[1]
    
    return skin_pixels / total_pixels
```

---

## 🚀 使用示例

```python
# 初始化检测器
detector = NSFWDetector()

# 检测单张图片
result = detector.detect("data/test.jpg")
print(f"NSFW检测: {result}")
# {'is_nsfw': False, 'confidence': 0.123, 'threshold': 0.8}

# 结合皮肤比例分析
skin_ratio = analyze_skin_ratio("data/test.jpg")
print(f"皮肤比例: {skin_ratio:.2%}")

# 综合判断
if result['is_nsfw'] or skin_ratio > 0.4:
    print("❌ 图片包含敏感内容，已过滤")
else:
    print("✅ 图片安全")

# 批量处理
import os
for filename in os.listdir("data/images"):
    if filename.endswith(".jpg"):
        path = os.path.join("data/images", filename)
        result = detector.detect(path)
        if result['is_nsfw']:
            os.remove(path)
            print(f"已删除: {filename}")
```

---

## ⚡ 过滤流程

```
图片输入
    ↓
NSFW模型检测
    ↓
┌─────────────┐
│ is_nsfw=True?│──Yes──→ 删除图片
└─────────────┘
    ↓ No
皮肤比例分析
    ↓
┌─────────────┐
│ 皮肤比例>40%?│──Yes──→ 删除图片
└─────────────┘
    ↓ No
保留图片
```

---

## 📝 关键要点

1. **深度学习检测**：使用预训练模型识别敏感内容
2. **皮肤比例分析**：作为辅助判断指标
3. **阈值可调**：根据需求调整检测严格程度
4. **批量处理**：支持大规模图片过滤
5. **错误处理**：对损坏图片进行容错处理

---

*下篇预告：爬虫反爬机制突破*
