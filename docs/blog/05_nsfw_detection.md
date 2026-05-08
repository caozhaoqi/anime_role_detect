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
            
            # 置信度分级
            level = self._get_confidence_level(prob)
            
            return {
                'is_nsfw': prob > threshold,
                'confidence': prob,
                'threshold': threshold,
                'level': level,
                'action': self._get_action(prob)
            }
        except Exception as e:
            return {'is_nsfw': False, 'confidence': 0, 'threshold': threshold, 'level': 'safe', 'action': 'accept', 'error': str(e)}
    
    def _get_confidence_level(self, prob: float) -> str:
        """根据置信度返回分级"""
        if prob < 0.5:
            return 'safe'      # 安全
        elif prob < 0.8:
            return 'suspicious'  # 疑似
        else:
            return 'dangerous'   # 危险
    
    def _get_action(self, prob: float) -> str:
        """根据置信度返回建议操作"""
        if prob < 0.5:
            return 'accept'     # 直接通过
        elif prob < 0.8:
            return 'review'     # 人工审核
        else:
            return 'reject'     # 直接拒绝
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

### 基础检测
```python
# 初始化检测器
detector = NSFWDetector()

# 检测单张图片（包含分级）
result = detector.detect("data/test.jpg")
print(f"NSFW检测结果:")
print(f"  置信度: {result['confidence']:.4f}")
print(f"  分级: {result['level']}")
print(f"  建议操作: {result['action']}")
# 输出示例:
#   置信度: 0.6500
#   分级: suspicious
#   建议操作: review
```

### 分级处理流程
```python
def process_image(image_path: str, detector: NSFWDetector):
    """根据分级处理图片"""
    result = detector.detect(image_path)
    
    # 获取皮肤比例作为辅助判断
    skin_ratio = analyze_skin_ratio(image_path)
    
    # 综合判断逻辑
    if result['level'] == 'dangerous' or (result['level'] == 'suspicious' and skin_ratio > 0.5):
        # 危险或疑似+高皮肤比例 → 直接拒绝
        os.remove(image_path)
        return {'status': 'rejected', 'reason': 'NSFW detected'}
    
    elif result['level'] == 'suspicious':
        # 疑似 → 移到待审核目录
        os.rename(image_path, f"data/review/{os.path.basename(image_path)}")
        return {'status': 'pending_review', 'reason': 'suspicious content'}
    
    else:
        # 安全 → 保留
        return {'status': 'accepted', 'reason': 'safe content'}

# 使用示例
result = process_image("data/test.jpg", detector)
print(f"处理结果: {result}")
```

### 批量处理（带统计）
```python
import os
from collections import defaultdict

def batch_process(folder_path: str, detector: NSFWDetector):
    """批量处理文件夹中的图片"""
    stats = defaultdict(int)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".webp")):
            path = os.path.join(folder_path, filename)
            result = process_image(path, detector)
            stats[result['status']] += 1
    
    print(f"批量处理完成:")
    print(f"  接受: {stats['accepted']} 张")
    print(f"  待审核: {stats['pending_review']} 张")
    print(f"  拒绝: {stats['rejected']} 张")

# 使用示例
batch_process("data/images", detector)
```

---

## ⚡ 分级过滤流程

### 置信度分级标准

| 置信度范围 | 分级 | 建议操作 | 说明 |
|-----------|------|---------|------|
| 0.0 - 0.5 | safe（安全） | accept（通过） | 图片内容安全，直接保留 |
| 0.5 - 0.8 | suspicious（疑似） | review（审核） | 需要人工审核确认 |
| 0.8 - 1.0 | dangerous（危险） | reject（拒绝） | 直接删除或屏蔽 |

### 处理流程图

```
图片输入
    ↓
NSFW模型检测
    ↓
┌───────────────────────────────────┐
│ 置信度 < 0.5?                    │
└───────────────────────────────────┘
    ↓ Yes
┌───────────────────────────────────┐
│ safe: 直接通过                    │
└───────────────────────────────────┘
    ↓ No
┌───────────────────────────────────┐
│ 置信度 < 0.8?                    │
└───────────────────────────────────┘
    ↓ Yes
┌───────────────────────────────────┐
│ suspicious: 移至待审核目录        │
│ 结合皮肤比例二次判断              │
│   - 皮肤比例 > 50% → 拒绝        │
│   - 否则 → 待审核                │
└───────────────────────────────────┘
    ↓ No
┌───────────────────────────────────┐
│ dangerous: 直接删除              │
└───────────────────────────────────┘
```

### 分级处理逻辑

```python
def classify_image(prob: float, skin_ratio: float) -> str:
    """根据NSFW概率和皮肤比例综合判断"""
    if prob < 0.5:
        return 'accept'
    elif prob < 0.8:
        # 疑似级别，结合皮肤比例判断
        if skin_ratio > 0.5:
            return 'reject'  # 高皮肤比例增加风险
        else:
            return 'review'
    else:
        return 'reject'
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
