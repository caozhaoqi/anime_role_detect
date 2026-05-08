# 【技术难点】图像预处理与特征提取

> 图像预处理是影响模型识别准确率的关键步骤，本文介绍我们的预处理管道设计。

---

## 🔍 问题背景

原始图像数据存在以下问题：

1. **尺寸不一致**：不同图片大小差异大
2. **颜色偏差**：光照、对比度差异
3. **格式多样**：JPG、PNG、WebP等多种格式
4. **噪声干扰**：压缩失真、水印等

---

## 💡 解决方案：预处理管道

### 多角色检测流程

系统采用**两阶段检测**方案：先使用目标检测模型定位角色，再进行分类识别。

```python
from ultralytics import YOLO
from PIL import Image
import numpy as np

class MultiRoleDetector:
    def __init__(self):
        # 加载 YOLOv8 目标检测模型（预训练在动漫角色数据集上）
        self.detector = YOLO('models/yolov8n-anime.pt')
        self.processor = ImageProcessor(input_size=224)
    
    def detect_and_crop(self, image_path: str, confidence: float = 0.5):
        """检测图片中的角色并裁剪出每个角色"""
        # 读取图片
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # 使用 YOLO 检测角色
        results = self.detector(image_np, conf=confidence)
        
        cropped_images = []
        for result in results:
            for box in result.boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # 裁剪角色区域
                cropped = image.crop((x1, y1, x2, y2))
                cropped_images.append({
                    'image': cropped,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': box.conf.cpu().numpy()[0]
                })
        
        return cropped_images
    
    def classify_roles(self, image_path: str, model) -> list:
        """检测并分类图片中的所有角色"""
        # 1. 检测并裁剪角色
        cropped_list = self.detect_and_crop(image_path)
        
        results = []
        for item in cropped_list:
            # 2. 预处理裁剪后的角色图片
            tensor = self.processor.transform(item['image']).unsqueeze(0)
            
            # 3. 分类识别
            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.softmax(output, dim=1)
                top1 = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, top1].item()
            
            results.append({
                'bbox': item['bbox'],
                'prediction': top1,
                'confidence': confidence,
                'detection_confidence': item['confidence']
            })
        
        return results
```

### 标准化处理

```python
from torchvision import transforms
from PIL import Image

class ImageProcessor:
    def __init__(self, input_size: int = 224):
        self.transform = transforms.Compose([
            # 1. 尺寸调整
            transforms.Resize((input_size, input_size)),
            
            # 2. 随机裁剪（训练阶段）
            # transforms.RandomCrop(input_size),
            
            # 3. 随机水平翻转（训练阶段）
            # transforms.RandomHorizontalFlip(),
            
            # 4. 转换为张量
            transforms.ToTensor(),
            
            # 5. 归一化（使用 ImageNet 均值和标准差）
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess(self, image_path: str) -> torch.Tensor:
        """预处理单张图片"""
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)
        return tensor
```

### 特征提取

```python
import torch
import numpy as np

def extract_features(image_tensor: torch.Tensor, model) -> np.ndarray:
    """提取特征向量"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image_tensor.to(device)
    
    # 获取特征提取层输出
    with torch.no_grad():
        # EfficientNet 专用方法
        if hasattr(model, 'extract_features'):
            features = model.extract_features(image_tensor)
        else:
            # 其他模型通过移除分类层获取特征
            features = model.features(image_tensor)
        
        # 全局平均池化
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
    
    return features.cpu().numpy()
```

---

## 🚀 使用示例

### 单角色分类
```python
# 初始化处理器
processor = ImageProcessor(input_size=224)

# 预处理图片
tensor = processor.preprocess("data/test.jpg")
print(f"张量形状: {tensor.shape}")  # torch.Size([1, 3, 224, 224])

# 加载模型
model = ModelManager().load_model('efficientnet-b3')

# 提取特征
features = extract_features(tensor, model)
print(f"特征维度: {features.shape}")  # (1, 1536)

# 进行分类
with torch.no_grad():
    output = model(tensor)
    probabilities = torch.softmax(output, dim=1)
    top5 = torch.topk(probabilities, 5)
    
print(f"Top 5 预测: {top5.indices[0].tolist()}")
```

### 多角色检测
```python
# 初始化多角色检测器
detector = MultiRoleDetector()

# 加载分类模型
model = ModelManager().load_model('efficientnet-b3')

# 检测并分类图片中的所有角色
results = detector.classify_roles("data/group_photo.jpg", model)

# 输出结果
print(f"检测到 {len(results)} 个角色:")
for i, result in enumerate(results):
    print(f"角色 {i+1}:")
    print(f"  位置: {result['bbox']}")
    print(f"  预测: {result['prediction']}")
    print(f"  分类置信度: {result['confidence']:.2f}")
    print(f"  检测置信度: {result['detection_confidence']:.2f}")
```

### 多角色检测流程图
```
输入图片
    ↓
YOLOv8 目标检测
    ↓
┌─────────────────────────────┐
│ 检测到的每个角色边界框      │
│ (x1, y1, x2, y2)           │
└─────────────────────────────┘
    ↓
逐个裁剪角色区域
    ↓
┌─────────────────────────────┐
│ 裁剪后的角色图片            │
│ (224x224)                  │
└─────────────────────────────┘
    ↓
EfficientNet 分类
    ↓
输出每个角色的识别结果
```

---

## ⚡ 数据增强策略（训练阶段）

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 📝 关键要点

1. **尺寸统一**：将所有图片调整为相同大小
2. **归一化**：使用 ImageNet 统计量进行标准化
3. **数据增强**：训练阶段增加随机性，提高泛化能力
4. **特征提取**：利用模型中间层输出作为特征向量
5. **设备感知**：自动选择 GPU/CPU 进行处理

---

*下篇预告：NSFW 内容过滤*
