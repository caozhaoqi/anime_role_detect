#!/usr/bin/env python3
"""EfficientNet-B3 模型推理测试"""
import sys, json, torch
from pathlib import Path
from torchvision import transforms, models
from PIL import Image

MODEL_DIR = Path("models/efficientnet_b3_anime_20260616_132028")

# 加载配置
with open(MODEL_DIR / "training_results.json") as f:
    results = json.load(f)
class_names = results["class_names"]
num_classes = results["num_classes"]
print(f"📋 模型: {results['model_name']}, 类别数: {num_classes}, Top-1 Acc: {results.get('best_accuracy', 'N/A'):.4f}")

# 加载模型
model = models.efficientnet_b3(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_DIR / "model_best.pth", map_location="cpu", weights_only=True))
model.eval()
print(f"✅ 模型已加载 ({MODEL_DIR.name})")

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 推理
images = sys.argv[1:] if len(sys.argv) > 1 else []
if not images:
    # 从验证集随机取图
    test_dir = Path("data/final_dataset")
    for cls in sorted(class_names)[:5]:
        cls_dir = test_dir / cls
        if cls_dir.exists():
            imgs = list(cls_dir.glob("*"))[:3]
            images.extend(str(p) for p in imgs)
    print(f"📂 自动从验证集选取 {len(images)} 张测试图")

for img_path in images:
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top5 = torch.topk(probs, k=min(5, num_classes))

    print(f"\n{'='*60}")
    print(f"📷 {img_path}")
    print(f"{'='*60}")
    for i in range(top5.indices.size(0)):
        idx = top5.indices[i].item()
        cls = class_names[idx]
        pct = top5.values[i].item() * 100
        bar = "█" * int(pct // 10) + "░" * (10 - int(pct // 10))
        print(f"  {bar} {pct:5.1f}%  -> {cls}")