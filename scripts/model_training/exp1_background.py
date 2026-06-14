#!/usr/bin/env python3
"""实验1: 背景遮挡实验 - 极简版"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image, ImageFilter
import gc

TRAIN_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    print("🧪 实验1: 背景遮挡实验")
    device = get_device()
    
    # 获取类别
    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    
    # 加载模型
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_DIR / "mobilenetv2_best.pth", map_location=device, weights_only=True))
    model = model.to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 收集样本（每类1张，最多10个）
    test_samples = []
    for char_dir in TRAIN_DIR.iterdir():
        if char_dir.is_dir() and len(test_samples) < 10:
            imgs = list(char_dir.glob('*.jpg'))[:1] + list(char_dir.glob('*.png'))[:1]
            if imgs:
                test_samples.append((imgs[0], class_to_idx.get(char_dir.name, -1)))
    
    print(f"测试样本: {len(test_samples)}")
    
    def test_with_processing(processor_fn, desc):
        correct = 0
        for img_path, label_idx in test_samples:
            try:
                img = Image.open(img_path).convert('RGB')
                img = processor_fn(img)
                tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(tensor).argmax(1).item()
                    if pred == label_idx:
                        correct += 1
                
                del img, tensor
                gc.collect()
            except:
                pass
        return correct / len(test_samples) * 100
    
    # 测试
    print("原始图片...", end=" ", flush=True)
    orig_acc = test_with_processing(lambda x: x, "原始")
    print(f"{orig_acc:.1f}%")
    
    print("中心模糊...", end=" ", flush=True)
    blur_acc = test_with_processing(lambda x: x.filter(ImageFilter.GaussianBlur(30)), "模糊")
    print(f"{blur_acc:.1f}%")
    
    print("纯灰背景...", end=" ", flush=True)
    gray_fn = lambda x: Image.new('RGB', x.size, (128, 128, 128))
    gray_acc = test_with_processing(gray_fn, "灰")
    print(f"{gray_acc:.1f}%")
    
    print(f"\n📊 结果:")
    print(f"  原始: {orig_acc:.1f}%")
    print(f"  模糊: {blur_acc:.1f}%")
    print(f"  灰背景: {gray_acc:.1f}%")
    
    if orig_acc - gray_acc > 10:
        print(f"  ⚠️ 模型严重依赖背景！下降 {orig_acc-gray_acc:.1f}%")
    else:
        print(f"  ✅ 模型对背景依赖较低")

if __name__ == "__main__":
    main()