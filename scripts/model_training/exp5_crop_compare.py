#!/usr/bin/env python3
"""
实验5: 不同裁剪方式对比
1. YOLO人脸检测裁剪
2. 中心区域裁剪
3. 保留边缘裁剪
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import gc
from ultralytics import YOLO

TRAIN_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
FINAL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model(num_classes, device):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_DIR / "mobilenetv2_best.pth", map_location=device, weights_only=True))
    return model.to(device).eval()

def center_crop(img, ratio=0.7):
    """中心裁剪，保留70%"""
    w, h = img.size
    new_w, new_h = int(w * ratio), int(h * ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))

def remove_bg_by_color(img):
    """简单背景移除：基于边缘颜色"""
    import numpy as np
    img_array = np.array(img.convert('RGB'))
    
    # 获取边缘颜色
    h, w = img_array.shape[:2]
    edge_colors = [
        img_array[0, :].mean(axis=0),      # 上
        img_array[-1, :].mean(axis=0),      # 下
        img_array[:, 0].mean(axis=0),       # 左
        img_array[:, -1].mean(axis=0),      # 右
    ]
    bg_color = np.mean(edge_colors, axis=0)
    
    # 创建mask
    distances = np.sqrt(np.sum((img_array - bg_color) ** 2, axis=2))
    threshold = np.percentile(distances, 85)
    mask = distances > threshold
    
    # 找主体区域
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return img
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # 扩展
    margin = 20
    rmin, rmax = max(0, rmin-margin), min(h, rmax+margin)
    cmin, cmax = max(0, cmin-margin), min(w, cmax+margin)
    
    return img.crop((cmin, rmin, cmax, rmax))

def test_method(model, device, dataset_dir, class_to_idx, method_fn, method_name):
    """测试单个方法"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 收集样本（每个角色2张）
    test_samples = []
    for char_dir in dataset_dir.iterdir():
        if char_dir.is_dir() and char_dir.name in class_to_idx:
            imgs = list(char_dir.glob('*.jpg'))[:2] + list(char_dir.glob('*.png'))[:2]
            for img_path in imgs[:2]:
                test_samples.append((str(img_path), class_to_idx[char_dir.name]))
    
    if not test_samples:
        return None
    
    print(f"\n{method_name}: {len(test_samples)} 样本")
    
    correct = 0
    for img_path, label_idx in test_samples:
        try:
            img = Image.open(img_path).convert('RGB')
            img = method_fn(img)
            tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if model(tensor).argmax(1).item() == label_idx:
                    correct += 1
            
            del img, tensor
            gc.collect()
        except:
            pass
    
    return correct / len(test_samples) * 100

def main():
    print("=" * 60)
    print("🔬 实验5: 裁剪方式对比")
    print("=" * 60)
    
    device = get_device()
    
    # 获取类别
    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    print(f"📊 类别数: {len(classes)}")
    
    # 加载模型
    print("\n📦 加载模型...")
    model = load_model(len(classes), device)
    
    # 加载YOLO
    print("📦 加载YOLO...")
    detector = YOLO('yolov8n.pt')
    
    def yolo_crop(img):
        results = detector.predict(img, verbose=False, conf=0.3)
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box
            pad = 0.3
            px, py = int((x2-x1)*pad), int((y2-y1)*pad)
            return img.crop((max(0,x1-px), max(0,y1-py), min(img.width,x2+px), min(img.height,y2+py)))
        return img
    
    # 测试不同方法
    methods = [
        (lambda x: x, "原图"),
        (lambda x: center_crop(x, 0.9), "中心裁剪90%"),
        (lambda x: center_crop(x, 0.7), "中心裁剪70%"),
        (lambda x: center_crop(x, 0.5), "中心裁剪50%"),
        (remove_bg_by_color, "边缘背景移除"),
        (yolo_crop, "YOLO人脸裁剪"),
    ]
    
    print("\n" + "=" * 60)
    print("📊 final_dataset 结果")
    print("=" * 60)
    
    results = {}
    for fn, name in methods:
        acc = test_method(model, device, FINAL_DIR, class_to_idx, fn, name)
        if acc:
            results[name] = acc
            print(f"  {name}: {acc:.1f}%")
    
    print("\n" + "=" * 60)
    print("📊 training_dataset 结果")
    print("=" * 60)
    
    results2 = {}
    for fn, name in methods:
        acc = test_method(model, device, TRAIN_DIR, class_to_idx, fn, name)
        if acc:
            results2[name] = acc
            print(f"  {name}: {acc:.1f}%")
    
    # 汇总
    print("\n" + "=" * 60)
    print("📋 结论")
    print("=" * 60)
    
    print("\nfinal_dataset 最佳方法:", max(results.items(), key=lambda x: x[1]))
    print("training 最佳方法:", max(results2.items(), key=lambda x: x[1]))

if __name__ == "__main__":
    main()