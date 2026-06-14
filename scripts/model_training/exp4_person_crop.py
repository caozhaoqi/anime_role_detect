#!/usr/bin/env python3
"""
实验4: 人物裁剪测试
使用YOLO检测角色主体，裁剪后测试准确率
验证背景污染假设
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import gc

# YOLO
from ultralytics import YOLO

TRAIN_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
FINAL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model(num_classes, device):
    """加载MobileNetV2"""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_DIR / "mobilenetv2_best.pth", map_location=device, weights_only=True))
    model = model.to(device).eval()
    return model

def crop_and_center(img, box, padding=0.2):
    """裁剪检测区域并居中"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    
    # 添加padding
    px, py = int(w * padding), int(h * padding)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(img.width, x2 + px)
    y2 = min(img.height, y2 + py)
    
    # 裁剪
    cropped = img.crop((x1, y1, x2, y2))
    return cropped

def detect_and_crop(img_path, model_detector, device):
    """检测人物并裁剪"""
    try:
        img = Image.open(img_path).convert('RGB')
        
        # 使用YOLO检测
        results = model_detector.predict(img, verbose=False, conf=0.3)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            # 没有检测到，返回原图
            return img
        
        # 取最大的人脸/人物区域
        boxes = results[0].boxes
        if len(boxes) == 0:
            return img
        
        # 选择最大的检测框
        areas = [(box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]) 
                 for box in boxes]
        max_idx = areas.index(max(areas))
        box = boxes[max_idx].xyxy[0].cpu().numpy()
        
        cropped = crop_and_center(img, box, padding=0.3)
        return cropped
        
    except Exception as e:
        return Image.open(img_path).convert('RGB')

def test_with_crop(model, device, dataset_dir, class_to_idx, desc="测试"):
    """使用裁剪后的图片测试"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载YOLO检测器 - 使用动漫人脸检测
    print("  加载YOLO检测模型...")
    try:
        # 尝试使用动漫人脸检测模型
        detector = YOLO('yolov8n.pt')  # 使用通用YOLO
    except:
        detector = None
    
    # 收集测试样本（每个角色2张）
    test_samples = []
    for char_dir in dataset_dir.iterdir():
        if char_dir.is_dir():
            char_name = char_dir.name
            if char_name in class_to_idx:
                imgs = list(char_dir.glob('*.jpg'))[:2] + list(char_dir.glob('*.png'))[:2]
                for img_path in imgs[:2]:
                    test_samples.append((str(img_path), class_to_idx[char_name]))
    
    if not test_samples:
        return None, None, 0
    
    print(f"  测试样本: {len(test_samples)}")
    
    # 测试原图
    print(f"  {desc} - 原图...", end=" ", flush=True)
    correct_orig = 0
    for img_path, label_idx in tqdm(test_samples, desc="原图"):
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(tensor).argmax(1).item()
                if pred == label_idx:
                    correct_orig += 1
            
            del img, tensor
            gc.collect()
        except:
            pass
    
    acc_orig = correct_orig / len(test_samples) * 100
    print(f"{acc_orig:.1f}%")
    
    # 测试裁剪图
    print(f"  {desc} - 裁剪图...", end=" ", flush=True)
    correct_crop = 0
    for img_path, label_idx in tqdm(test_samples, desc="裁剪"):
        try:
            img = detect_and_crop(img_path, detector, device) if detector else Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(tensor).argmax(1).item()
                if pred == label_idx:
                    correct_crop += 1
            
            del img, tensor
            gc.collect()
        except:
            pass
    
    acc_crop = correct_crop / len(test_samples) * 100
    print(f"{acc_crop:.1f}%")
    
    return acc_orig, acc_crop, len(test_samples)

def main():
    print("=" * 60)
    print("🔬 实验4: 人物裁剪测试")
    print("=" * 60)
    
    device = get_device()
    print(f"📱 设备: {device}")
    
    # 获取类别
    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"📊 类别数: {num_classes}")
    
    # 加载模型
    print("\n📦 加载分类模型...")
    model = load_model(num_classes, device)
    
    # 测试 final_dataset
    print("\n" + "=" * 60)
    print("📊 测试 final_dataset")
    print("=" * 60)
    acc_orig, acc_crop, n_samples = test_with_crop(model, device, FINAL_DIR, class_to_idx, "final_dataset")
    
    # 测试 training_dataset
    print("\n" + "=" * 60)
    print("📊 测试 training_dataset (测试集)")
    print("=" * 60)
    acc_orig2, acc_crop2, n_samples2 = test_with_crop(model, device, TRAIN_DIR, class_to_idx, "training")
    
    # 结果汇总
    print("\n" + "=" * 60)
    print("📊 实验结果汇总")
    print("=" * 60)
    
    print(f"\n{'数据集':<15} {'原图':>10} {'裁剪图':>10} {'提升':>10}")
    print("-" * 50)
    
    if acc_orig is not None:
        improvement = acc_crop - acc_orig
        arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
        print(f"{'final_dataset':<15} {acc_orig:>9.1f}% {acc_crop:>9.1f}% {arrow}{abs(improvement):>8.1f}%")
    
    if acc_orig2 is not None:
        improvement2 = acc_crop2 - acc_orig2
        arrow2 = "↑" if improvement2 > 0 else "↓" if improvement2 < 0 else "="
        print(f"{'training':<15} {acc_orig2:>9.1f}% {acc_crop2:>9.1f}% {arrow2}{abs(improvement2):>8.1f}%")
    
    # 结论
    print("\n" + "=" * 60)
    print("📋 结论")
    print("=" * 60)
    
    if acc_orig and acc_crop:
        improvement = acc_crop - acc_orig
        if improvement > 10:
            print(f"\n✅ 实锤背景污染！裁剪后准确率提升 {improvement:.1f}%")
            print("\n建议:")
            print("  1. 使用人物检测器预处理训练数据")
            print("  2. 训练时随机背景替换")
            print("  3. 考虑ArcFace进一步提升")
        elif improvement > 0:
            print(f"\n⚡ 裁剪有一定帮助，提升 {improvement:.1f}%")
        else:
            print(f"\n⚠️ 裁剪未带来提升，可能问题不在背景")
            print("   建议检查其他因素")

if __name__ == "__main__":
    main()