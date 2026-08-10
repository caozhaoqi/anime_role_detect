#!/usr/bin/env python3
"""
使用 final_dataset 对模型进行分类测试
"""
import sys
import os

# 解码策略统一由 src/common/preprocess 提供，导入即继承，本脚本不再自行设置。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.common.preprocess  # noqa: E402

import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile
import warnings

warnings.filterwarnings('ignore')

# 配置
FINAL_DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
TRAIN_DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")
MODEL_NAME = "MobileNetV2"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_and_preprocess_image(img_path):
    """加载并预处理图片"""
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(img)
    except Exception as e:
        return None


def main():
    print("=" * 70)
    print(f"🎯 分类测试: MobileNetV2 + final_dataset")
    print("=" * 70)
    
    device = get_device()
    print(f"📱 使用设备: {device}")
    
    # 获取训练集类别
    print("\n📂 加载训练集类别...")
    train_classes = sorted([d.name for d in TRAIN_DATA_DIR.iterdir() if d.is_dir()])
    class_to_idx = {cls: i for i, cls in enumerate(train_classes)}
    idx_to_class = {i: cls for i, cls in enumerate(train_classes)}
    num_classes = len(train_classes)
    print(f"  训练集类别数: {num_classes}")
    
    # 加载模型
    print("\n� 加载模型...")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model_path = MODEL_DIR / f"{MODEL_NAME.lower()}_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"✅ 加载模型: {model_path}")
    
    # 收集final_dataset中的图片
    print("\n📂 收集 final_dataset 图片...")
    test_images = []
    for char_dir in FINAL_DATA_DIR.iterdir():
        if char_dir.is_dir():
            char_name = char_dir.name
            char_lower = char_name.lower()
            
            # 匹配训练集类别
            matched_class = None
            for cls in train_classes:
                if cls.lower() == char_lower:
                    matched_class = cls
                    break
            
            if matched_class:
                label_idx = class_to_idx[matched_class]
                for img_path in list(char_dir.glob('*.jpg')) + list(char_dir.glob('*.png')):
                    test_images.append((str(img_path), label_idx, matched_class))
    
    print(f"  测试图片数: {len(test_images)}")
    
    # 测试
    print("\n🔍 开始测试...")
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    batch_size = 32
    for i in tqdm(range(0, len(test_images), batch_size), desc="测试中"):
        batch = test_images[i:i+batch_size]
        batch_tensors = []
        batch_labels = []
        
        for img_path, label_idx, class_name in batch:
            tensor = load_and_preprocess_image(img_path)
            if tensor is not None:
                batch_tensors.append(tensor)
                batch_labels.append(label_idx)
                if class_name not in class_total:
                    class_correct[class_name] = 0
                    class_total[class_name] = 0
                class_total[class_name] += 1
        
        if not batch_tensors:
            continue
            
        batch_tensor = torch.stack(batch_tensors).to(device)
        batch_labels = torch.tensor(batch_labels).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            _, preds = torch.max(outputs, 1)
            
            for pred, label in zip(preds, batch_labels):
                total += 1
                true_class = idx_to_class[label.item()]
                if pred == label:
                    correct += 1
                    class_correct[true_class] += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    
    # 输出结果
    print("\n" + "=" * 70)
    print("📊 测试结果")
    print("=" * 70)
    print(f"模型: {MODEL_NAME}")
    print(f"测试图片数: {total}")
    print(f"正确预测: {correct}")
    print(f"**测试准确率: {accuracy:.2f}%**")
    print("=" * 70)
    
    # 各类别准确率
    class_acc = [(k, class_correct[k] / class_total[k] * 100) for k in class_total if class_total[k] > 0]
    class_acc.sort(key=lambda x: -x[1])
    
    print("\n📈 各类别准确率:")
    print("\nTOP 10 最佳:")
    for cls, acc in class_acc[:10]:
        print(f"  {cls}: {acc:.1f}% ({class_correct[cls]}/{class_total[cls]})")
    
    print("\nTOP 10 最差:")
    for cls, acc in class_acc[-10:]:
        print(f"  {cls}: {acc:.1f}% ({class_correct[cls]}/{class_total[cls]})")
    
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()
