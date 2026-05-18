#!/usr/bin/env python3
"""NSFW检测工具 - 使用项目现有的PyTorch模型加载方式"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import json

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TORCH_NUM_THREADS'] = '1'

DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "combined_dataset"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "nsfw_detection_project"

LABELS = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    """加载NSFW检测模型 - 使用项目中的方式"""
    try:
        model = models.mobilenet_v2(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, len(LABELS))
        
        model_path = Path(__file__).parent.parent.parent / "models" / "nsfw_model" / "nsfw_model.pth"
        
        if model_path.exists():
            print(f"加载模型权重: {model_path}")
            model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))
        else:
            print("警告: NSFW模型权重文件不存在，使用预训练MobileNetV2")
        
        model.eval()
        print("成功加载NSFW模型")
        return model
    
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        print(f"异常堆栈: {traceback.format_exc()}")
        return None

def preprocess_image(image_path):
    """预处理图像"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def analyze_nsfw(image_path, model):
    """分析NSFW内容"""
    try:
        img_tensor = preprocess_image(image_path)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            scores = probabilities[0].numpy()
        
        details = {}
        for i, label in enumerate(LABELS):
            details[label] = float(scores[i])
        
        max_score = float(max(scores))
        max_index = np.argmax(scores)
        predicted_label = LABELS[max_index]
        
        nsfw_categories = ['porn', 'sexy', 'hentai']
        
        thresholds = {
            'porn': 0.4,
            'sexy': 0.6,
            'hentai': 0.5
        }
        
        is_nsfw = False
        if predicted_label in nsfw_categories:
            threshold = thresholds.get(predicted_label, 0.5)
            is_nsfw = max_score > threshold
        
        nsfw_score = 0
        for category in nsfw_categories:
            nsfw_score += details.get(category, 0)
        nsfw_score = min(nsfw_score, 1.0) * 100
        
        if is_nsfw or nsfw_score > 50:
            label = "NSFW"
        elif nsfw_score > 25:
            label = "Suggestive"
        else:
            label = "Safe"
        
        return str(image_path), nsfw_score, label
    
    except Exception as e:
        print(f"检测失败 {image_path}: {e}")
        return str(image_path), 0.0, "错误"

def process_dataset(dataset_path, output_path, sample_limit=None):
    """处理整个数据集"""
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(dataset_path.rglob(f'*{ext}'))
    
    if sample_limit:
        image_files = image_files[:sample_limit]
    
    print(f"找到 {len(image_files)} 张图片")
    
    model = load_model()
    if model is None:
        print("错误: 无法加载模型")
        return None
    
    nsfw_count = 0
    suggestive_count = 0
    safe_count = 0
    results = []
    
    total = len(image_files)
    for i, img_path in enumerate(image_files):
        path, score, label = analyze_nsfw(img_path, model)
        
        results.append({
            'path': path,
            'score': score,
            'label': label
        })
        
        if label == "NSFW":
            nsfw_count += 1
        elif label == "Suggestive":
            suggestive_count += 1
        else:
            safe_count += 1
        
        if (i + 1) % 50 == 0:
            print(f"已处理: {i + 1}/{total} | NSFW: {nsfw_count} | Suggestive: {suggestive_count} | Safe: {safe_count}")
    
    print(f"\n处理完成!")
    print(f"=" * 60)
    print(f"总图片数: {len(image_files)}")
    print(f"NSFW: {nsfw_count} ({nsfw_count/len(image_files)*100:.1f}%)")
    print(f"Suggestive: {suggestive_count} ({suggestive_count/len(image_files)*100:.1f}%)")
    print(f"Safe: {safe_count} ({safe_count/len(image_files)*100:.1f}%)")
    
    with open(output_path / "nsfw_detection_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n检测结果已保存到 {output_path / 'nsfw_detection_results.json'}")
    
    return {
        'total_images': len(image_files),
        'nsfw_count': nsfw_count,
        'suggestive_count': suggestive_count,
        'safe_count': safe_count
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSFW检测工具 - 使用项目现有的PyTorch方式")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="数据集路径")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="输出路径")
    parser.add_argument("--sample", type=int, default=None, help="仅处理前N张图片")
    
    args = parser.parse_args()
    
    if not Path(args.dataset).exists():
        print(f"错误: 数据集路径不存在: {args.dataset}")
        sys.exit(1)
    
    process_dataset(args.dataset, args.output, args.sample)