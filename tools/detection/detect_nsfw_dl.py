#!/usr/bin/env python3
"""NSFW检测工具 - 使用PyTorch深度学习方法"""

import os
import platform

# 检查是否是macOS环境
is_macos = platform.system() == 'Darwin'

# 如果是macOS环境，直接回退到基于规则的检测方法
if is_macos:
    print("检测到 macOS 环境，使用基于规则的NSFW检测方法")
    # 修改当前脚本路径，调用基于规则的检测
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rule_based_script = os.path.join(script_dir, 'detect_nsfw.py')
    
    # 读取基于规则的检测代码并执行
    with open(rule_based_script, 'r', encoding='utf-8') as f:
        code = f.read()
        exec(code)
    sys.exit(0)

# 以下是非macOS环境的深度学习检测代码
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

import sys
from pathlib import Path
import argparse
import json
from PIL import Image

DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "combined_dataset"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "nsfw_detection_dl"

def load_model():
    """加载预训练的NSFW检测模型"""
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        
        model_name = "Falconsai/nsfw_image_detection"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        
        model.eval()
        print(f"加载模型成功: {model_name}")
        return processor, model
    
    except ImportError:
        print("警告: transformers库未安装，请安装: pip install transformers torch")
        return None, None
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        print(f"异常堆栈: {traceback.format_exc()}")
        return None, None

def analyze_nsfw(image_path, processor, model):
    """使用深度学习模型检测NSFW"""
    try:
        image = Image.open(str(image_path)).convert('RGB')
        
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        label = model.config.id2label[predicted_class_idx]
        nsfw_score = probabilities[0][model.config.label2id.get('nsfw', 0)].item() * 100
        
        if label.lower() == 'nsfw' or nsfw_score > 50:
            final_label = "NSFW"
        elif nsfw_score > 25:
            final_label = "Suggestive"
        else:
            final_label = "Safe"
        
        return str(image_path), nsfw_score, final_label
    
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
    
    processor, model = load_model()
    if processor is None or model is None:
        print("错误: 无法加载深度学习模型")
        return None
    
    nsfw_count = 0
    suggestive_count = 0
    safe_count = 0
    results = []
    
    total = len(image_files)
    for i, img_path in enumerate(image_files):
        path, score, label = analyze_nsfw(img_path, processor, model)
        
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
    
    summary = {
        'total_images': len(image_files),
        'nsfw_count': nsfw_count,
        'suggestive_count': suggestive_count,
        'safe_count': safe_count,
        'detection_method': 'Deep Learning (Transformers)',
        'model': 'Falconsai/nsfw_image_detection'
    }
    
    with open(output_path / "detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n检测结果已保存到 {output_path / 'nsfw_detection_results.json'}")
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSFW检测工具 - 使用深度学习方法")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="数据集路径")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="输出路径")
    parser.add_argument("--sample", type=int, default=None, help="仅处理前N张图片")
    
    args = parser.parse_args()
    
    if not Path(args.dataset).exists():
        print(f"错误: 数据集路径不存在: {args.dataset}")
        sys.exit(1)
    
    process_dataset(args.dataset, args.output, args.sample)