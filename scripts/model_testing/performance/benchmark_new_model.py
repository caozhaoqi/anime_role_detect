#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新模型基准测试脚本
对刚训练完成的 mobilenetv2_loli_74_mps 模型进行性能评估
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from datetime import datetime
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("benchmark")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("benchmark")

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 配置
BATCH_SIZE = 8
IMAGE_SIZE = 224
NUM_WORKERS = 0
DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
MODEL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/models/mobilenetv2_loli_74_mps'


class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, class_filter=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_filter = class_filter
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # 获取所有类别
        all_class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # 如果有类别过滤，只保留指定的类别
        if class_filter:
            class_names = sorted([name for name in all_class_names if name in class_filter])
        else:
            class_names = all_class_names
        
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

        logger.info(f"数据集: {len(self.samples)} 样本, {len(self.class_to_idx)} 类别")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"加载图像失败 {img_path}: {e}")
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
            if self.transform:
                image = self.transform(image)
            return image, label


def load_trained_model(model_dir):
    """加载训练好的模型"""
    model_path = os.path.join(model_dir, 'model_best.pth')
    class_map_path = os.path.join(model_dir, 'class_to_idx.json')
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None, None
    
    if not os.path.exists(class_map_path):
        logger.error(f"类别映射文件不存在: {class_map_path}")
        return None, None
    
    # 加载类别映射
    with open(class_map_path, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 获取类别数
    num_classes = len(class_to_idx)
    
    # 创建模型结构
    model = models.mobilenet_v2(pretrained=False)
    feature_dim = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(feature_dim, num_classes)
    )
    
    # 加载权重
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    logger.info(f"成功加载模型: {model_dir}")
    return model, class_to_idx


def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []
    correct = 0
    total = 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  已处理 {batch_idx + 1}/{len(dataloader)} 批次")

    accuracy = 100. * correct / total
    avg_inference_time = np.mean(inference_times) * 1000
    std_inference_time = np.std(inference_times) * 1000
    throughput = 1.0 / (avg_inference_time / 1000)

    return {
        'accuracy': accuracy,
        'avg_inference_time_ms': avg_inference_time,
        'std_inference_time_ms': std_inference_time,
        'throughput_samples_per_sec': throughput,
        'class_accuracy': {int(k): 100. * class_correct[k] / class_total[k] if class_total[k] > 0 else 0
                          for k in class_total.keys()},
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs
    }


def print_benchmark_report(result, class_to_idx, idx_to_class, model_name):
    """打印基准测试报告"""
    print("\n" + "=" * 70)
    print(" " * 20 + f"{model_name} 模型基准测试报告")
    print("=" * 70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试数据集: {DATA_DIR}")
    print(f"数据集大小: {len(result['all_labels'])} 样本")
    print(f"类别数: {len(class_to_idx)}")
    print("=" * 70)

    print("\n" + "-" * 70)
    print(" " * 25 + "1. 总体性能")
    print("-" * 70)
    print(f"准确率: {result['accuracy']:.2f}%")
    print(f"平均推理时间: {result['avg_inference_time_ms']:.2f} ± {result['std_inference_time_ms']:.2f} ms")
    print(f"吞吐量: {result['throughput_samples_per_sec']:.2f} samples/s")
    print("-" * 70)

    print("\n" + "-" * 70)
    print(" " * 25 + "2. 各类别准确率（前15个）")
    print("-" * 70)
    print(f"{'类别名称':<25} {'准确率':<10} {'样本数':<10}")
    print("-" * 70)
    
    sorted_classes = sorted(class_to_idx.keys())[:15]
    for class_name in sorted_classes:
        class_idx = class_to_idx[class_name]
        acc = result['class_accuracy'].get(class_idx, 0)
        count = sum(1 for l in result['all_labels'] if l == class_idx)
        print(f"{class_name[:23]:<25} {acc:.2f}%{'':<5} {count:<10}")
    
    print("-" * 70)
    print(f"共 {len(class_to_idx)} 个类别，显示前15个...")

    print("\n" + "=" * 70)
    print(" " * 25 + "测试报告结束")
    print("=" * 70 + "\n")


def main():
    logger.info("=" * 60)
    logger.info("开始新模型基准测试")
    logger.info("=" * 60)

    # 使用CPU运行（更稳定）
    device = torch.device('cpu')
    logger.info(f"使用设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载模型和类别映射
    logger.info(f"加载模型: {MODEL_DIR}")
    model, model_class_to_idx = load_trained_model(MODEL_DIR)
    
    if model is None:
        logger.error("模型加载失败")
        return
    
    logger.info(f"模型类别数: {len(model_class_to_idx)}")

    # 加载测试数据（只使用模型中有的类别）
    logger.info(f"加载测试数据: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        logger.error(f"数据目录不存在: {DATA_DIR}")
        return
    
    dataset = SimpleImageDataset(DATA_DIR, transform=transform, class_filter=set(model_class_to_idx.keys()))
    class_to_idx = dataset.class_to_idx
    idx_to_class = dataset.idx_to_class
    
    if len(class_to_idx) == 0:
        logger.error("没有匹配的类别")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    logger.info(f"测试数据集大小: {len(dataset)}")
    logger.info(f"类别数: {len(class_to_idx)}")

    # 评估模型
    logger.info("评估模型性能...")
    result = evaluate_model(model, dataloader, device)

    logger.info(f"  准确率: {result['accuracy']:.2f}%")
    logger.info(f"  推理时间: {result['avg_inference_time_ms']:.2f}±{result['std_inference_time_ms']:.2f} ms")
    logger.info(f"  吞吐量: {result['throughput_samples_per_sec']:.2f} samples/s")

    # 打印报告
    print_benchmark_report(result, class_to_idx, idx_to_class, 'mobilenetv2_loli_74_mps')

    # 保存报告
    report_path = os.path.join(MODEL_DIR, 'benchmark_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'dataset_size': len(dataset),
            'num_classes': len(class_to_idx),
            'class_mapping': class_to_idx,
            'accuracy': result['accuracy'],
            'avg_inference_time_ms': result['avg_inference_time_ms'],
            'std_inference_time_ms': result['std_inference_time_ms'],
            'throughput_samples_per_sec': result['throughput_samples_per_sec'],
            'class_accuracy': result['class_accuracy']
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"基准测试报告已保存到: {report_path}")

    logger.info("=" * 60)
    logger.info("基准测试完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()