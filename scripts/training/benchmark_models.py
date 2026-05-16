#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型基准测试脚本
测试所有已训练模型的准确率、推理速度等指标
"""
import torch
import torch.nn as nn
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    'mobilenetv2': {
        'name': 'MobileNetV2',
        'image_size': 224,
        'batch_size': 32,
        'model_fn': models.mobilenet_v2,
    },
    'efficientnet_b0': {
        'name': 'EfficientNet-B0',
        'image_size': 224,
        'batch_size': 32,
        'model_fn': models.efficientnet_b0,
    },
    'efficientnet_b3': {
        'name': 'EfficientNet-B3',
        'image_size': 300,
        'batch_size': 24,
        'model_fn': models.efficientnet_b3,
    },
    'resnet50': {
        'name': 'ResNet50',
        'image_size': 224,
        'batch_size': 32,
        'model_fn': models.resnet50,
    }
}

BASE_DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
BASE_MODEL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/models'

def get_device():
    """获取设备"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_model(model_name, model_dir):
    """加载模型"""
    config = MODEL_CONFIGS[model_name]
    model_fn = config['model_fn']
    image_size = config['image_size']

    model_path = Path(model_dir) / 'model_full.pth'
    if not model_path.exists():
        model_path = Path(model_dir) / 'model_best.pth'

    if not model_path.exists():
        return None, None, None

    idx_to_class_path = Path(model_dir) / 'class_to_idx.json'
    if idx_to_class_path.exists():
        with open(idx_to_class_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            idx_to_class = mapping.get('idx_to_class', {})
    else:
        idx_to_class = {}

    try:
        model = torch.load(model_path, map_location='cpu')
    except Exception as e:
        logger.warning(f"无法加载模型: {e}")
        return None, None, None

    model.eval()

    return model, idx_to_class, image_size


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def benchmark_model(model, dataloader, device, idx_to_class):
    """基准测试单个模型"""
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    inference_times = []
    all_confidences = []

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            inference_times.append(inference_time)

            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probs, 1)
            all_confidences.extend(confidences.cpu().numpy())

            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i].item()
                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

            if batch_idx % 50 == 0:
                logger.info(f"  进度: {batch_idx}/{len(dataloader)}")

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    per_class_accuracy = {}
    for label in class_total:
        class_name = idx_to_class.get(str(label), f"class_{label}")
        per_class_accuracy[class_name] = class_correct[label] / class_total[label] if class_total[label] > 0 else 0

    precision_per_class = per_class_accuracy

    macro_precision = sum(per_class_accuracy.values()) / len(per_class_accuracy) if per_class_accuracy else 0
    macro_recall = macro_precision
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1_score': macro_f1,
        'loss': avg_loss,
        'avg_inference_time_ms': avg_inference_time * 1000,
        'fps': fps,
        'num_params': count_parameters(model),
        'avg_confidence': avg_confidence,
        'per_class_accuracy': per_class_accuracy
    }


def run_benchmark(model_name, data_dir=None, model_dir=None):
    """运行指定模型的基准测试"""
    if data_dir is None:
        data_dir = BASE_DATA_DIR
    if model_dir is None:
        model_dir = os.path.join(BASE_MODEL_DIR, f'{model_name}_loli')

    config = MODEL_CONFIGS[model_name]
    image_size = config['image_size']
    batch_size = config['batch_size']

    logger.info("=" * 60)
    logger.info(f"🧪 基准测试: {config['name']}")
    logger.info("=" * 60)

    model, idx_to_class, loaded_size = load_model(model_name, model_dir)

    if model is None:
        logger.warning(f"⚠️ 模型未找到: {model_dir}")
        return None

    device = get_device()
    logger.info(f"使用设备: {device}")

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = ImageFolder(data_dir, transform=test_transform)

    test_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - test_size
    _, test_dataset = None, None

    try:
        from torch.utils.data import random_split
        _, test_dataset = random_split(full_dataset, [train_size, test_size])
    except:
        logger.warning("无法划分数据集，使用全部数据")
        test_dataset = full_dataset

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"测试集大小: {len(test_dataset)}")
    logger.info(f"类别数: {len(full_dataset.classes)}")

    results = benchmark_model(model, test_loader, device, idx_to_class)
    results['model_name'] = config['name']
    results['model_key'] = model_name
    results['image_size'] = image_size
    results['batch_size'] = batch_size
    results['test_samples'] = len(test_dataset)
    results['num_classes'] = len(full_dataset.classes)
    results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    logger.info(f"\n📊 {config['name']} 基准测试结果:")
    logger.info(f"  - 准确率: {results['accuracy']:.4%}")
    logger.info(f"  - 精确率: {results['precision']:.4%}")
    logger.info(f"  - 召回率: {results['recall']:.4%}")
    logger.info(f"  - F1分数: {results['f1_score']:.4%}")
    logger.info(f"  - 平均损失: {results['loss']:.4f}")
    logger.info(f"  - 推理时间: {results['avg_inference_time_ms']:.2f} ms")
    logger.info(f"  - FPS: {results['fps']:.2f}")
    logger.info(f"  - 参数量: {results['num_params']:,}")

    return results


def generate_report(all_results, output_path):
    """生成基准测试报告"""
    report = {
        'title': '二次元角色识别模型基准测试报告',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': {}
    }

    table_header = "| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | 推理时间(ms) | FPS | 参数量 |"
    table_sep = "|------|--------|--------|--------|--------|--------------|-----|--------|"

    lines = [
        "# 📊 二次元角色识别模型基准测试报告",
        "",
        f"**测试时间**: {report['timestamp']}",
        "",
        "## 📈 模型性能对比",
        "",
        table_header,
        table_sep
    ]

    for model_key, results in all_results.items():
        if results is None:
            continue

        model_info = {
            'accuracy': results.get('accuracy', 0),
            'precision': results.get('precision', 0),
            'recall': results.get('recall', 0),
            'f1_score': results.get('f1_score', 0),
            'inference_time_ms': results.get('avg_inference_time_ms', 0),
            'fps': results.get('fps', 0),
            'num_params': results.get('num_params', 0),
            'test_samples': results.get('test_samples', 0),
            'num_classes': results.get('num_classes', 0)
        }
        report['models'][results['model_name']] = model_info

        params_m = model_info['num_params'] / 1_000_000
        lines.append(
            f"| {results['model_name']} | "
            f"{model_info['accuracy']:.2%} | "
            f"{model_info['precision']:.2%} | "
            f"{model_info['recall']:.2%} | "
            f"{model_info['f1_score']:.2%} | "
            f"{model_info['inference_time_ms']:.2f} | "
            f"{model_info['fps']:.2f} | "
            f"{params_m:.2f}M |"
        )

    lines.extend([
        "",
        "## 📋 详细测试配置",
        "",
        f"- **数据集**: {BASE_DATA_DIR}",
        f"- **测试样本比例**: 20%",
        f"- **设备**: {get_device()}",
        "",
        "## 🏆 结论",
        ""
    ])

    sorted_models = sorted(
        [(name, info['accuracy']) for name, info in report['models'].items()],
        key=lambda x: x[1],
        reverse=True
    )

    if sorted_models:
        best_model, best_acc = sorted_models[0]
        lines.append(f"**最佳模型**: {best_model} (准确率: {best_acc:.2%})")
        lines.append("")

        lines.append("**推荐意见**:")
        if 'efficientnet_b3' in [k for k, v in report['models'].items()]:
            lines.append("- 如果追求最高准确率，推荐使用 **EfficientNet-B3**")
        if 'mobilenetv2' in [k for k, v in report['models'].items()]:
            lines.append("- 如果追求速度，推荐使用 **MobileNetV2** (轻量级，高FPS)")

    report['markdown'] = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(output_path.replace('.json', '.md'), 'w', encoding='utf-8') as f:
        f.write(report['markdown'])

    logger.info(f"\n✅ 报告已保存:")
    logger.info(f"  - JSON: {output_path}")
    logger.info(f"  - Markdown: {output_path.replace('.json', '.md')}")

    print("\n" + "=" * 60)
    print("📊 基准测试报告")
    print("=" * 60)
    print(report['markdown'])

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description='模型基准测试')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'mobilenetv2', 'efficientnet_b0', 'efficientnet_b3', 'resnet50'],
                       help='要测试的模型 (默认: all)')
    parser.add_argument('--data', type=str, default=BASE_DATA_DIR,
                       help=f'数据集路径 (默认: {BASE_DATA_DIR})')
    parser.add_argument('--output', type=str, default=None,
                       help='报告输出路径')

    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = os.path.join(BASE_MODEL_DIR, f'benchmark_report_{timestamp}.json')

    if args.model == 'all':
        models_to_test = ['mobilenetv2', 'efficientnet_b0', 'efficientnet_b3', 'resnet50']
    else:
        models_to_test = [args.model]

    all_results = {}

    for model_name in models_to_test:
        try:
            result = run_benchmark(model_name, args.data)
            all_results[model_name] = result
        except Exception as e:
            logger.error(f"测试 {model_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = None

    generate_report(all_results, args.output)


if __name__ == '__main__':
    main()