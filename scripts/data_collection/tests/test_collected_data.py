#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试采集数据的模型分类性能
使用分类结果文件来评估模型的性能
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger

    logger = get_logger("test_collected")
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("test_collected")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BATCH_SIZE = 8
IMAGE_SIZE = 224
NUM_WORKERS = 0
DATA_DIR = "./data/downloaded_images"
MODEL_DIR = "./models"
CLASSIFIED_DIR = "./auto_spider_img/classified_all_v2"


class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        class_names = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

        logger.info(f"数据集: {len(self.samples)} 样本, {len(self.class_to_idx)} 类别")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            logger.error(f"加载图像失败 {img_path}: {e}")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
            if self.transform:
                image = self.transform(image)
            return image, label, img_path


def get_model(model_type, num_classes):
    if model_type == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(feature_dim, num_classes))
    elif model_type == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(feature_dim, num_classes))
    elif model_type == "resnet18":
        model = models.resnet18(pretrained=True)
        feature_dim = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(feature_dim, num_classes))
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    return model


def load_trained_model(model_type):
    model_path = os.path.join(MODEL_DIR, model_type, "model_full.pth")
    if not os.path.exists(model_path):
        logger.warning(f"模型文件不存在: {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_type" not in checkpoint:
        model = checkpoint
    else:
        model = checkpoint

    logger.info(f"加载模型: {model_type}")
    return model


def evaluate_model_on_collected_data(model, dataloader, device, class_to_idx, idx_to_class):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    inference_times = []
    correct = 0
    total = 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    misclassified = []

    with torch.no_grad():
        for images, labels, paths in dataloader:
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
            all_paths.extend(paths)

            for pred, label, path in zip(predicted.cpu().numpy(), labels.cpu().numpy(), paths):
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
                else:
                    misclassified.append(
                        {
                            "path": path,
                            "true_label": idx_to_class[label],
                            "pred_label": idx_to_class[pred],
                            "true_idx": label,
                            "pred_idx": pred,
                        }
                    )

    accuracy = 100.0 * correct / total
    avg_inference_time = np.mean(inference_times) * 1000
    std_inference_time = np.std(inference_times) * 1000
    throughput = 1.0 / (avg_inference_time / 1000)

    return {
        "accuracy": accuracy,
        "avg_inference_time_ms": avg_inference_time,
        "std_inference_time_ms": std_inference_time,
        "throughput_samples_per_sec": throughput,
        "class_accuracy": {
            k: 100.0 * class_correct[k] / class_total[k] if class_total[k] > 0 else 0
            for k in class_total.keys()
        },
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_probs": all_probs,
        "all_paths": all_paths,
        "misclassified": misclassified,
        "class_distribution": dict(class_total),
    }


def calculate_confusion_matrix(preds, labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(preds, labels):
        cm[label][pred] += 1
    return cm


def print_test_report(results, class_to_idx, idx_to_class):
    print("\n" + "=" * 80)
    print(" " * 20 + "采集数据模型测试报告")
    print("=" * 80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试设备: CPU")
    print(f"测试数据集: {DATA_DIR}")
    print(f"采集分类结果: {CLASSIFIED_DIR}")
    print("=" * 80)

    print("\n" + "-" * 80)
    print(" " * 25 + "1. 总体性能对比")
    print("-" * 80)
    print(f"{'模型名称':<20} {'准确率':<12} {'推理时间(ms)':<15} {'吞吐量(sample/s)':<20}")
    print("-" * 80)

    for model_type, result in results.items():
        if "error" in result:
            print(f"{model_type:<20} {'失败':<12} {'-':<15} {'-':<20}")
        else:
            print(
                f"{model_type:<20} {result['accuracy']:.2f}%{'':<6} "
                f"{result['avg_inference_time_ms']:.2f}±{result['std_inference_time_ms']:.2f}{'':<6} "
                f"{result['throughput_samples_per_sec']:.2f}"
            )

    print("-" * 80)

    print("\n" + "-" * 80)
    print(" " + "2. 各类别准确率")
    print("-" * 80)
    print(f"{'类别名称':<25} ", end="")
    for model_type in results.keys():
        if "error" not in results[model_type]:
            print(f"{model_type:<15}", end="")
    print()
    print("-" * 80)

    for class_idx in sorted(idx_to_class.keys()):
        class_name = idx_to_class[class_idx]
        print(f"{class_name:<25} ", end="")
        for model_type in results.keys():
            if "error" not in results[model_type]:
                acc = results[model_type]["class_accuracy"].get(class_idx, 0)
                print(f"{acc:.2f}%{'':<10}", end="")
        print()

    print("-" * 80)

    for model_type, result in results.items():
        if "error" in result:
            continue

        print("\n" + "-" * 80)
        print(f" " * 20 + f"3. {model_type} 混淆矩阵")
        print("-" * 80)

        num_classes = len(idx_to_class)
        cm = calculate_confusion_matrix(result["all_preds"], result["all_labels"], num_classes)

        header = "真实\\预测"
        print(f"{header:<15}", end="")
        for i in range(num_classes):
            short_name = idx_to_class[i][:8] if len(idx_to_class[i]) > 8 else idx_to_class[i]
            print(f"{short_name:<12}", end="")
        print()
        print("-" * 80)

        for i in range(num_classes):
            short_name = idx_to_class[i][:8] if len(idx_to_class[i]) > 8 else idx_to_class[i]
            print(f"{short_name:<15}", end="")
            for j in range(num_classes):
                print(f"{cm[i][j]:<12}", end="")
            print()

        print("-" * 80)

    print("\n" + "-" * 80)
    print(" " + "4. 模型性能分析")
    print("-" * 80)

    for model_type, result in results.items():
        if "error" in result:
            continue

        print(f"\n{model_type}:")
        print(f"  总体准确率: {result['accuracy']:.2f}%")
        print(f"  推理速度: {result['avg_inference_time_ms']:.2f}ms")
        print(f"  吞吐量: {result['throughput_samples_per_sec']:.2f} samples/s")
        print(f"  误分类样本数: {len(result['misclassified'])}")

        if result["misclassified"]:
            print("  前5个误分类样本:")
            for i, item in enumerate(result["misclassified"][:5]):
                print(f"    {i+1}. {os.path.basename(item['path'])}")
                print(f"       真实: {item['true_label']} -> 预测: {item['pred_label']}")

    print("-" * 80)
    print("\n" + "=" * 80)
    print(" " * 25 + "测试报告结束")
    print("=" * 80 + "\n")


def load_classified_results():
    classified_files = {
        "萝莉": os.path.join(CLASSIFIED_DIR, "萝莉.txt"),
        "不属于萝莉": os.path.join(CLASSIFIED_DIR, "不属于萝莉.txt"),
        "无明显特征": os.path.join(CLASSIFIED_DIR, "无明显特征.txt"),
        "未找到角色": os.path.join(CLASSIFIED_DIR, "未找到角色.txt"),
    }

    classified_roles = {}
    for category, file_path in classified_files.items():
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                roles = [line.strip() for line in f if line.strip()]
                classified_roles[category] = roles
            logger.info(f"加载 {category}: {len(roles)} 个角色")
        else:
            logger.warning(f"文件不存在: {file_path}")
            classified_roles[category] = []

    return classified_roles


def main():
    logger.info("=" * 60)
    logger.info("开始采集数据模型测试")
    logger.info("=" * 60)

    device = torch.device("cpu")
    logger.info(f"使用设备: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    logger.info(f"加载测试数据: {DATA_DIR}")
    dataset = SimpleImageDataset(DATA_DIR, transform=transform)
    idx_to_class = dataset.idx_to_class
    class_to_idx = dataset.class_to_idx
    num_classes = len(idx_to_class)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    logger.info(f"测试数据集大小: {len(dataset)}")
    logger.info(f"类别数: {num_classes}")
    logger.info(f"类别映射: {idx_to_class}")

    logger.info(f"加载采集分类结果: {CLASSIFIED_DIR}")
    classified_roles = load_classified_results()

    model_types = ["mobilenet_v2", "efficientnet_b0", "resnet18"]
    results = {}

    for model_type in model_types:
        logger.info(f"\n测试模型: {model_type}")
        try:
            model = load_trained_model(model_type)
            if model is None:
                results[model_type] = {"error": "模型文件不存在"}
                continue

            result = evaluate_model_on_collected_data(
                model, dataloader, device, class_to_idx, idx_to_class
            )
            results[model_type] = result

            logger.info(f"  准确率: {result['accuracy']:.2f}%")
            logger.info(
                f"  推理时间: {result['avg_inference_time_ms']:.2f}±{result['std_inference_time_ms']:.2f} ms"
            )
            logger.info(f"  吞吐量: {result['throughput_samples_per_sec']:.2f} samples/s")
            logger.info(f"  误分类样本: {len(result['misclassified'])}")

        except Exception as e:
            logger.error(f"模型 {model_type} 测试失败: {e}")
            import traceback

            logger.error(f"异常堆栈: {traceback.format_exc()}")
            results[model_type] = {"error": str(e)}

    print_test_report(results, class_to_idx, idx_to_class)

    report_path = "./collected_data_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "device": "cpu",
                "dataset_size": len(dataset),
                "num_classes": num_classes,
                "class_mapping": idx_to_class,
                "classified_roles": classified_roles,
                "results": {
                    k: {
                        key: val
                        for key, val in v.items()
                        if key
                        not in [
                            "all_preds",
                            "all_labels",
                            "all_probs",
                            "all_paths",
                            "misclassified",
                        ]
                    }
                    for k, v in results.items()
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"测试报告已保存到: {report_path}")

    logger.info("=" * 60)
    logger.info("采集数据模型测试完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
