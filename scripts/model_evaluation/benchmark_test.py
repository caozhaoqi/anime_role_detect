#!/usr/bin/env python3
"""模型基准测试脚本"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append((img_path, self.class_to_idx[class_name]))

        self.class_names = list(self.class_to_idx.keys())

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
            logger.warning(f"跳过损坏图片: {img_path} - {e}")
            image = Image.new('RGB', (224, 224), color=0)
            if self.transform:
                image = self.transform(image)
            return image, label


class BenchmarkEvaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.device = None

    def load_model(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(f"使用设备: {self.device}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
        else:
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            model_path_lower = self.model_path.lower()
            if "efficientnet" in model_path_lower:
                model = models.efficientnet_b3()
                num_classes = state_dict.get("classifier.1.weight", torch.randn(100, 1536)).shape[0]
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif "resnet" in model_path_lower:
                model = models.resnet50()
                num_classes = state_dict.get("fc.weight", torch.randn(100, 2048)).shape[0]
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            else:
                model = models.mobilenet_v2()
                num_classes = state_dict.get("classifier.1.weight", torch.randn(100, 1280)).shape[0]
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

            model.load_state_dict(state_dict, strict=False)
            self.model = model

        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"模型加载成功: {self.model_path}")

    def benchmark(self, test_loader):
        total_correct = 0
        total_samples = 0
        inference_times = []
        top3_correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                start_time = time.time()
                outputs = self.model(images)
                end_time = time.time()

                inference_times.append(end_time - start_time)

                _, preds = torch.max(outputs, 1)
                total_correct += torch.sum(preds == labels).item()

                _, top3_preds = torch.topk(outputs, 3)
                top3_correct += sum(labels[i] in top3_preds[i] for i in range(len(labels)))
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        top3_accuracy = top3_correct / total_samples if total_samples > 0 else 0
        avg_inference_time = np.mean(inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

        return {
            "accuracy": accuracy,
            "top3_accuracy": top3_accuracy,
            "avg_inference_time": avg_inference_time,
            "fps": fps,
            "total_samples": total_samples,
            "correct_predictions": total_correct,
        }


class SyntheticDataset(Dataset):
    def __init__(self, num_samples=500, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.class_names = [f"class_{i}" for i in range(100)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = Image.new('RGB', (224, 224), color=(idx % 256, (idx * 7) % 256, (idx * 13) % 256))
        if self.transform:
            image = self.transform(image)
        return image, idx % 100


def main():
    model_configs = [
        {"name": "EfficientNet-B3", "path": "models/efficientnet_b3/model_best.pth"},
        {"name": "EfficientNet-B3 (Full)", "path": "models/efficientnet_b3/model_full.pth"},
    ]

    test_dir = os.path.join(project_root, "data", "training_dataset")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
        test_dataset = TestDataset(test_dir, transform=transform)
        logger.info(f"测试数据集: {len(test_dataset)} 样本, {len(test_dataset.class_names)} 类别")
    else:
        logger.warning(f"真实测试数据集不存在: {test_dir}")
        logger.info("使用合成数据集进行性能测试...")
        test_dataset = SyntheticDataset(num_samples=500, transform=transform)
        logger.info(f"合成测试数据集: {len(test_dataset)} 样本, {len(test_dataset.class_names)} 类别")

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    logger.info(f"测试批次大小: 32\n")

    results = []

    for config in model_configs:
        model_path = os.path.join(project_root, config["path"])
        if not os.path.exists(model_path):
            logger.warning(f"模型不存在: {model_path}")
            continue

        logger.info(f"=" * 50)
        logger.info(f"基准测试: {config['name']}")
        logger.info(f"=" * 50)

        evaluator = BenchmarkEvaluator(model_path)
        evaluator.load_model()

        result = evaluator.benchmark(test_loader)

        logger.info(f"Top-1准确率: {result['accuracy']:.4f}")
        logger.info(f"Top-3准确率: {result['top3_accuracy']:.4f}")
        logger.info(f"平均推理时间: {result['avg_inference_time']:.4f} s")
        logger.info(f"FPS: {result['fps']:.2f}")
        logger.info(f"测试样本: {result['total_samples']}")
        logger.info(f"正确预测: {result['correct_predictions']}\n")

        results.append({
            "model_name": config["name"],
            "model_path": model_path,
            **result,
        })

    logger.info("=" * 50)
    logger.info("基准测试汇总")
    logger.info("=" * 50)
    logger.info(f"{'模型名称':<20} {'Top-1准确率':<12} {'Top-3准确率':<12} {'推理时间(s)':<12} {'FPS':<8}")
    logger.info("-" * 70)

    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        logger.info(f"{r['model_name']:<20} {r['accuracy']:.4f}         {r['top3_accuracy']:.4f}         {r['avg_inference_time']:.4f}         {r['fps']:.2f}")

    report_path = os.path.join(project_root, "models", "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
