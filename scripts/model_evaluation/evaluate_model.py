#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本 - 使用独立数据集评估训练好的模型
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from collections import defaultdict
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ModelEvaluator:
    """
    模型评估器
    """

    def __init__(self, model_path, results_path):
        """
        初始化评估器

        Args:
            model_path: 模型文件路径
            results_path: 训练结果JSON路径
        """
        self.model_path = model_path
        self.results_path = results_path
        self.model = None
        self.device = None
        self.class_names = None
        self.transform = None

    def load_model(self):
        """
        加载训练好的模型
        """
        # 读取训练结果配置
        with open(self.results_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # 获取设备
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"📦 使用设备: {self.device}")

        # 创建模型架构
        num_classes = self.config["num_classes"]
        self.model = models.mobilenet_v2(num_classes=num_classes)

        # 加载权重
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✅ 模型加载成功: {self.model_path}")

        # 加载类别名称
        self._load_class_names()

        # 定义图像变换
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.config["image_size"], self.config["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_class_names(self):
        """
        加载类别名称列表
        """
        # 从数据目录获取类别名称
        data_dir = os.path.join(project_root, "data", "balanced_dataset", "train")
        if os.path.exists(data_dir):
            self.class_names = sorted(
                [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            )
            print(f"📋 加载 {len(self.class_names)} 个类别")
        else:
            self.class_names = [f"class_{i}" for i in range(self.config["num_classes"])]

    def predict(self, image_path, top_k=5):
        """
        预测单张图片

        Args:
            image_path: 图片路径
            top_k: 返回前k个预测结果

        Returns:
            预测结果列表 [(class_name, probability), ...]
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, top_k)

            results = []
            for i in range(top_k):
                class_idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                class_name = (
                    self.class_names[class_idx]
                    if class_idx < len(self.class_names)
                    else f"unknown_{class_idx}"
                )
                results.append((class_name, prob))

            return results

        except Exception as e:
            print(f"❌ 预测失败 {image_path}: {e}")
            return []

    def evaluate_directory(self, data_dir):
        """
        评估整个目录

        Args:
            data_dir: 数据目录（包含子目录作为类别）

        Returns:
            评估结果字典
        """
        results = {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "per_class": {},
            "confusion_matrix": None,
        }

        # 统计每类的预测结果
        per_class_stats = defaultdict(lambda: {"total": 0, "correct": 0})

        # 获取所有类别
        classes = sorted(
            [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        )
        class_to_idx = {c: i for i, c in enumerate(classes)}

        # 初始化混淆矩阵
        num_classes = len(classes)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for class_name in classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            print(f"\n🔍 评估类别: {class_name}")

            for filename in os.listdir(class_dir):
                if not filename.endswith(".jpg"):
                    continue

                image_path = os.path.join(class_dir, filename)
                predictions = self.predict(image_path, top_k=1)

                if predictions:
                    pred_class, prob = predictions[0]
                    results["total"] += 1
                    per_class_stats[class_name]["total"] += 1

                    # 更新混淆矩阵
                    if class_name in class_to_idx and pred_class in class_to_idx:
                        true_idx = class_to_idx[class_name]
                        pred_idx = class_to_idx[pred_class]
                        confusion_matrix[true_idx][pred_idx] += 1

                    if pred_class == class_name:
                        results["correct"] += 1
                        per_class_stats[class_name]["correct"] += 1
                        print(f"  ✅ {filename} -> {pred_class} ({prob:.2f})")
                    else:
                        print(f"  ❌ {filename} -> {pred_class} ({prob:.2f}) [真实: {class_name}]")

        # 计算准确率
        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"]

        # 计算每类准确率
        for class_name, stats in per_class_stats.items():
            if stats["total"] > 0:
                per_class_acc = stats["correct"] / stats["total"]
                results["per_class"][class_name] = {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": per_class_acc,
                }

        results["confusion_matrix"] = confusion_matrix.tolist()
        results["classes"] = classes

        return results

    def evaluate_no_role_images(self, no_role_dir, output_file=None):
        """
        评估 no_role 目录中的图片（无标签数据）

        Args:
            no_role_dir: no_role 目录路径
            output_file: 输出结果文件
        """
        results = []

        print(f"\n🔍 评估 no_role 目录...")

        for filename in os.listdir(no_role_dir):
            if not filename.endswith(".jpg"):
                continue

            image_path = os.path.join(no_role_dir, filename)
            predictions = self.predict(image_path, top_k=3)

            if predictions:
                result = {
                    "filename": filename,
                    "predictions": [{"class": c, "probability": p} for c, p in predictions],
                }
                results.append(result)

                print(f"  {filename}: {', '.join([f'{c} ({p:.2f})' for c, p in predictions])}")

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 结果已保存: {output_file}")

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="模型评估工具")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--results_path", type=str, required=True, help="训练结果JSON路径")
    parser.add_argument("--eval_dir", type=str, help="评估数据目录（带标签）")
    parser.add_argument("--no_role_dir", type=str, help="no_role 目录（无标签数据）")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="输出文件")

    args = parser.parse_args()

    # 创建评估器
    evaluator = ModelEvaluator(args.model_path, args.results_path)
    evaluator.load_model()

    all_results = {}

    # 评估带标签的数据
    if args.eval_dir:
        print(f"\n{'='*60}")
        print(f"📊 评估带标签数据: {args.eval_dir}")
        print("=" * 60)

        eval_results = evaluator.evaluate_directory(args.eval_dir)
        all_results["labeled_evaluation"] = eval_results

        print(f"\n{'='*60}")
        print("📊 评估结果")
        print("=" * 60)
        print(f"总样本数: {eval_results['total']}")
        print(f"正确数: {eval_results['correct']}")
        print(f"准确率: {eval_results['accuracy'] * 100:.2f}%")

        # 输出每类准确率（只显示部分）
        print("\n📋 部分类别准确率:")
        sorted_classes = sorted(
            eval_results["per_class"].items(), key=lambda x: x[1]["accuracy"], reverse=True
        )
        for class_name, stats in sorted_classes[:10]:
            print(
                f"  {class_name}: {stats['accuracy'] * 100:.2f}% ({stats['correct']}/{stats['total']})"
            )

    # 评估 no_role 数据
    if args.no_role_dir:
        print(f"\n{'='*60}")
        print(f"🔮 评估无标签数据: {args.no_role_dir}")
        print("=" * 60)

        no_role_results = evaluator.evaluate_no_role_images(args.no_role_dir, args.output)
        all_results["no_role_predictions"] = no_role_results

    # 保存综合结果
    if all_results:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 综合评估结果已保存: {args.output}")


if __name__ == "__main__":
    main()
