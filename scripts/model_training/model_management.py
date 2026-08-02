#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型版本管理脚本 - 管理模型版本、评估模型性能
"""

import os
import sys
import torch
import json
import glob
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger

    logger = get_logger("model_management")
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("model_management")

MODEL_DIR = "./models"


def list_models():
    """列出所有模型版本"""
    logger.info("=" * 60)
    logger.info("模型版本列表")
    logger.info("=" * 60)

    # 查找所有模型目录
    model_dirs = []
    for item in os.listdir(MODEL_DIR):
        item_path = os.path.join(MODEL_DIR, item)
        if os.path.isdir(item_path):
            # 检查是否是模型目录
            if os.path.exists(os.path.join(item_path, "model_best.pth")):
                model_dirs.append(item_path)

    # 按时间排序
    model_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    if not model_dirs:
        logger.info("没有找到模型")
        return

    # 显示模型信息
    for i, model_dir in enumerate(model_dirs):
        model_name = os.path.basename(model_dir)

        # 读取模型元数据
        metadata_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            accuracy = metadata.get("accuracy", "N/A")
            test_accuracy = metadata.get("test_accuracy", "N/A")
            timestamp = metadata.get("timestamp", "N/A")
        else:
            # 尝试从checkpoint读取
            checkpoint_path = os.path.join(model_dir, "model_best.pth")
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
                    accuracy = checkpoint.get("accuracy", "N/A")
                    timestamp = checkpoint.get("timestamp", "N/A")
                    test_accuracy = "N/A"
                except:
                    accuracy = "N/A"
                    timestamp = "N/A"
                    test_accuracy = "N/A"
            else:
                accuracy = "N/A"
                timestamp = "N/A"
                test_accuracy = "N/A"

        # 检查是否是最佳模型
        best_model_path = os.path.join(MODEL_DIR, "best_incremental_model.txt")
        is_best = False
        if os.path.exists(best_model_path):
            with open(best_model_path, "r") as f:
                best_model = f.read().strip()
            if model_dir == best_model:
                is_best = True

        best_mark = " [最佳]" if is_best else ""
        logger.info(f"{i+1}. {model_name}{best_mark}")
        logger.info(f"   准确率: {accuracy}")
        logger.info(f"   测试准确率: {test_accuracy}")
        logger.info(f"   时间: {timestamp}")
        logger.info(f"   路径: {model_dir}")
        logger.info("-" * 40)


def compare_models(model1_path, model2_path, test_data_dir):
    """比较两个模型的性能"""
    logger.info("=" * 60)
    logger.info("模型性能比较")
    logger.info("=" * 60)
    logger.info(f"模型1: {model1_path}")
    logger.info(f"模型2: {model2_path}")
    logger.info(f"测试数据: {test_data_dir}")

    # 加载测试数据
    from torchvision import transforms
    from PIL import Image
    import torch
    from torch.utils.data import DataLoader, Dataset

    class SimpleImageDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.samples = []

            class_names = sorted(
                [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            )
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}

            for class_name in class_names:
                class_dir = os.path.join(root_dir, class_name)
                class_idx = class_to_idx[class_name]
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            try:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label
            except Exception as e:
                logger.error(f"加载失败 {img_path}: {e}")
                image = Image.new("RGB", (224, 224), (128, 128, 128))
                if self.transform:
                    image = self.transform(image)
                return image, label

    # 数据变换
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 加载测试数据
    test_dataset = SimpleImageDataset(test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 加载模型1
    def load_model(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        model_type = checkpoint["model_type"]

        from torchvision import models
        import torch.nn as nn

        if model_type == "mobilenet_v2":
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, len(checkpoint["class_to_idx"])
            )
        elif model_type == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, len(checkpoint["class_to_idx"])
            )
        elif model_type == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(checkpoint["class_to_idx"]))
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    try:
        model1 = load_model(os.path.join(model1_path, "model_best.pth"))
        model2 = load_model(os.path.join(model2_path, "model_best.pth"))
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return

    # 评估函数
    def evaluate(model, dataloader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    # 评估模型
    acc1 = evaluate(model1, test_loader)
    acc2 = evaluate(model2, test_loader)

    logger.info(f"模型1准确率: {acc1:.2f}%")
    logger.info(f"模型2准确率: {acc2:.2f}%")

    if acc1 > acc2:
        logger.info("模型1性能更好")
    elif acc2 > acc1:
        logger.info("模型2性能更好")
    else:
        logger.info("两个模型性能相当")


def create_model_summary():
    """创建模型性能摘要"""
    logger.info("=" * 60)
    logger.info("模型性能摘要")
    logger.info("=" * 60)

    # 查找所有模型
    model_dirs = []
    for item in os.listdir(MODEL_DIR):
        item_path = os.path.join(MODEL_DIR, item)
        if os.path.isdir(item_path):
            if os.path.exists(os.path.join(item_path, "model_best.pth")):
                model_dirs.append(item_path)

    # 按时间排序
    model_dirs.sort(key=lambda x: os.path.getmtime(x))

    # 收集模型信息
    model_info = []
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)

        # 读取元数据
        metadata_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            accuracy = metadata.get("accuracy", "N/A")
            test_accuracy = metadata.get("test_accuracy", "N/A")
            timestamp = metadata.get("timestamp", "N/A")
            is_incremental = metadata.get("is_incremental", False)
        else:
            # 尝试从checkpoint读取
            checkpoint_path = os.path.join(model_dir, "model_best.pth")
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
                    accuracy = checkpoint.get("accuracy", "N/A")
                    timestamp = checkpoint.get("timestamp", "N/A")
                    test_accuracy = "N/A"
                    is_incremental = checkpoint.get("is_incremental", False)
                except:
                    accuracy = "N/A"
                    timestamp = "N/A"
                    test_accuracy = "N/A"
                    is_incremental = False
            else:
                accuracy = "N/A"
                timestamp = "N/A"
                test_accuracy = "N/A"
                is_incremental = False

        model_info.append(
            {
                "name": model_name,
                "accuracy": accuracy,
                "test_accuracy": test_accuracy,
                "timestamp": timestamp,
                "is_incremental": is_incremental,
                "path": model_dir,
            }
        )

    # 保存摘要
    summary_path = os.path.join(MODEL_DIR, "model_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    # 显示摘要
    logger.info(f"共找到 {len(model_info)} 个模型")
    for info in model_info:
        incremental_mark = " [增量]" if info["is_incremental"] else ""
        logger.info(f"{info['name']}{incremental_mark}")
        logger.info(f"  准确率: {info['accuracy']}")
        logger.info(f"  测试准确率: {info['test_accuracy']}")
        logger.info(f"  时间: {info['timestamp']}")
        logger.info("-" * 40)

    logger.info(f"摘要已保存到: {summary_path}")


def rollback_model(target_version):
    """回滚到指定版本"""
    logger.info("=" * 60)
    logger.info(f"回滚模型到版本: {target_version}")
    logger.info("=" * 60)

    # 查找目标版本
    target_path = os.path.join(MODEL_DIR, target_version)
    if not os.path.exists(target_path) or not os.path.isdir(target_path):
        logger.error(f"目标版本不存在: {target_version}")
        return

    # 检查是否是有效的模型目录
    if not os.path.exists(os.path.join(target_path, "model_best.pth")):
        logger.error(f"目标版本不是有效的模型目录")
        return

    # 更新最佳模型链接
    best_model_link = os.path.join(MODEL_DIR, "best_incremental_model.txt")
    with open(best_model_link, "w") as f:
        f.write(target_path)

    logger.info(f"模型已回滚到: {target_path}")
    logger.info(f"最佳模型链接已更新")


def promote_model(src_dir, target="models/efficientnet_b3"):
    """将已验证的候选模型目录提升为生产模型目录（与 rollback_model 对称）。

    流程：
      1. 安全校验（src 必须存在且为目录、含 model_best.pth）；
      2. 把当前 target 整体备份为 target + ".bak_<timestamp>"；
      3. 将 src_dir 内容复制覆盖到 target。

    本函数不自动调用自身，也不做任何评测/消费判定——评测闸门由调用方负责。
    """
    import shutil

    # 1) 安全校验
    if not os.path.isdir(src_dir):
        logger.error(f"promote 失败：源目录不存在或不是目录: {src_dir}")
        return False
    if not os.path.exists(os.path.join(src_dir, "model_best.pth")):
        logger.error(f"promote 失败：源目录缺少 model_best.pth: {src_dir}")
        return False

    # target 解析：非绝对路径一律相对 project_root（不受 cwd 影响）
    target_path = target if os.path.isabs(target) else os.path.join(project_root, target)
    logger.info(f"promote: src={src_dir}")
    logger.info(f"promote: target={target_path}")

    # 2) 备份当前 target（若存在且为有效模型目录）
    if os.path.isdir(target_path) and os.path.exists(os.path.join(target_path, "model_best.pth")):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{target_path}.bak_{ts}"
        # 避免覆盖已有同名备份
        while os.path.exists(backup_path):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_path = f"{target_path}.bak_{ts}"
        shutil.copytree(target_path, backup_path)
        logger.info(f"已备份当前生产模型: {backup_path}")
    else:
        logger.warning(f"当前 target 不存在或不是有效模型目录，跳过备份: {target_path}")

    # 3) 复制 src 覆盖到 target
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    # 若 target 已存在，先清空再复制，确保覆盖语义干净
    if os.path.isdir(target_path):
        shutil.rmtree(target_path)
    shutil.copytree(src_dir, target_path)
    logger.info(f"已提升候选模型到生产目录: {target_path}")
    logger.info("promote 完成（未自动调用自身，亦未做评测判定）。")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="模型版本管理脚本")
    parser.add_argument(
        "command", choices=["list", "compare", "summary", "rollback", "promote"], help="执行的命令"
    )
    parser.add_argument("--model1", help="第一个模型路径（用于compare命令）")
    parser.add_argument("--model2", help="第二个模型路径（用于compare命令）")
    parser.add_argument("--test_data", help="测试数据目录（用于compare命令）")
    parser.add_argument("--version", help="目标版本（用于rollback命令）")
    parser.add_argument("--src", help="候选模型目录（用于promote命令）")
    parser.add_argument("--target", default="models/efficientnet_b3", help="目标生产目录（用于promote命令）")

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "compare":
        if not args.model1 or not args.model2 or not args.test_data:
            parser.error("compare命令需要指定 --model1、--model2 和 --test_data")
        compare_models(args.model1, args.model2, args.test_data)
    elif args.command == "summary":
        create_model_summary()
    elif args.command == "rollback":
        if not args.version:
            parser.error("rollback命令需要指定 --version")
        rollback_model(args.version)
    elif args.command == "promote":
        if not args.src:
            parser.error("promote命令需要指定 --src")
        ok = promote_model(args.src, target=args.target)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
