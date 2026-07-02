#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二次元角色识别模型训练脚本
- 自动从 data/final_dataset 获取数据
- 过滤图片数 > 50 的角色
- CPU版本，适配 Win/Mac/Linux (Debian)
- 支持三种模型架构: MobileNetV2, EfficientNet-B0, ResNet50
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cpu")
logger.info(f"使用设备: {DEVICE}")

MODEL_CONFIGS = {
    "mobilenetv2": {
        "name": "MobileNetV2",
        "image_size": 224,
        "batch_size": 32,
        "model_fn": models.mobilenet_v2,
        "weights": "IMAGENET1K_V1",
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "image_size": 224,
        "batch_size": 32,
        "model_fn": models.efficientnet_b0,
        "weights": "IMAGENET1K_V1",
    },
    "resnet50": {
        "name": "ResNet50",
        "image_size": 224,
        "batch_size": 32,
        "model_fn": models.resnet50,
        "weights": "IMAGENET1K_V1",
    },
}

NUM_EPOCHS = 50
PATIENCE = 10
EARLY_STOP_THRESHOLD = 0.001
LEARNING_RATE = 1e-4
MIN_IMAGES_PER_CHARACTER = 50

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "final_dataset"
MODEL_SAVE_PATH = PROJECT_ROOT / "models"


def get_platform():
    """获取操作系统平台"""
    if sys.platform.startswith("win"):
        return "Windows"
    elif sys.platform.startswith("darwin"):
        return "macOS"
    elif sys.platform.startswith("linux"):
        return "Linux"
    return sys.platform


def collect_valid_characters(dataset_path):
    """收集图片数大于阈值的角色"""
    valid_characters = []
    character_stats = {}

    if not dataset_path.exists():
        logger.error(f"数据集路径不存在: {dataset_path}")
        return [], {}

    for item in sorted(dataset_path.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue

        image_files = list(item.glob("*.jpg")) + list(item.glob("*.jpeg")) + list(item.glob("*.png"))
        image_count = len(image_files)

        if image_count >= MIN_IMAGES_PER_CHARACTER:
            valid_characters.append(item.name)
            character_stats[item.name] = image_count
            logger.info(f"  ✅ {item.name}: {image_count} 张图片")
        else:
            logger.info(f"  ❌ {item.name}: {image_count} 张图片 (低于阈值 {MIN_IMAGES_PER_CHARACTER})")

    return valid_characters, character_stats


def filter_dataset_by_characters(dataset, valid_characters):
    """根据角色列表过滤数据集"""
    valid_indices = []
    for idx in range(len(dataset)):
        _, label = dataset.samples[idx]
        class_name = dataset.classes[label]
        if class_name in valid_characters:
            valid_indices.append(idx)
    return Subset(dataset, valid_indices)


def create_model(model_name, num_classes):
    """创建指定模型"""
    config = MODEL_CONFIGS[model_name]
    model_fn = config["model_fn"]
    weights = config["weights"]

    logger.info(f"创建模型: {config['name']}, 类别数: {num_classes}")

    if model_name == "mobilenetv2":
        model = model_fn(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "efficientnet_b0":
        model = model_fn(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
        model = model_fn(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"未知模型: {model_name}")

    return model.to(DEVICE)


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, patience):
    """训练模型"""
    since = time.time()

    best_model_wts = model.state_dict().copy()
    best_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            logger.info(f"  {phase} Loss: {epoch_loss:.4f}, {phase} Acc: {epoch_acc:.4%}")

            if phase == "val":
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())
                scheduler.step(epoch_acc)

                if epoch_acc > best_acc + EARLY_STOP_THRESHOLD:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict().copy()
                    patience_counter = 0
                    logger.info(f"  ✅ 保存最佳模型，准确率: {best_acc:.4%}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"  验证准确率连续 {patience} 轮未提升，提前停止训练")
                        model.load_state_dict(best_model_wts)
                        return model, best_acc, history
            else:
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())

    time_elapsed = time.time() - since
    logger.info(f"\n训练完成，耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    logger.info(f"最佳验证准确率: {best_acc:.4%}")

    model.load_state_dict(best_model_wts)
    return model, best_acc, history


def train(model_name, data_dir=None, model_dir=None):
    """训练指定模型"""
    if data_dir is None:
        data_dir = DATASET_PATH
    if model_dir is None:
        model_dir = MODEL_SAVE_PATH / f"{model_name}_trained"

    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    config = MODEL_CONFIGS[model_name]
    image_size = config["image_size"]
    batch_size = config["batch_size"]

    logger.info("=" * 60)
    logger.info(f"🎬 训练 {config['name']} 模型")
    logger.info(f"平台: {get_platform()}")
    logger.info("=" * 60)

    logger.info(f"\n📦 收集有效角色 (>= {MIN_IMAGES_PER_CHARACTER} 张图片)...")
    valid_characters, character_stats = collect_valid_characters(data_dir)

    if not valid_characters:
        logger.error("没有找到足够图片的角色，训练终止")
        return None

    logger.info(f"\n✅ 找到 {len(valid_characters)} 个有效角色")
    logger.info(f"总图片数: {sum(character_stats.values())}")

    logger.info(f"\n🔄 准备数据集...")

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = ImageFolder(str(data_dir), transform=train_transform)
    filtered_dataset = filter_dataset_by_characters(full_dataset, valid_characters)

    train_size = int(0.8 * len(filtered_dataset))
    val_size = len(filtered_dataset) - train_size
    train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform

    logger.info(f"数据集: {len(filtered_dataset)} 样本")
    logger.info(f"训练集: {train_size}, 验证集: {val_size}")

    num_workers = 0 if get_platform() == "Windows" else 4

    dataloaders = {
        "train": DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }

    valid_class_indices = sorted(set(filtered_dataset.indices))
    original_class_to_idx = full_dataset.class_to_idx
    valid_classes = []
    valid_class_to_idx = {}

    for idx in valid_class_indices:
        _, label = full_dataset.samples[idx]
        class_name = full_dataset.classes[label]
        if class_name not in valid_class_to_idx:
            valid_class_to_idx[class_name] = len(valid_classes)
            valid_classes.append(class_name)

    idx_to_class = {v: k for k, v in valid_class_to_idx.items()}

    with open(model_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(
            {"class_to_idx": valid_class_to_idx, "idx_to_class": idx_to_class},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"✅ 保存类别映射")

    num_classes = len(valid_classes)
    model = create_model(model_name, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    try:
        logger.info(f"\n🚀 开始训练 {config['name']}")
        model, best_acc, history = train_model(
            model, dataloaders, criterion, optimizer, scheduler, NUM_EPOCHS, PATIENCE
        )

        torch.save(model.state_dict(), model_dir / "model_best.pth")
        torch.save(model, model_dir / "model_full.pth")

        logger.info(f"✅ 模型已保存: {model_dir / 'model_best.pth'}")

        results = {
            "model": config["name"],
            "model_key": model_name,
            "accuracy": best_acc.item(),
            "epochs": NUM_EPOCHS,
            "batch_size": batch_size,
            "learning_rate": LEARNING_RATE,
            "image_size": image_size,
            "device": str(DEVICE),
            "platform": get_platform(),
            "num_classes": num_classes,
            "train_samples": train_size,
            "val_samples": val_size,
            "characters": valid_classes,
            "character_stats": character_stats,
            "history": history,
            "timestamp": datetime.now().isoformat(),
        }

        with open(model_dir / "training_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 训练结果已保存")

        return results

    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description="二次元角色识别模型训练")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "mobilenetv2", "efficientnet_b0", "resnet50"],
        help="要训练的模型 (默认: all)",
    )
    parser.add_argument(
        "--data", type=str, default=str(DATASET_PATH), help=f"数据集路径 (默认: {DATASET_PATH})"
    )
    parser.add_argument(
        "--output", type=str, default=str(MODEL_SAVE_PATH), help=f"模型输出目录 (默认: {MODEL_SAVE_PATH})"
    )
    parser.add_argument(
        "--min-images", type=int, default=50, help="最小图片数阈值 (默认: 50)"
    )

    args = parser.parse_args()

    global MIN_IMAGES_PER_CHARACTER
    MIN_IMAGES_PER_CHARACTER = args.min_images

    logger.info(f"平台: {get_platform()}")
    logger.info(f"设备: {DEVICE}")
    logger.info(f"数据集路径: {args.data}")
    logger.info(f"最小图片数阈值: {MIN_IMAGES_PER_CHARACTER}")

    if args.model == "all":
        models_to_train = ["mobilenetv2", "efficientnet_b0", "resnet50"]
    else:
        models_to_train = [args.model]

    all_results = {}

    for model_name in models_to_train:
        try:
            result = train(model_name, args.data, args.output)
            all_results[model_name] = result
        except Exception as e:
            logger.error(f"训练 {model_name} 失败: {e}")
            all_results[model_name] = {"error": str(e)}

    logger.info("\n" + "=" * 60)
    logger.info("🎬 所有训练任务结束")
    logger.info("=" * 60)

    if all_results:
        summary_lines = []
        for k, v in all_results.items():
            if isinstance(v, dict) and "accuracy" in v:
                summary_lines.append(f"- {MODEL_CONFIGS[k]['name']}: {v['accuracy']:.4%}")
            elif isinstance(v, dict) and "error" in v:
                summary_lines.append(f"- {MODEL_CONFIGS[k]['name']}: ❌ {v['error']}")
            else:
                summary_lines.append(f"- {MODEL_CONFIGS[k]['name']}: N/A")

        logger.info("\n训练结果汇总:")
        for line in summary_lines:
            logger.info(line)


if __name__ == "__main__":
    main()