#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二次元角色识别模型训练脚本
- 自动从 data/final_dataset 获取数据
- 过滤图片数 > 50 的角色
- 自动检测并使用 GPU (CUDA/MPS)，CPU 作为备选
- 支持三种模型架构: MobileNetV2, EfficientNet-B0, ResNet50
- 自动获取系统配置，智能选择最佳训练参数
- 严格资源限制: 防止被系统 OOM kill
"""
import os
import sys
import json
import time
import logging
import argparse
import shutil
import psutil
import gc
import platform
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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

MODEL_CONFIGS = {
    "mobilenetv2": {
        "name": "MobileNetV2",
        "image_size": 224,
        "batch_size": 8,
        "model_fn": models.mobilenet_v2,
        "weights": "IMAGENET1K_V1",
        "memory_estimate_gb": 0.8,
        "speed": "fast",
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "image_size": 224,
        "batch_size": 8,
        "model_fn": models.efficientnet_b0,
        "weights": "IMAGENET1K_V1",
        "memory_estimate_gb": 1.0,
        "speed": "medium",
    },
    "resnet50": {
        "name": "ResNet50",
        "image_size": 224,
        "batch_size": 8,
        "model_fn": models.resnet50,
        "weights": "IMAGENET1K_V1",
        "memory_estimate_gb": 1.5,
        "speed": "slow",
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

MEMORY_SAFETY_MARGIN = 0.8
CPU_SAFETY_MARGIN = 0.7


def get_platform():
    """获取操作系统平台"""
    if sys.platform.startswith("win"):
        return "Windows"
    elif sys.platform.startswith("darwin"):
        return "macOS"
    elif sys.platform.startswith("linux"):
        return "Linux"
    return sys.platform


def detect_system_config():
    """自动检测系统配置信息"""
    config = {}

    config["platform"] = get_platform()
    config["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    config["python_executable"] = sys.executable

    total_cpu = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    logical_cpu = psutil.cpu_count() or 1
    config["physical_cpu_cores"] = total_cpu
    config["logical_cpu_cores"] = logical_cpu

    virtual_memory = psutil.virtual_memory()
    total_mem_gb = virtual_memory.total / (1024 ** 3)
    available_mem_gb = virtual_memory.available / (1024 ** 3)
    used_mem_gb = virtual_memory.used / (1024 ** 3)
    mem_percent = virtual_memory.percent

    config["total_memory_gb"] = round(total_mem_gb, 2)
    config["available_memory_gb"] = round(available_mem_gb, 2)
    config["used_memory_gb"] = round(used_mem_gb, 2)
    config["memory_usage_percent"] = mem_percent

    swap_memory = psutil.swap_memory()
    config["swap_total_gb"] = round(swap_memory.total / (1024 ** 3), 2)
    config["swap_used_gb"] = round(swap_memory.used / (1024 ** 3), 2)

    disk_usage = psutil.disk_usage("/")
    config["disk_total_gb"] = round(disk_usage.total / (1024 ** 3), 2)
    config["disk_used_gb"] = round(disk_usage.used / (1024 ** 3), 2)
    config["disk_free_gb"] = round(disk_usage.free / (1024 ** 3), 2)

    config["torch_version"] = torch.__version__
    config["cuda_available"] = torch.cuda.is_available()
    config["mps_available"] = torch.backends.mps.is_available()

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        config["gpu_count"] = gpu_count
        config["gpu_devices"] = []
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            config["gpu_devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "cuda_capability": f"{props.major}.{props.minor}",
            })

    return config


def recommend_config(system_config):
    """根据系统配置推荐最佳训练参数"""
    total_mem = system_config["total_memory_gb"]
    available_mem = system_config["available_memory_gb"]
    total_cpu = system_config["physical_cpu_cores"]
    is_cuda = system_config["cuda_available"]
    is_mps = system_config["mps_available"]

    safe_mem = min(total_mem * MEMORY_SAFETY_MARGIN, available_mem) - 1.0
    safe_mem = max(safe_mem, 1.0)

    safe_cpu = max(1, int(total_cpu * CPU_SAFETY_MARGIN))

    recommended = {
        "max_memory_gb": round(safe_mem, 2),
        "max_cpu_cores": safe_cpu,
    }

    if safe_mem >= 8:
        recommended["model"] = "resnet50"
        recommended["image_size"] = 224
        recommended["batch_size"] = 8
        recommended["mode"] = "full"
    elif safe_mem >= 4:
        recommended["model"] = "efficientnet_b0"
        recommended["image_size"] = 196
        recommended["batch_size"] = 8
        recommended["mode"] = "normal"
    elif safe_mem >= 2:
        recommended["model"] = "mobilenetv2"
        recommended["image_size"] = 160
        recommended["batch_size"] = 4
        recommended["mode"] = "light"
    elif safe_mem >= 1:
        recommended["model"] = "mobilenetv2"
        recommended["image_size"] = 128
        recommended["batch_size"] = 2
        recommended["mode"] = "tiny"
    else:
        recommended["model"] = "mobilenetv2"
        recommended["image_size"] = 96
        recommended["batch_size"] = 1
        recommended["mode"] = "ultra_tiny"

    if is_cuda:
        gpu_mem = system_config["gpu_devices"][0]["total_memory_gb"]
        recommended["max_memory_gb"] = min(recommended["max_memory_gb"], gpu_mem)
        recommended["device"] = "cuda"
        if recommended["batch_size"] < 8 and gpu_mem >= 4:
            recommended["batch_size"] = min(16, recommended["batch_size"] * 2)
    elif is_mps:
        recommended["device"] = "mps"
    else:
        recommended["device"] = "cpu"
        recommended["batch_size"] = max(1, recommended["batch_size"] // 2)

    return recommended


def detect_device(preferred_device=None):
    """自动检测可用设备"""
    if preferred_device is not None:
        if preferred_device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"使用指定设备: CUDA ({torch.cuda.get_device_name(0)})")
            return device, "cuda"
        elif preferred_device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info(f"使用指定设备: MPS")
            return device, "mps"
        elif preferred_device == "cpu":
            device = torch.device("cpu")
            logger.info(f"使用指定设备: CPU")
            return device, "cpu"
        else:
            logger.warning(f"指定的设备 {preferred_device} 不可用，将自动检测")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            logger.info(f"检测到 CUDA GPU {i}: {gpu_name}, 显存: {gpu_mem:.2f}GB")
        device = torch.device("cuda")
        logger.info(f"使用设备: CUDA ({torch.cuda.get_device_name(0)})")
        return device, "cuda"

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"使用设备: Apple MPS (Apple Silicon GPU)")
        return device, "mps"

    else:
        device = torch.device("cpu")
        logger.info(f"使用设备: CPU")
        return device, "cpu"


def set_resource_limits(max_cpu_cores, max_memory_gb, device_type="cpu"):
    """设置资源使用限制"""
    cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count() or 4
    actual_cpu = min(max_cpu_cores, cpu_count)

    logger.info(f"设置资源限制:")
    logger.info(f"  CPU核心数: {actual_cpu}/{cpu_count}")
    logger.info(f"  内存上限: {max_memory_gb}GB")

    os.environ["OMP_NUM_THREADS"] = str(actual_cpu)
    os.environ["MKL_NUM_THREADS"] = str(actual_cpu)
    os.environ["OPENBLAS_NUM_THREADS"] = str(actual_cpu)
    os.environ["BLAS_NUM_THREADS"] = str(actual_cpu)
    os.environ["NUMEXPR_NUM_THREADS"] = str(actual_cpu)
    os.environ["TORCH_NUM_THREADS"] = str(actual_cpu)

    torch.set_num_threads(actual_cpu)

    if device_type == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"  GPU显存: {gpu_mem:.2f}GB")

    return actual_cpu


def check_memory_usage(max_memory_gb, device_type="cpu", device=None):
    """检查当前内存使用情况"""
    if device_type == "cuda" and device is not None:
        gpu_mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        gpu_mem_cached = torch.cuda.memory_reserved(device) / (1024 ** 3)
        logger.info(f"GPU内存: 已分配 {gpu_mem_allocated:.2f}GB, 缓存 {gpu_mem_cached:.2f}GB")

        if gpu_mem_allocated > max_memory_gb * 0.85:
            logger.warning(f"GPU内存使用过高: {gpu_mem_allocated:.2f}GB / {max_memory_gb}GB")
            torch.cuda.empty_cache()
            new_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)
            logger.info(f"清理后GPU内存: {new_mem:.2f}GB")

    else:
        process = psutil.Process()
        mem_info = process.memory_info()
        used_gb = mem_info.rss / (1024 ** 3)

        if used_gb > max_memory_gb * 0.85:
            logger.warning(f"内存使用过高: {used_gb:.2f}GB / {max_memory_gb}GB")
            gc.collect()

            new_mem = psutil.Process().memory_info().rss / (1024 ** 3)
            logger.info(f"GC后内存: {new_mem:.2f}GB")

    return used_gb if device_type != "cuda" else torch.cuda.memory_allocated(device) / (1024 ** 3)


def calculate_batch_size(max_memory_gb, model_name, image_size=224, device_type="cpu"):
    """根据内存上限计算合适的 batch_size"""
    config = MODEL_CONFIGS[model_name]
    base_memory = config["memory_estimate_gb"]

    available_memory = max_memory_gb - base_memory - 0.5

    if available_memory <= 0:
        return 1

    bytes_per_pixel = 4
    bytes_per_image = image_size * image_size * 3 * bytes_per_pixel
    bytes_per_batch = bytes_per_image * config["batch_size"]
    gb_per_batch = bytes_per_batch / (1024 ** 3)

    max_batch = int(available_memory / gb_per_batch)
    batch_size = max(min(max_batch, config["batch_size"]), 1)

    if device_type == "cuda":
        batch_size = min(batch_size, 64)
        logger.info(f"GPU模式下，{config['name']} 推荐batch_size: {batch_size}")

    return batch_size


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


def create_filtered_dataset_dir(source_dir, valid_characters, temp_dir):
    """创建只包含有效角色的临时数据集目录"""
    temp_dir.mkdir(parents=True, exist_ok=True)

    for char_name in valid_characters:
        src_char_dir = source_dir / char_name
        dst_char_dir = temp_dir / char_name

        if not src_char_dir.exists():
            continue

        dst_char_dir.mkdir(exist_ok=True)

        for img_file in src_char_dir.glob("*.jpg"):
            try:
                os.link(str(img_file), str(dst_char_dir / img_file.name))
            except OSError:
                shutil.copy2(str(img_file), str(dst_char_dir / img_file.name))

        for img_file in src_char_dir.glob("*.jpeg"):
            try:
                os.link(str(img_file), str(dst_char_dir / img_file.name))
            except OSError:
                shutil.copy2(str(img_file), str(dst_char_dir / img_file.name))

        for img_file in src_char_dir.glob("*.png"):
            try:
                os.link(str(img_file), str(dst_char_dir / img_file.name))
            except OSError:
                shutil.copy2(str(img_file), str(dst_char_dir / img_file.name))

    logger.info(f"✅ 已创建临时数据集目录: {temp_dir}")


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

    return model


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, patience, max_memory_gb, device, device_type, gradient_accumulation_steps=4):
    """训练模型"""
    since = time.time()

    best_model_wts = model.state_dict().copy()
    best_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 30)

        check_memory_usage(max_memory_gb, device_type, device)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            step_counter = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss = loss / gradient_accumulation_steps
                        loss.backward()
                        step_counter += 1

                        if step_counter % gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            check_memory_usage(max_memory_gb, device_type, device)

                running_loss += loss.item() * inputs.size(0) * gradient_accumulation_steps
                running_corrects += torch.sum(preds == labels.data)

                del inputs, labels, outputs, preds, loss
                gc.collect()
                if device_type == "cuda":
                    torch.cuda.empty_cache()

            if phase == "train" and step_counter % gradient_accumulation_steps != 0:
                optimizer.step()

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


def train(model_name, data_dir=None, model_dir=None, max_cpu_cores=1, max_memory_gb=2, batch_size=None, image_size=224, device_type="cpu", device=None):
    """训练指定模型"""
    if data_dir is None:
        data_dir = DATASET_PATH
    if model_dir is None:
        model_dir = MODEL_SAVE_PATH / f"{model_name}_trained"

    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    set_resource_limits(max_cpu_cores, max_memory_gb, device_type)

    if batch_size is None:
        batch_size = calculate_batch_size(max_memory_gb, model_name, image_size, device_type)

    config = MODEL_CONFIGS[model_name]

    logger.info("=" * 60)
    logger.info(f"🎬 训练 {config['name']} 模型")
    logger.info(f"平台: {get_platform()}")
    logger.info(f"设备: {device_type.upper()}")
    logger.info(f"资源限制: CPU={max_cpu_cores}核, 内存={max_memory_gb}GB, Batch={batch_size}, ImageSize={image_size}")
    logger.info("=" * 60)

    logger.info(f"\n📦 收集有效角色 (>= {MIN_IMAGES_PER_CHARACTER} 张图片)...")
    valid_characters, character_stats = collect_valid_characters(data_dir)

    if not valid_characters:
        logger.error("没有找到足够图片的角色，训练终止")
        return None

    logger.info(f"\n✅ 找到 {len(valid_characters)} 个有效角色")
    logger.info(f"总图片数: {sum(character_stats.values())}")

    logger.info(f"\n🔄 创建临时数据集目录...")
    temp_dir = PROJECT_ROOT / "data" / f"temp_filtered_dataset_{model_name}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    create_filtered_dataset_dir(data_dir, valid_characters, temp_dir)

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

    full_dataset = ImageFolder(str(temp_dir), transform=train_transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform

    logger.info(f"数据集: {len(full_dataset)} 样本, {len(full_dataset.classes)} 类别")
    logger.info(f"训练集: {train_size}, 验证集: {val_size}")
    logger.info(f"角色列表: {full_dataset.classes}")

    num_workers = 0 if device_type == "mps" or get_platform() == "Windows" else min(max_cpu_cores, 2)
    pin_memory = device_type == "cuda"

    dataloaders = {
        "train": DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        ),
        "val": DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        ),
    }

    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    with open(model_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(
            {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"✅ 保存类别映射")

    num_classes = len(full_dataset.classes)
    model = create_model(model_name, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    gradient_accumulation_steps = max(1, 32 // batch_size)

    try:
        logger.info(f"\n🚀 开始训练 {config['name']}, 梯度累积步数: {gradient_accumulation_steps}")
        model, best_acc, history = train_model(
            model, dataloaders, criterion, optimizer, scheduler, NUM_EPOCHS, PATIENCE, max_memory_gb, device, device_type, gradient_accumulation_steps
        )

        model = model.to("cpu")
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
            "device": str(device),
            "device_type": device_type,
            "platform": get_platform(),
            "num_classes": num_classes,
            "train_samples": train_size,
            "val_samples": val_size,
            "characters": full_dataset.classes,
            "character_stats": character_stats,
            "history": history,
            "max_cpu_cores": max_cpu_cores,
            "max_memory_gb": max_memory_gb,
            "gradient_accumulation_steps": gradient_accumulation_steps,
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

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"✅ 清理临时数据集目录: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(description="二次元角色识别模型训练")
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "all", "mobilenetv2", "efficientnet_b0", "resnet50"],
        help="要训练的模型 (默认: auto 自动选择)",
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
    parser.add_argument(
        "--max-cpu", type=int, default=None, help="最大CPU核心数 (默认: 自动检测)"
    )
    parser.add_argument(
        "--max-memory", type=float, default=None, help="最大内存使用(GB) (默认: 自动检测)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="手动指定batch_size (默认: 自动计算)"
    )
    parser.add_argument(
        "--image-size", type=int, default=None, help="图片尺寸 (默认: 自动计算)"
    )
    parser.add_argument(
        "--low-memory", action="store_true", help="低内存模式"
    )
    parser.add_argument(
        "--tiny", action="store_true", help="极小内存模式"
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="指定设备 (默认: 自动检测)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="仅显示推荐配置，不实际训练"
    )

    args = parser.parse_args()

    global MIN_IMAGES_PER_CHARACTER
    MIN_IMAGES_PER_CHARACTER = args.min_images

    logger.info("\n" + "=" * 60)
    logger.info("🔍 系统配置检测")
    logger.info("=" * 60)

    system_config = detect_system_config()

    logger.info(f"平台: {system_config['platform']}")
    logger.info(f"Python版本: {system_config['python_version']}")
    logger.info(f"PyTorch版本: {system_config['torch_version']}")
    logger.info(f"\nCPU核心数: {system_config['physical_cpu_cores']} (物理) / {system_config['logical_cpu_cores']} (逻辑)")
    logger.info(f"内存: {system_config['used_memory_gb']}/{system_config['total_memory_gb']}GB (可用: {system_config['available_memory_gb']}GB)")
    logger.info(f"内存使用率: {system_config['memory_usage_percent']}%")
    logger.info(f"磁盘: {system_config['disk_used_gb']}/{system_config['disk_total_gb']}GB (剩余: {system_config['disk_free_gb']}GB)")

    if system_config["cuda_available"]:
        logger.info(f"\nCUDA可用: ✅")
        for gpu in system_config["gpu_devices"]:
            logger.info(f"  GPU {gpu['index']}: {gpu['name']}, 显存: {gpu['total_memory_gb']}GB")
    else:
        logger.info(f"\nCUDA可用: ❌")

    if system_config["mps_available"]:
        logger.info(f"MPS可用: ✅ (Apple Silicon GPU)")
    else:
        logger.info(f"MPS可用: ❌")

    recommended = recommend_config(system_config)

    logger.info("\n" + "=" * 60)
    logger.info("🤖 推荐配置")
    logger.info("=" * 60)
    logger.info(f"模式: {recommended['mode']}")
    logger.info(f"推荐模型: {MODEL_CONFIGS[recommended['model']]['name']}")
    logger.info(f"推荐设备: {recommended['device'].upper()}")
    logger.info(f"推荐CPU核心数: {recommended['max_cpu_cores']}")
    logger.info(f"推荐内存上限: {recommended['max_memory_gb']}GB")
    logger.info(f"推荐Batch Size: {recommended['batch_size']}")
    logger.info(f"推荐图片尺寸: {recommended['image_size']}x{recommended['image_size']}")

    if args.dry_run:
        logger.info("\n💡 使用 --dry-run 参数，仅显示配置，不进行训练")
        return

    device, device_type = detect_device(args.device or recommended["device"])

    max_cpu = args.max_cpu if args.max_cpu is not None else recommended["max_cpu_cores"]
    max_memory = args.max_memory if args.max_memory is not None else recommended["max_memory_gb"]
    batch_size = args.batch_size if args.batch_size is not None else recommended["batch_size"]
    image_size = args.image_size if args.image_size is not None else recommended["image_size"]

    if device_type == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        max_memory = min(max_memory, gpu_mem)

    if args.tiny:
        max_cpu = 1
        max_memory = 1
        batch_size = 1
        image_size = 96
        logger.info("⚠️ 启用极小内存模式")
    elif args.low_memory:
        max_cpu = 1
        max_memory = 2
        batch_size = 2
        image_size = 128
        logger.info("⚠️ 启用低内存模式")

    if args.model == "auto":
        model_name = recommended["model"]
        models_to_train = [model_name]
        logger.info(f"\n✅ 自动选择模型: {MODEL_CONFIGS[model_name]['name']}")
    elif args.model == "all":
        models_to_train = ["mobilenetv2", "efficientnet_b0", "resnet50"]
    else:
        models_to_train = [args.model]

    logger.info(f"\n📋 最终训练配置:")
    logger.info(f"  模型: {', '.join([MODEL_CONFIGS[m]['name'] for m in models_to_train])}")
    logger.info(f"  设备: {device_type.upper()}")
    logger.info(f"  CPU: {max_cpu}核")
    logger.info(f"  内存: {max_memory}GB")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  图片尺寸: {image_size}x{image_size}")
    logger.info(f"  数据集: {args.data}")

    all_results = {}

    for model_name in models_to_train:
        try:
            result = train(model_name, args.data, args.output, max_cpu, max_memory, batch_size, image_size, device_type, device)
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