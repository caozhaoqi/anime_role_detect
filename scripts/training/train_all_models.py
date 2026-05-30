#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二次元角色识别模型训练 - 支持多种模型
支持 MobileNetV2, EfficientNet-B0, EfficientNet-B3, ResNet50
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pathlib import Path
import json
import logging
import time
import copy
import os
import sys
import argparse
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_notification_config():
    """加载通知配置文件"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notification_config.json"
    )
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            notification_config = json.load(f)

        os.environ["NOTIFICATION_ENABLED"] = "true"
        os.environ["NOTIFICATION_PLATFORM"] = notification_config["platform"]
        os.environ["FEISHU_APP_ID"] = notification_config["feishu"]["app_id"]
        os.environ["FEISHU_APP_SECRET"] = notification_config["feishu"]["app_secret"]
        os.environ["FEISHU_RECEIVE_ID"] = notification_config["feishu"]["receive_id"]
        os.environ["FEISHU_RECEIVE_ID_TYPE"] = notification_config["feishu"]["receive_id_type"]
        logger.info(f"✅ 已加载通知配置: {config_path}")
    else:
        logger.warning(f"⚠️ 未找到通知配置文件: {config_path}")


load_notification_config()


def get_best_device():
    """获取最佳可用设备"""
    if torch.backends.mps.is_available():
        logger.info("✅ MPS设备可用")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("✅ CUDA设备可用")
        return torch.device("cuda")
    else:
        logger.info("⚠️ 仅CPU可用")
        return torch.device("cpu")


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
    "efficientnet_b3": {
        "name": "EfficientNet-B3",
        "image_size": 300,
        "batch_size": 24,
        "model_fn": models.efficientnet_b3,
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

DEVICE = get_best_device()
NUM_EPOCHS = 50
PATIENCE = 10
EARLY_STOP_THRESHOLD = 0.001
LEARNING_RATE = 1e-4

BASE_DATA_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset"
BASE_MODEL_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/models"


def send_feishu_message(title, message):
    """发送飞书消息"""
    try:
        sys.path.insert(
            0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        from src.services.notification_service import send_notification

        success = send_notification(f"**{title}**\n\n{message}", level="info")
        if success:
            logger.info("✅ 飞书通知发送成功")
            return True
        else:
            logger.warning("❌ 飞书通知发送失败")
            return False
    except Exception as e:
        logger.warning(f"❌ 发送飞书通知失败: {e}")
        return False


class TrainingProgressTracker:
    """训练进度跟踪器 - 每30分钟发送一次进展"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.start_time = time.time()
        self.last_notify_time = self.start_time
        self.notify_interval = 30 * 60  # 30分钟
        self.current_epoch = 0
        self.best_acc = 0.0
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.running = True
        self.lock = threading.Lock()

    def update(self, epoch, train_loss, val_loss, val_acc):
        """更新训练进度"""
        with self.lock:
            self.current_epoch = epoch
            self.train_loss = train_loss
            self.val_loss = val_loss
            if val_acc > self.best_acc:
                self.best_acc = val_acc

    def should_notify(self):
        """检查是否需要发送通知"""
        now = time.time()
        return now - self.last_notify_time >= self.notify_interval

    def send_progress(self, total_epochs):
        """发送训练进展"""
        with self.lock:
            elapsed = time.time() - self.start_time
            elapsed_str = f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"

            progress = (self.current_epoch + 1) / total_epochs * 100
            message = f"""📊 训练进度: {progress:.1f}%
Epoch: {self.current_epoch + 1}/{total_epochs}
⏱️ 已运行: {elapsed_str}
📈 最佳准确率: {self.best_acc:.4%}
📉 训练损失: {self.train_loss:.4f}
📉 验证损失: {self.val_loss:.4f}"""

            send_feishu_message(f"🔄 {MODEL_CONFIGS[self.model_name]['name']} 训练中", message)
            self.last_notify_time = time.time()


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
    elif model_name == "efficientnet_b3":
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


def train_model(model, dataloaders, criterion, optimizer, num_epochs, patience, model_name):
    """训练模型"""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    progress_tracker = TrainingProgressTracker(model_name)

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info("-" * 10)

        current_train_loss = 0.0
        current_val_loss = 0.0
        current_val_acc = 0.0

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

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

                if batch_idx % 100 == 0 and phase == "train":
                    logger.info(
                        f"  Batch {batch_idx}/{len(dataloaders[phase])}, Loss: {loss.item():.4f}"
                    )

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            logger.info(f"  {phase} Loss: {epoch_loss:.4f}, {phase} Acc: {epoch_acc:.4%}")

            if phase == "val":
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())
                current_val_loss = epoch_loss
                current_val_acc = epoch_acc

                if epoch_acc > best_acc + EARLY_STOP_THRESHOLD:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
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
                current_train_loss = epoch_loss

        progress_tracker.update(epoch, current_train_loss, current_val_loss, current_val_acc)

        if progress_tracker.should_notify():
            progress_tracker.send_progress(num_epochs)

    time_elapsed = time.time() - since
    logger.info(f"训练完成，耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    logger.info(f"最佳验证准确率: {best_acc:.4%}")

    model.load_state_dict(best_model_wts)
    return model, best_acc, history


def train(model_name, data_dir=None, model_dir=None):
    """训练指定模型"""
    if data_dir is None:
        data_dir = BASE_DATA_DIR
    if model_dir is None:
        model_dir = os.path.join(BASE_MODEL_DIR, f"{model_name}_loli")

    config = MODEL_CONFIGS[model_name]
    image_size = config["image_size"]
    batch_size = config["batch_size"]

    logger.info("=" * 60)
    logger.info(f"🎬 训练 {config['name']} 模型")
    logger.info("=" * 60)

    send_feishu_message(
        f"🚀 {config['name']} 训练任务开始",
        f"""数据集: {data_dir}
配置:
- Model: {config['name']}
- Epochs: {NUM_EPOCHS}
- Batch Size: {batch_size}
- Image Size: {image_size}
- Learning Rate: {LEARNING_RATE}
- 设备: {DEVICE}

时间: {time.strftime('%Y-%m-%d %H:%M:%S')}""",
    )

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"使用设备: {DEVICE}")
    logger.info(f"加载数据: {data_dir}")

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

    full_dataset = ImageFolder(data_dir, transform=train_transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform

    logger.info(f"数据集: {len(full_dataset)} 样本, {len(full_dataset.classes)} 类别")
    logger.info(f"训练集: {train_size}, 验证集: {val_size}")

    dataloaders = {
        "train": DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        ),
        "val": DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        ),
    }

    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    with open(Path(model_dir) / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(
            {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"✅ 保存类别映射")

    num_classes = len(full_dataset.classes)
    model = create_model(model_name, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    try:
        logger.info(f"\n🚀 开始训练 {config['name']}")
        model, best_acc, history = train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            model_name=model_name,
        )

        torch.save(model.state_dict(), Path(model_dir) / "model_best.pth")
        torch.save(model, Path(model_dir) / "model_full.pth")

        logger.info(f"✅ 模型已保存: {Path(model_dir) / 'model_best.pth'}")

        results = {
            "model": config["name"],
            "model_key": model_name,
            "accuracy": best_acc.item(),
            "epochs": NUM_EPOCHS,
            "batch_size": batch_size,
            "learning_rate": LEARNING_RATE,
            "image_size": image_size,
            "device": str(DEVICE),
            "num_classes": num_classes,
            "train_samples": train_size,
            "val_samples": val_size,
            "history": history,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(Path(model_dir) / "training_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 训练结果已保存")

        send_feishu_message(
            f"🎉 {config['name']} 训练完成",
            f"""训练结果:
- ✅ 最佳验证准确率: {best_acc:.4%}
- 📁 模型保存路径: {model_dir}

时间: {time.strftime('%Y-%m-%d %H:%M:%S')}""",
        )

        return results

    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback

        logger.error(traceback.format_exc())
        send_feishu_message(f"❌ {config['name']} 训练异常", f"训练失败: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="训练角色识别模型")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "mobilenetv2", "efficientnet_b0", "efficientnet_b3", "resnet50"],
        help="要训练的模型 (默认: all)",
    )
    parser.add_argument(
        "--data", type=str, default=BASE_DATA_DIR, help=f"数据集路径 (默认: {BASE_DATA_DIR})"
    )
    parser.add_argument(
        "--output", type=str, default=BASE_MODEL_DIR, help=f"模型输出目录 (默认: {BASE_MODEL_DIR})"
    )

    args = parser.parse_args()

    if args.model == "all":
        models_to_train = ["mobilenetv2", "efficientnet_b0", "efficientnet_b3", "resnet50"]
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

    summary = "\n".join(
        [
            f"- {MODEL_CONFIGS[k]['name']}: {v.get('accuracy', 'N/A') if isinstance(v, dict) else 'Error'}"
            for k, v in all_results.items()
        ]
    )
    send_feishu_message("📊 全部训练任务完成", f"训练结果汇总:\n{summary}")


if __name__ == "__main__":
    main()
