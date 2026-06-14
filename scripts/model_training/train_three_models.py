#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充训练数据 - 检查并补充缺失的图片
同时训练三种模型：MobileNetV2、ResNet50、EfficientNet-B0
"""
import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pathlib import Path
from PIL import Image
import logging
import time
import copy
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入通知服务
try:
    from src.services.notification_service import send_notification
    NOTIFICATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 通知服务导入失败: {e}")
    NOTIFICATION_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 配置路径
DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
SOURCE_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")

# 训练配置
IMAGE_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
PATIENCE = 10

# 允许加载截断的图片
Image.MAX_IMAGE_PIXELS = None


def supplement_missing_images():
    """补充缺失的图片"""
    logger.info("=" * 70)
    logger.info("开始补充缺失图片")
    logger.info("=" * 70)

    # 检查训练目录中每个角色的图片数
    training_chars = {}
    for char_dir in DATA_DIR.iterdir():
        if char_dir.is_dir():
            img_count = len([
                f for f in char_dir.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
            ])
            training_chars[char_dir.name] = {
                "count": img_count,
                "path": char_dir
            }

    # 检查源目录中有哪些角色可以补充
    source_chars = {}
    for char_dir in SOURCE_DIR.iterdir():
        if char_dir.is_dir():
            img_count = len([
                f for f in char_dir.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
            ])
            source_chars[char_dir.name] = {
                "count": img_count,
                "path": char_dir
            }

    logger.info(f"训练集角色数: {len(training_chars)}")
    logger.info(f"源数据集角色数: {len(source_chars)}")

    # 补充缺失的角色
    missing_chars = set(source_chars.keys()) - set(training_chars.keys())
    if missing_chars:
        logger.info(f"\n发现 {len(missing_chars)} 个缺失角色，补充中...")
        for char_name in missing_chars:
            source_path = source_chars[char_name]["path"]
            target_path = DATA_DIR / char_name
            
            # 如果目标目录已存在，跳过
            if target_path.exists():
                logger.info(f"  ⏭️ 跳过: {char_name} (目录已存在)")
                continue
                
            # 只复制有图片的目录
            if source_chars[char_name]["count"] > 0:
                shutil.copytree(source_path, target_path)
                logger.info(f"  ✅ 补充角色: {char_name} ({source_chars[char_name]['count']} 张图片)")
            else:
                logger.info(f"  ⚠️ 跳过: {char_name} (源目录为空)")

    # 统计图片不足的角色
    insufficient_chars = {name: info for name, info in training_chars.items() if info["count"] < 100}
    if insufficient_chars:
        logger.info(f"\n发现 {len(insufficient_chars)} 个角色图片不足100张，补充中...")
        for char_name, info in insufficient_chars.items():
            if char_name in source_chars:
                source_path = source_chars[char_name]["path"]
                target_path = info["path"]

                # 获取已存在的图片文件名
                existing_files = set(f.name for f in target_path.iterdir() if f.is_file())

                # 从源目录补充新图片
                supplemented_count = 0
                for img_file in source_path.iterdir():
                    if (img_file.is_file() and
                        img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"] and
                        img_file.name not in existing_files):

                        # 检查图片是否损坏
                        try:
                            with Image.open(img_file) as img:
                                img.load()
                            shutil.copy2(img_file, target_path / img_file.name)
                            supplemented_count += 1
                            existing_files.add(img_file.name)
                        except:
                            pass

                        # 达到100张后停止
                        if len(existing_files) >= 100:
                            break

                if supplemented_count > 0:
                    logger.info(f"  ✅ 补充 {char_name}: +{supplemented_count} 张 (现有 {len(existing_files)} 张)")

    # 清理损坏的图片
    logger.info("\n清理损坏的图片...")
    total_deleted = 0
    for char_dir in DATA_DIR.iterdir():
        if char_dir.is_dir():
            for img_file in char_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    try:
                        with Image.open(img_file) as img:
                            img.load()
                    except:
                        img_file.unlink()
                        total_deleted += 1
                        logger.info(f"  🗑️ 删除损坏文件: {char_dir.name}/{img_file.name}")

    if total_deleted > 0:
        logger.info(f"已删除 {total_deleted} 个损坏文件")

    logger.info("\n" + "=" * 70)
    logger.info("图片补充完成")
    logger.info("=" * 70)


def get_device():
    """获取最佳设备"""
    if torch.backends.mps.is_available():
        logger.info("✅ 使用 MPS 加速")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("✅ 使用 CUDA 加速")
        return torch.device("cuda")
    else:
        logger.info("⚠️ 使用 CPU")
        return torch.device("cpu")


def create_model(model_name, num_classes, device):
    """创建模型"""
    if model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "EfficientNet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"未知模型: {model_name}")

    return model.to(device)


def train_model(model, dataloaders, dataset_sizes, device, model_name):
    """训练单个模型"""
    logger.info(f"\n{'='*70}")
    logger.info(f"开始训练 {model_name}")
    logger.info(f"{'='*70}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        logger.info(f"\n{model_name} - Epoch {epoch+1}/{NUM_EPOCHS}")

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"  Batch {batch_idx+1}, Loss: {loss.item():.4f}")

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            logger.info(f"  {phase.capitalize()} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            if phase == 'val':
                scheduler.step(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0

                    # 保存最佳模型
                    save_path = MODEL_DIR / f"{model_name.lower()}_best.pth"
                    torch.save(best_model_wts, save_path)
                    logger.info(f"  ✅ 保存最佳模型: {save_path}")
                else:
                    patience_counter += 1

                if patience_counter >= PATIENCE:
                    logger.info(f"  ⏰ 早停: 验证准确率连续 {PATIENCE} 轮未提升")
                    break

        if patience_counter >= PATIENCE:
            break

    # 加载最佳模型
    model.load_state_dict(best_model_wts)

    # 保存最终模型
    final_path = MODEL_DIR / f"{model_name.lower()}_final.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"\n{model_name} 训练完成！最佳准确率: {best_acc:.4f}")
    logger.info(f"模型保存路径: {final_path}")

    return best_acc


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("二次元角色识别模型训练")
    logger.info("训练三种模型: MobileNetV2, ResNet50, EfficientNet")
    logger.info("=" * 70)

    # 发送开始通知
    if NOTIFICATION_AVAILABLE:
        send_notification(
            "🚀 模型训练任务开始",
            f"数据集: {DATA_DIR}\n配置: {NUM_EPOCHS} 轮, 批次 {BATCH_SIZE}, 学习率 {LEARNING_RATE}\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # 补充图片
    supplement_missing_images()

    # 准备数据
    logger.info("\n准备数据...")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(DATA_DIR, transform=transform)
    num_classes = len(dataset.classes)

    logger.info(f"  类别数: {num_classes}")
    logger.info(f"  图片总数: {len(dataset)}")

    # 划分训练集和验证集
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }
    dataset_sizes = {'train': train_size, 'val': val_size}

    device = get_device()

    # 训练三种模型
    models_to_train = ["MobileNetV2", "ResNet50", "EfficientNet"]
    results = {}

    for model_name in models_to_train:
        logger.info(f"\n{'#'*70}")
        logger.info(f"## 训练模型: {model_name}")
        logger.info(f"{'#'*70}")

        model = create_model(model_name, num_classes, device)
        best_acc = train_model(model, dataloaders, dataset_sizes, device, model_name)
        results[model_name] = best_acc

        # 发送单个模型训练完成通知
        if NOTIFICATION_AVAILABLE:
            send_notification(
                f"✅ {model_name} 训练完成",
                f"最佳准确率: {best_acc:.4f}\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        # 清理内存
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 输出总结
    logger.info("\n" + "=" * 70)
    logger.info("所有模型训练完成！")
    logger.info("=" * 70)
    logger.info("\n模型性能对比:")
    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {model_name}: {acc:.4f}")

    best_model = max(results.items(), key=lambda x: x[1])
    logger.info(f"\n🏆 最佳模型: {best_model[0]} (准确率: {best_model[1]:.4f})")
    logger.info(f"\n模型保存目录: {MODEL_DIR}")
    logger.info("=" * 70)

    # 发送完成通知
    if NOTIFICATION_AVAILABLE:
        summary = f"""训练配置:
- 数据集: {num_classes} 个角色, {len(dataset)} 张图片
- 轮次: {NUM_EPOCHS}
- 批次: {BATCH_SIZE}

训练结果:
{chr(10).join([f"- {name}: {acc:.4f}" for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True)])}

🏆 最佳模型: {best_model[0]} (准确率: {best_model[1]:.4f})
📁 模型保存路径: {MODEL_DIR}

时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
        send_notification("🎉 训练任务完成", summary)


if __name__ == "__main__":
    main()