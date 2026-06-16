import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision import models
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict, Counter

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ sklearn未安装，将跳过混淆矩阵")

# 支持的模型列表
MODEL_ZOO = {
    "mobilenetv2": models.mobilenet_v2,
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b3": models.efficientnet_b3,
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
}


def get_device():
    """获取最佳可用设备"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_augmentations(augment_level="high", image_size=224):
    """获取数据增强变换"""
    base_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if augment_level == "low":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif augment_level == "medium":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-15, 15)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:  # high
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=(-20, 20)),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
                ),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            ]
        )

    return train_transform, base_transform


def rand_bbox(size, lam):
    """生成随机边界框"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets, alpha=1.0):
    """CutMix 数据增强"""
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    return data, targets, shuffled_targets, lam


def mixup_data(data, targets, alpha=1.0):
    """Mixup 数据增强"""
    lam = np.random.beta(alpha, alpha)
    batch_size = data.size(0)
    index = torch.randperm(batch_size)
    mixed_data = lam * data + (1 - lam) * data[index]
    targets_a, targets_b = targets, targets[index]
    return mixed_data, targets_a, targets_b, lam


def compute_topk_accuracy(outputs, labels, k=(1, 3, 5)):
    """计算 Top-1, Top-3, Top-5 准确率"""
    maxk = max(k)
    batch_size = labels.size(0)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    results = {}
    for k_val in k:
        if k_val <= outputs.size(1):
            correct_k = correct[:k_val].reshape(-1).float().sum(0, keepdim=True)
            results[f"top{k_val}"] = correct_k.item() / batch_size
    return results


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, args):
    """训练模型"""
    best_acc = 0.0
    train_history = []
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None

    # 混淆矩阵收集
    all_val_preds = []
    all_val_labels = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = {k: 0.0 for k in [1, 3, 5]}
            total_samples = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase}"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0)
                total_samples += batch_size

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # 数据增强选择：随机选 mixup / cutmix / 无增强
                    use_aug = phase == "train" and (args.mixup or args.cutmix)
                    if use_aug:
                        aug_choice = random.choice(["mixup", "cutmix", "none"])
                        if args.mixup and aug_choice == "mixup":
                            use_aug = "mixup"
                        elif args.cutmix and aug_choice == "cutmix":
                            use_aug = "cutmix"
                        else:
                            use_aug = "none"
                    else:
                        use_aug = "none"

                    if use_aug == "mixup":
                        inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, args.mixup_alpha)
                        outputs = model(inputs)
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                        # mixup下不精确计算准确率
                        for k in running_corrects:
                            running_corrects[k] += 0
                    elif use_aug == "cutmix":
                        inputs, labels_a, labels_b, lam = cutmix(inputs, labels)
                        outputs = model(inputs)
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                        preds = torch.argmax(outputs, dim=1)
                        running_corrects[1] += lam * torch.sum(preds == labels_a.data).item() + (
                            1 - lam
                        ) * torch.sum(preds == labels_b.data).item()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        # Top-k 准确率
                        topk = compute_topk_accuracy(outputs, labels, k=(1, 3, 5))
                        for k, v in topk.items():
                            k_val = int(k.replace("top", ""))
                            running_corrects[k_val] += v * batch_size
                        preds = torch.argmax(outputs, dim=1)
                        # val阶段收集预测用于混淆矩阵
                        if phase == "val":
                            all_val_preds.extend(preds.cpu().numpy().tolist())
                            all_val_labels.extend(labels.cpu().numpy().tolist())

                if phase == "train":
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * batch_size

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects[1] / total_samples

            log_msg = f"{phase} Loss: {epoch_loss:.4f} Acc@1: {epoch_acc:.4f}"
            if phase == "val" and use_aug == "none":
                for k in [3, 5]:
                    if running_corrects[k] > 0:
                        log_msg += f" Acc@{k}: {running_corrects[k]/total_samples:.4f}"
            print(log_msg)

            train_history.append({
                "epoch": epoch + 1,
                "phase": phase,
                "loss": epoch_loss,
                "accuracy": epoch_acc,
                "top3": running_corrects.get(3, 0) / total_samples if total_samples > 0 else 0,
                "top5": running_corrects.get(5, 0) / total_samples if total_samples > 0 else 0,
            })

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pth"))

        # 早停检查
        if epoch > 5:
            recent_accs = [h["accuracy"] for h in train_history[-6:-1] if h["phase"] == "val"]
            if len(recent_accs) >= 5 and max(recent_accs) - min(recent_accs) < 0.01:
                print(f"⚠️ 早停触发：连续5轮验证准确率变化小于1%")
                break

    # 混淆矩阵
    if HAS_SKLEARN and len(all_val_preds) > 0 and len(all_val_labels) > 0:
        try:
            cm = confusion_matrix(all_val_labels, all_val_preds)
            # 获取类别名称
            try:
                class_names = dataloaders["val"].dataset.classes
            except AttributeError:
                class_names = [str(i) for i in range(cm.shape[0])]
            n_classes = cm.shape[0]
            plt.figure(figsize=(min(40, n_classes * 1.2), min(40, n_classes * 1.2)))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap="Blues", xticks_rotation=90, values_format="d")
            plt.title(f"{args.model_name} - Confusion Matrix (val set)")
            plt.tight_layout()
            cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
            plt.savefig(cm_path, dpi=150)
            plt.close()
            print(f"📊 混淆矩阵已保存: {cm_path}")
        except Exception as e:
            print(f"⚠️ 混淆矩阵生成失败: {e}")
    elif len(all_val_preds) == 0:
        print("⚠️ 无验证集预测数据，跳过混淆矩阵")

    return model, best_acc, train_history


def main():
    parser = argparse.ArgumentParser(description="高级模型训练脚本")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output_dir", type=str, default="./models", help="输出目录")
    parser.add_argument(
        "--model_name", type=str, default="efficientnet_b0", choices=MODEL_ZOO.keys()
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--augment_level", type=str, default="high", choices=["low", "medium", "high"]
    )
    parser.add_argument("--cutmix", action="store_true", default=False, help="启用 CutMix")
    parser.add_argument("--mixup", action="store_true", default=False, help="启用 MixUp")
    parser.add_argument("--mixup_alpha", type=float, default=0.8, help="MixUp/CutMix alpha 参数")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="标签平滑系数")
    parser.add_argument("--fp16", action="store_true", default=False, help="混合精度训练")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"{args.model_name}_anime_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取设备
    device = get_device()
    print(f"📱 使用设备: {device}")

    # 获取数据变换
    train_transform, val_transform = get_augmentations(args.augment_level, args.image_size)

    # 加载数据集
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes
    else:
        full_dataset = datasets.ImageFolder(args.data_dir, transform=train_transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        val_dataset.dataset.transform = val_transform
        num_classes = len(full_dataset.classes)
        class_names = full_dataset.classes

    print(f"📂 加载数据: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本")
    print(f"📋 类别数量: {num_classes}")

    # 创建数据加载器
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0),
    }

    # 创建模型
    print(f"🧠 使用模型: {args.model_name}")
    model = MODEL_ZOO[args.model_name](num_classes=num_classes)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)

    # 训练模型
    print("🚀 开始训练...")
    model, best_acc, train_history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, device, args
    )

    # 保存结果
    results = {
        "model_name": args.model_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "best_accuracy": best_acc,
        "image_size": args.image_size,
        "augment_level": args.augment_level,
        "cutmix": args.cutmix,
        "label_smoothing": args.label_smoothing,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": len([h for h in train_history if h["phase"] == "train"]),
        "train_history": train_history,
        "timestamp": timestamp,
    }

    with open(os.path.join(args.output_dir, "training_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n🎉 训练完成！最佳准确率: {best_acc:.4f}")
    print(f"📁 模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
