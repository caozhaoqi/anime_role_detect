#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量续训脚本：冻结 EfficientNet-B3 骨干，仅训练 classifier head。
适合在资源受限环境下快速完成训练。
"""

import os, sys, json, time, copy, logging, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "final_dataset"
MODEL_DIR = PROJECT_ROOT / "models" / "efficientnet_b3"


class FilteredImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=160)
    args = parser.parse_args()

    IMG_SIZE = args.img_size
    BATCH = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}, img={IMG_SIZE}, batch={BATCH}, epochs={EPOCHS}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 数据加载 ──
    valid_classes = []
    for cn in sorted(os.listdir(str(DATA_DIR))):
        cd = os.path.join(str(DATA_DIR), cn)
        if not os.path.isdir(cd): continue
        n = len([f for f in os.listdir(cd) if f.lower().endswith(('.jpg','.png','.jpeg','.webp'))])
        if n >= 30: valid_classes.append(cn)

    class_to_idx = {c: i for i, c in enumerate(sorted(valid_classes))}
    num_classes = len(class_to_idx)
    logger.info(f"Classes: {num_classes}")

    all_samples = []
    for cn, ci in class_to_idx.items():
        cd = os.path.join(str(DATA_DIR), cn)
        for fn in sorted(os.listdir(cd)):
            if fn.lower().endswith(('.jpg','.png','.jpeg','.webp')):
                all_samples.append((os.path.join(cd, fn), ci))

    random.seed(42)
    random.shuffle(all_samples)
    split = int(len(all_samples) * 0.85)
    train_samples = all_samples[:split]
    val_samples = all_samples[split:]
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # ── Transforms ──
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = FilteredImageDataset(train_samples, train_tf)
    val_ds = FilteredImageDataset(val_samples, val_tf)

    # Weighted sampler from samples (not dataset iteration)
    class_counts = {}
    for _, l in train_samples:
        class_counts[l] = class_counts.get(l, 0) + 1
    weights = [1.0 / class_counts[l] for _, l in train_samples]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    # ── 模型 ──
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 768),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(768),
        nn.Dropout(p=0.15),
        nn.Linear(768, num_classes),
    )

    # ── 续训：加载已有 checkpoint ──
    ckpt_path = MODEL_DIR / "model_best.pth"
    start_epoch = 0
    best_acc = 0.0
    metrics_history = []

    if ckpt_path.exists():
        logger.info(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        # 加载已有 metrics
        tr_path = MODEL_DIR / "training_results.json"
        if tr_path.exists():
            with open(tr_path) as f:
                metrics_history = json.load(f).get("metrics", [])
        logger.info(f"Resumed at epoch {start_epoch}, best_acc={best_acc:.4%}")
    else:
        logger.info("No checkpoint found, starting fresh")

    # ── 解冻策略：classifier + 最后2个 features block ──
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    # 解冻 features.7 和 features.8（EfficientNet-B3 最后两个 block）
    for name, param in model.named_parameters():
        if name.startswith("features.7") or name.startswith("features.8"):
            param.requires_grad = True

    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Params: {total} total, {trainable} trainable (classifier + last 2 blocks)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 对解冻的骨干层使用更小的学习率
    param_groups = [
        {"params": model.classifier.parameters(), "lr": LR},
        {"params": [p for n, p in model.named_parameters() if n.startswith(("features.7", "features.8"))], "lr": LR * 0.1},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # ── 训练循环 ──
    best_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    logger.info("=" * 50)
    logger.info(f"Training {EPOCHS} epochs (frozen backbone)")
    logger.info("=" * 50)

    t0 = time.time()

    for epoch in range(EPOCHS):
        ep_num = start_epoch + epoch
        # ── Train ──
        model.train()
        tloss, tcorrect, ttotal = 0.0, 0, 0
        for bi, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = outputs.max(1)
            tloss += loss.item() * inputs.size(0)
            tcorrect += (preds == labels).sum().item()
            ttotal += inputs.size(0)
            if bi % 20 == 0:
                logger.info(f"  E{ep_num+1} B{bi}/{len(train_loader)} loss={loss.item():.3f}")

        train_loss = tloss / ttotal
        train_acc = tcorrect / ttotal

        # ── Val ──
        model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                _, preds = outputs.max(1)
                vloss += loss.item() * inputs.size(0)
                vcorrect += (preds == labels).sum().item()
                vtotal += inputs.size(0)

        val_loss = vloss / vtotal
        val_acc = vcorrect / vtotal
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"E{ep_num+1}: train_loss={train_loss:.4f} acc={train_acc:.4%} | val_loss={val_loss:.4f} acc={val_acc:.4%} | lr={lr:.6f}")

        metrics_history.append({
            "epoch": ep_num + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": lr,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(f"  *** Best! {best_acc:.4%} ***")
            # Save checkpoint
            torch.save({
                "epoch": ep_num,
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc,
                "class_to_idx": class_to_idx,
            }, MODEL_DIR / "model_best.pth")
            logger.info(f"  Saved model_best.pth")
        else:
            patience_counter += 1

    # ── 保存最终结果 ──
    model.load_state_dict(best_wts)
    elapsed = time.time() - t0

    # model_full.pth
    torch.save({
        "epoch": start_epoch + EPOCHS,
        "model_state_dict": model.state_dict(),
        "best_acc": best_acc,
        "class_to_idx": class_to_idx,
    }, MODEL_DIR / "model_full.pth")

    # class_to_idx.json
    with open(MODEL_DIR / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

    # training_results.json
    results = {
        "model_name": "efficientnet_b3",
        "architecture": "EfficientNet-B3 (frozen backbone, classifier-only training)",
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "class_names": sorted(class_to_idx.keys()),
        "image_size": IMG_SIZE,
        "best_accuracy": float(best_acc),
        "training_time_seconds": elapsed,
        "training_config": {
            "image_size": IMG_SIZE, "batch_size": BATCH, "epochs": EPOCHS,
            "lr": LR, "frozen_backbone": True, "device": str(DEVICE),
        },
        "metrics": metrics_history,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(MODEL_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"Best accuracy: {best_acc:.4%}")
    logger.info(f"Files: model_best.pth, model_full.pth, class_to_idx.json, training_results.json")
    return best_acc


if __name__ == "__main__":
    main()
