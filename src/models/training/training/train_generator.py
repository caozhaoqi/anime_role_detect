#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import logging
from tqdm import tqdm
import json

# 
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_generator")


class CharacterDataset(Dataset):
    """"""

    def __init__(self, root_dir, transform=None, target_characters=None):
        """

        Args:
            root_dir: 
            transform: 
            target_characters: 
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        # 
        all_classes = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )

        # 
        if target_characters:
            classes = [c for c in all_classes if any(tc in c for tc in target_characters)]
        else:
            classes = all_classes

        idx = 0
        for cls in classes:
            self.class_to_idx[cls] = idx
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)
            logger.info(
                f" {cls}: {len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])} "
            )
            idx += 1

        logger.info(
            f" {len(self.class_to_idx)} {len(self.images)} "
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class FeatureExtractor(nn.Module):
    """"""

    def __init__(self, model_type="mobilenet_v2", num_classes=2):
        super().__init__()
        if model_type == "mobilenet_v2":
            self.base_model = models.mobilenet_v2(pretrained=True)
            # 
            self.feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        elif model_type == "efficientnet_b0":
            self.base_model = models.efficientnet_b0(pretrained=True)
            self.feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        elif model_type == "resnet18":
            self.base_model = models.resnet18(pretrained=True)
            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError(f": {model_type}")

    def forward(self, x):
        return self.base_model(x)


class DiffusionGenerator(nn.Module):
    """"""

    def __init__(self, feature_dim, image_size=(224, 224), num_channels=3):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels

        # 
        self.feature_proj = nn.Linear(feature_dim, 256)

        # UNet
        self.unet = nn.Sequential(
            # 
            nn.Conv2d(num_channels * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            # 
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # 0-1
        )

    def forward(self, x, noise, features):
        # 
        batch_size = x.size(0)

        # 
        return self.unet(torch.cat([x, noise], dim=1))


def diffusion_loss(generator, feature_extractor, images, labels, device):
    """"""
    batch_size = images.size(0)

    # 
    noise = torch.randn_like(images).to(device)

    # 
    with torch.no_grad():
        features = feature_extractor(images)

    # 
    generated = generator(images, noise, features)

    # 
    loss = nn.MSELoss()(generated, images)

    # 
    with torch.no_grad():
        gen_features = feature_extractor(generated)
    feature_loss = nn.MSELoss()(gen_features, features)

    return loss + 0.1 * feature_loss


def train_generator(
    generator,
    feature_extractor,
    train_loader,
    device,
    num_epochs=50,
    lr=0.0001,
    output_dir="models/generator",
):
    """"""
    logger.info(f": {device}")

    # 
    os.makedirs(output_dir, exist_ok=True)

    # 
    optimizer = optim.Adam(generator.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 
    losses = []
    best_loss = float("inf")

    # 
    patience = 10
    no_improve_count = 0

    for epoch in range(num_epochs):
        generator.train()
        feature_extractor.eval()  # 

        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images = images.to(device)

            optimizer.zero_grad()
            loss = diffusion_loss(generator, feature_extractor, images, labels, device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss: {avg_loss:.4f}")

        # 
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "generator_state_dict": generator.state_dict(),
                    "feature_extractor_state_dict": feature_extractor.state_dict(),
                    "loss": best_loss,
                },
                os.path.join(output_dir, "generator_best.pth"),
            )
            logger.info(f": {best_loss:.4f}")
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 
        if no_improve_count >= patience:
            logger.info(f": {patience} epoch")
            break

    # 
    results = {
        "num_epochs": num_epochs,
        "best_loss": best_loss,
        "final_loss": losses[-1],
        "losses": losses,
    }

    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f": {best_loss:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data-dir", type=str, default="data/train", help="")
    parser.add_argument(
        "--model-type",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "efficientnet_b0", "resnet18"],
        help="",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="")
    parser.add_argument("--epochs", type=int, default=50, help="")
    parser.add_argument("--lr", type=float, default=0.0001, help="")
    parser.add_argument("--output-dir", type=str, default="models/generator", help="")

    args = parser.parse_args()

    # 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f": {device}")

    # 
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 
    target_characters = ["_", "_"]

    # 
    logger.info("...")
    dataset = CharacterDataset(
        args.data_dir, transform=transform, target_characters=target_characters
    )

    # 
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    logger.info(f": {len(dataset)}")

    # 
    feature_extractor = FeatureExtractor(args.model_type)
    feature_extractor = feature_extractor.to(device)

    generator = DiffusionGenerator(feature_extractor.feature_dim)
    generator = generator.to(device)

    # 
    for param in feature_extractor.parameters():
        param.requires_grad = False

    # 
    results = train_generator(
        generator,
        feature_extractor,
        train_loader,
        device,
        num_epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
    )

    logger.info("")
    logger.info(f': {results["best_loss"]:.4f}')


if __name__ == "__main__":
    main()
