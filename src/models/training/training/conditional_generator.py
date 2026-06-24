#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 - 

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
logger = logging.getLogger("conditional_generator")


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


class ConditionalGenerator(nn.Module):
    """ - """

    def __init__(self, num_classes, latent_dim=100, image_size=(224, 224), num_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_channels = num_channels

        # 
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # 
        self.generator = nn.Sequential(
            # : latent_dim () + latent_dim ()
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            # 
            nn.Linear(2048, image_size[0] * image_size[1] * num_channels),
            nn.Tanh(),  # -11
        )

    def forward(self, noise, labels):
        # 
        label_emb = self.label_embedding(labels)

        # 
        combined = torch.cat([noise, label_emb], dim=1)

        # 
        generated = self.generator(combined)

        # 
        generated = generated.view(-1, self.num_channels, self.image_size[0], self.image_size[1])

        return generated


class ConditionalDiscriminator(nn.Module):
    """ - """

    def __init__(self, num_classes, image_size=(224, 224), num_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_channels = num_channels

        # 
        self.label_embedding = nn.Embedding(num_classes, image_size[0] * image_size[1])

        # 
        self.discriminator = nn.Sequential(
            # :  + 
            nn.Conv2d(num_channels + 1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid(),  # 0-1
        )

    def forward(self, images, labels):
        # 
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(-1, 1, self.image_size[0], self.image_size[1])

        # 
        combined = torch.cat([images, label_emb], dim=1)

        # 
        validity = self.discriminator(combined)

        return validity


class FeatureExtractor(nn.Module):
    """"""

    def __init__(self, model_type="mobilenet_v2", num_classes=2):
        super().__init__()
        if model_type == "mobilenet_v2":
            self.base_model = models.mobilenet_v2(pretrained=True)
            self.feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(self.feature_dim, num_classes)
        elif model_type == "efficientnet_b0":
            self.base_model = models.efficientnet_b0(pretrained=True)
            self.feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(self.feature_dim, num_classes)
        elif model_type == "resnet18":
            self.base_model = models.resnet18(pretrained=True)
            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(self.feature_dim, num_classes)
        else:
            raise ValueError(f": {model_type}")

    def forward(self, x):
        return self.base_model(x)


class ConditionalGAN(nn.Module):
    """GAN - """

    def __init__(
        self,
        num_classes,
        latent_dim=100,
        image_size=(224, 224),
        num_channels=3,
        model_type="mobilenet_v2",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # 
        self.generator = ConditionalGenerator(num_classes, latent_dim, image_size, num_channels)
        self.discriminator = ConditionalDiscriminator(num_classes, image_size, num_channels)
        self.feature_extractor = FeatureExtractor(model_type, num_classes)

    def forward(self, noise, labels, mode="generate"):
        """

        Args:
            noise: 
            labels: 
            mode: 'generate' , 'discriminate' , 'classify' 
        """
        if mode == "generate":
            return self.generator(noise, labels)
        elif mode == "discriminate":
            return self.discriminator(noise, labels)
        elif mode == "classify":
            return self.feature_extractor(noise)
        else:
            raise ValueError(f"mode: {mode}")


def train_conditional_gan(
    model, train_loader, device, num_epochs=100, lr=0.0002, output_dir="models/conditional_gan"
):
    """GAN"""
    logger.info(f"GAN: {device}")

    # 
    os.makedirs(output_dir, exist_ok=True)

    # 
    optimizer_g = optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_c = optim.Adam(model.feature_extractor.parameters(), lr=lr * 0.1)

    # 
    adversarial_loss = nn.BCELoss()
    classification_loss = nn.CrossEntropyLoss()

    # 
    g_losses = []
    d_losses = []
    c_losses = []

    for epoch in range(num_epochs):
        model.generator.train()
        model.discriminator.train()
        model.feature_extractor.train()

        total_g_loss = 0.0
        total_d_loss = 0.0
        total_c_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            batch_size = images.size(0)

            # 1>1
            if batch_size == 1:
                continue

            images = images.to(device)
            labels = labels.to(device)

            # 
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # 
            optimizer_d.zero_grad()

            # 
            real_validity = model(images, labels, mode="discriminate")
            d_loss_real = adversarial_loss(real_validity, real_labels)

            # 
            noise = torch.randn(batch_size, model.latent_dim).to(device)
            fake_images = model(noise, labels, mode="generate")
            fake_validity = model(fake_images.detach(), labels, mode="discriminate")
            d_loss_fake = adversarial_loss(fake_validity, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_d.step()

            # 
            optimizer_g.zero_grad()

            # 
            noise = torch.randn(batch_size, model.latent_dim).to(device)
            fake_images = model(noise, labels, mode="generate")
            fake_validity = model(fake_images, labels, mode="discriminate")

            # 
            g_loss = adversarial_loss(fake_validity, real_labels)

            # 
            real_features = model.feature_extractor(images)
            fake_features = model.feature_extractor(fake_images)
            feature_loss = nn.MSELoss()(real_features, fake_features)

            g_loss_total = g_loss + 0.1 * feature_loss
            g_loss_total.backward()
            optimizer_g.step()

            # 
            optimizer_c.zero_grad()

            # 
            predictions = model(images, labels, mode="classify")
            c_loss = classification_loss(predictions, labels)
            c_loss.backward()
            optimizer_c.step()

            total_g_loss += g_loss_total.item()
            total_d_loss += d_loss.item()
            total_c_loss += c_loss.item()

            pbar.set_postfix(
                {
                    "G_Loss": f"{g_loss_total.item():.4f}",
                    "D_Loss": f"{d_loss.item():.4f}",
                    "C_Loss": f"{c_loss.item():.4f}",
                }
            )

        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)
        avg_c_loss = total_c_loss / len(train_loader)

        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        c_losses.append(avg_c_loss)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}, C_Loss: {avg_c_loss:.4f}"
        )

        # 
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": model.generator.state_dict(),
                    "discriminator_state_dict": model.discriminator.state_dict(),
                    "feature_extractor_state_dict": model.feature_extractor.state_dict(),
                    "g_loss": avg_g_loss,
                    "d_loss": avg_d_loss,
                    "c_loss": avg_c_loss,
                },
                os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"),
            )
            logger.info(f": epoch {epoch+1}")

    # 
    torch.save(
        {
            "generator_state_dict": model.generator.state_dict(),
            "discriminator_state_dict": model.discriminator.state_dict(),
            "feature_extractor_state_dict": model.feature_extractor.state_dict(),
            "g_losses": g_losses,
            "d_losses": d_losses,
            "c_losses": c_losses,
        },
        os.path.join(output_dir, "model_final.pth"),
    )

    # 
    results = {
        "num_epochs": num_epochs,
        "final_g_loss": g_losses[-1],
        "final_d_loss": d_losses[-1],
        "final_c_loss": c_losses[-1],
        "g_losses": g_losses,
        "d_losses": d_losses,
        "c_losses": c_losses,
    }

    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("")
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
    parser.add_argument("--epochs", type=int, default=100, help="")
    parser.add_argument("--lr", type=float, default=0.0002, help="")
    parser.add_argument("--latent-dim", type=int, default=100, help="")
    parser.add_argument("--output-dir", type=str, default="models/conditional_gan", help="")

    args = parser.parse_args()

    # 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f": {device}")

    # 
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -11
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
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    logger.info(f": {len(dataset)}")

    # 
    num_classes = len(dataset.class_to_idx)
    model = ConditionalGAN(num_classes, args.latent_dim, model_type=args.model_type)
    model = model.to(device)

    logger.info(f": {num_classes}")
    logger.info(f": {sum(p.numel() for p in model.generator.parameters()):,}")
    logger.info(f": {sum(p.numel() for p in model.discriminator.parameters()):,}")
    logger.info(f": {sum(p.numel() for p in model.feature_extractor.parameters()):,}")

    # 
    results = train_conditional_gan(
        model, train_loader, device, num_epochs=args.epochs, lr=args.lr, output_dir=args.output_dir
    )

    logger.info("")
    logger.info(f'G: {results["final_g_loss"]:.4f}')
    logger.info(f'D: {results["final_d_loss"]:.4f}')
    logger.info(f'C: {results["final_c_loss"]:.4f}')


if __name__ == "__main__":
    main()
