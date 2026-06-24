#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
import logging
import numpy as np

# 
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("generate_conditional")


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


def load_model(model_path, num_classes, latent_dim=100, model_type="mobilenet_v2", device="mps"):
    """GAN

    Args:
        model_path: 
        num_classes: 
        latent_dim: 
        model_type: 
        device: 
    """
    # 
    model = ConditionalGAN(num_classes, latent_dim, model_type=model_type)
    model = model.to(device)

    # 
    checkpoint = torch.load(model_path, map_location=device)
    model.generator.load_state_dict(checkpoint["generator_state_dict"])
    model.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    model.feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])

    logger.info(f": {model_path}")
    return model


def generate_conditional_images(
    model, num_images_per_class=5, output_dir="data/conditional_generated", device="mps"
):
    """

    Args:
        model: GAN
        num_images_per_class: 
        output_dir: 
        device: 
    """
    os.makedirs(output_dir, exist_ok=True)

    model.generator.eval()

    logger.info(f" {num_images_per_class} ")

    for class_idx in range(model.num_classes):
        class_dir = os.path.join(output_dir, f"class_{class_idx}")
        os.makedirs(class_dir, exist_ok=True)

        for i in range(num_images_per_class):
            try:
                # 
                noise = torch.randn(1, model.latent_dim).to(device)

                # 
                labels = torch.LongTensor([class_idx]).to(device)

                # 
                with torch.no_grad():
                    generated = model(noise, labels, mode="generate")

                # PIL
                generated = generated.squeeze().cpu().permute(1, 2, 0).numpy()
                # -110255
                generated = ((generated + 1) * 127.5).astype(np.uint8)
                image = Image.fromarray(generated)

                # 
                output_path = os.path.join(class_dir, f"generated_{i+1}.png")
                image.save(output_path)
                logger.info(f": {output_path}")

            except Exception as e:
                logger.error(f": {e}")
                continue

    logger.info("")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model-path", type=str, default="models/conditional_gan/model_final.pth", help=""
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "efficientnet_b0", "resnet18"],
        help="",
    )
    parser.add_argument("--num-classes", type=int, default=2, help="")
    parser.add_argument("--latent-dim", type=int, default=100, help="")
    parser.add_argument(
        "--num-images-per-class", type=int, default=5, help=""
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/conditional_generated", help=""
    )

    args = parser.parse_args()

    # 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f": {device}")

    # 
    model = load_model(args.model_path, args.num_classes, args.latent_dim, args.model_type, device)

    # 
    generate_conditional_images(model, args.num_images_per_class, args.output_dir, device)


if __name__ == "__main__":
    main()
