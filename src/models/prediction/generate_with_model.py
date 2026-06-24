#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import logging
import numpy as np

# 
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("generate_with_model")


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
        return self.unet(torch.cat([x, noise], dim=1))


def generate_images(
    generator, feature_extractor, num_images=5, output_dir="data/generated_model", device="mps"
):
    """

    Args:
        generator: 
        feature_extractor: 
        num_images: 
        output_dir: 
        device: 
    """
    os.makedirs(output_dir, exist_ok=True)

    # 
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    generator.eval()
    feature_extractor.eval()

    logger.info(f": {num_images}")

    for i in range(num_images):
        try:
            # 
            noise = torch.randn(1, 3, 224, 224).to(device)

            # 
            random_features = torch.randn(1, feature_extractor.feature_dim).to(device)

            # 
            with torch.no_grad():
                generated = generator(noise, noise, random_features)

            # PIL
            generated = generated.squeeze().cpu().permute(1, 2, 0).numpy()
            generated = (generated * 255).astype(np.uint8)
            image = Image.fromarray(generated)

            # 
            output_path = os.path.join(output_dir, f"generated_{i+1}.png")
            image.save(output_path)
            logger.info(f": {output_path}")

        except Exception as e:
            logger.error(f": {e}")
            continue

    logger.info("")


def load_model(model_path, model_type="mobilenet_v2", device="mps"):
    """

    Args:
        model_path: 
        model_type: 
        device: 
    """
    # 
    feature_extractor = FeatureExtractor(model_type)
    feature_extractor = feature_extractor.to(device)

    # 
    generator = DiffusionGenerator(feature_extractor.feature_dim)
    generator = generator.to(device)

    # 
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])

    logger.info(f": {model_path}")
    return generator, feature_extractor


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model-path", type=str, default="models/generator/generator_best.pth", help=""
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "efficientnet_b0", "resnet18"],
        help="",
    )
    parser.add_argument("--num-images", type=int, default=5, help="")
    parser.add_argument("--output-dir", type=str, default="data/generated_model", help="")

    args = parser.parse_args()

    # 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f": {device}")

    # 
    generator, feature_extractor = load_model(args.model_path, args.model_type, device)

    # 
    generate_images(generator, feature_extractor, args.num_images, args.output_dir, device)


if __name__ == "__main__":
    main()
