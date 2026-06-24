#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import logging
import argparse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_ensemble")


class ModelEnsemble:
    """"""

    def __init__(self, model_paths, device="cpu"):
        self.models = []
        self.device = device

        for model_path in model_paths:
            logger.info(f": {model_path}")
            checkpoint = torch.load(model_path, map_location=device)

            # 
            if "efficientnet" in model_path.lower():
                model = models.efficientnet_b3(pretrained=False)
                num_classes = (
                    checkpoint["class_to_idx"].__len__() if "class_to_idx" in checkpoint else 22
                )
                model.classifier = nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(model.classifier[1].in_features, 768),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(768),
                    nn.Dropout(0.2),
                    nn.Linear(768, num_classes),
                )
            else:
                model = models.mobilenet_v2(pretrained=False)
                num_classes = (
                    checkpoint["class_to_idx"].__len__() if "class_to_idx" in checkpoint else 22
                )
                model.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(model.classifier[1].in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.15),
                    nn.Linear(512, num_classes),
                )

            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            model.eval()

            self.models.append(model)
            logger.info(f": {num_classes}")

        logger.info(f" {len(self.models)} ")

    def predict(self, image, transform):
        """

        Args:
            image: PIL Image
            transform: 

        Returns:
            
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        img_tensor = transform(image).unsqueeze(0).to(self.device)

        # 
        all_predictions = []
        with torch.no_grad():
            for model in self.models:
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                all_predictions.append(probs.cpu().numpy())

        # 
        ensemble_probs = self._ensemble_predictions(all_predictions)

        return ensemble_probs

    def _ensemble_predictions(self, predictions):
        """

        Args:
            predictions: list of numpy arrays, shape (num_models, 1, num_classes)

        Returns:
            
        """
        predictions = np.array(predictions)

        # 1: 
        avg_probs = np.mean(predictions, axis=0)[0]

        # 2: 
        weights = np.array([1.0] * len(predictions))
        weighted_probs = np.average(predictions, axis=0, weights=weights)[0]

        # 3: 
        max_probs = np.max(predictions, axis=0)[0]

        # 4: 
        votes = np.argmax(predictions, axis=2)
        vote_probs = np.zeros_like(predictions[0][0])
        for vote in votes[0]:
            vote_probs[vote] += 1
        vote_probs = vote_probs / len(predictions)

        # 
        return avg_probs

    def predict_batch(self, images, transform):
        """"""
        if isinstance(images[0], str):
            images = [Image.open(img).convert("RGB") for img in images]

        batch_tensors = torch.stack([transform(img) for img in images]).to(self.device)

        all_predictions = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(batch_tensors)
                probs = F.softmax(outputs, dim=1)
                all_predictions.append(probs.cpu().numpy())

        predictions = np.array(all_predictions)
        avg_probs = np.mean(predictions, axis=0)

        return avg_probs


def evaluate_ensemble(ensemble, data_dir, class_to_idx, transform):
    """"""

    device = ensemble.device
    correct = 0
    total = 0
    per_class_correct = {cls: 0 for cls in class_to_idx}
    per_class_total = {cls: 0 for cls in class_to_idx}

    for character in os.listdir(data_dir):
        character_dir = os.path.join(data_dir, character)
        if not os.path.isdir(character_dir):
            continue

        if character not in class_to_idx:
            continue

        for img_name in os.listdir(character_dir):
            img_path = os.path.join(character_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue

            # 
            probs = ensemble.predict(img_path, transform)
            predicted_class = np.argmax(probs)

            # 
            true_class = class_to_idx[character]

            # 
            total += 1
            per_class_total[character] += 1

            if predicted_class == true_class:
                correct += 1
                per_class_correct[character] += 1

    accuracy = 100 * correct / total if total > 0 else 0
    logger.info(f": {accuracy:.2f}% ({correct}/{total})")

    # 
    logger.info("\n:")
    for character in sorted(class_to_idx.keys()):
        if per_class_total[character] > 0:
            class_acc = 100 * per_class_correct[character] / per_class_total[character]
            logger.info(
                f"  {character}: {class_acc:.2f}% ({per_class_correct[character]}/{per_class_total[character]})"
            )

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model-paths", type=str, nargs="+", required=True, help=""
    )
    parser.add_argument("--data-dir", type=str, default="../../data/train", help="")
    parser.add_argument(
        "--device", type=str, default="mps", choices=["cpu", "mps", "cuda"], help=""
    )

    args = parser.parse_args()

    device = torch.device(
        args.device
        if (args.device == "cuda" and torch.cuda.is_available())
        or (args.device == "mps" and torch.backends.mps.is_available())
        else "cpu"
    )
    logger.info(f": {device}")

    # 
    ensemble = ModelEnsemble(args.model_paths, device=device)

    # 
    transform = transforms.Compose(
        [
            transforms.Resize((288, 288)),
            transforms.CenterCrop((288, 288)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 
    class_to_idx = {}
    for model_path in args.model_paths:
        checkpoint = torch.load(model_path, map_location=device)
        if "class_to_idx" in checkpoint:
            class_to_idx.update(checkpoint["class_to_idx"])
            break

    if not class_to_idx:
        logger.error("")
        return

    logger.info(f": {len(class_to_idx)}")

    # 
    accuracy = evaluate_ensemble(ensemble, args.data_dir, class_to_idx, transform)

    logger.info(f"\n: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
