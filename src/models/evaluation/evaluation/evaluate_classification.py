#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

arona_plana
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import logging
from tqdm import tqdm
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("evaluate_classification")


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


def load_model(model_path, device):
    """"""
    # 
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {})
    num_classes = len(class_to_idx)

    # 
    state_dict_keys = list(checkpoint["model_state_dict"].keys())
    if "conv1.weight" in state_dict_keys:
        # ResNet
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif "features.0.0.weight" in state_dict_keys:
        # MobileNetV2
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif "features.0.weight" in state_dict_keys:
        # EfficientNet
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("")

    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, class_to_idx


def evaluate_classification(model, test_loader, device, class_names):
    """"""
    logger.info("...")

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="")):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 
    accuracy = accuracy_score(all_labels, all_predictions)
    logger.info(f": {accuracy * 100:.2f}%")

    # 
    unique_labels = sorted(list(set(all_labels)))
    actual_class_names = [class_names[label] for label in unique_labels]

    # 
    report = classification_report(
        all_labels,
        all_predictions,
        labels=unique_labels,
        target_names=actual_class_names,
        output_dict=True,
    )

    # 
    cm = confusion_matrix(all_labels, all_predictions, labels=unique_labels)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "actual_classes": actual_class_names,
    }


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model-path", type=str, default="models/arona_plana/model_best.pth", help=""
    )
    parser.add_argument("--data-dir", type=str, default="data/train", help="")
    parser.add_argument("--batch-size", type=int, default=8, help="")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="")

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
    logger.info("...")
    dataset = CharacterDataset(args.data_dir, transform=transform)

    # 
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f": {len(dataset)}")

    # 
    logger.info("...")
    model, model_class_to_idx = load_model(args.model_path, device)

    # 
    logger.info("...")
    filtered_images = []
    filtered_labels = []
    filtered_class_to_idx = {}

    # 
    for cls_name, cls_idx in dataset.class_to_idx.items():
        if cls_name in model_class_to_idx:
            filtered_class_to_idx[cls_name] = model_class_to_idx[cls_name]

    # 
    for img_path, label in zip(dataset.images, dataset.labels):
        cls_name = list(dataset.class_to_idx.keys())[
            list(dataset.class_to_idx.values()).index(label)
        ]
        if cls_name in model_class_to_idx:
            filtered_images.append(img_path)
            filtered_labels.append(model_class_to_idx[cls_name])

    # 
    class FilteredDataset(Dataset):
        def __init__(self, images, labels, transform):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            image = Image.open(img_path).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    filtered_dataset = FilteredDataset(filtered_images, filtered_labels, transform)
    filtered_loader = DataLoader(
        filtered_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    logger.info(f": {len(filtered_dataset)}")

    # 
    class_names = list(filtered_class_to_idx.keys())
    classification_results = evaluate_classification(model, filtered_loader, device, class_names)

    # 
    results = {
        "model_path": args.model_path,
        "classification": classification_results,
        "class_names": class_names,
        "actual_classes": classification_results.get("actual_classes", []),
        "test_set_size": len(dataset),
        "filtered_test_set_size": len(filtered_dataset),
    }

    # 
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "classification_evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f": {output_path}")

    # 
    logger.info("\n" + "=" * 50)
    logger.info("")
    logger.info("=" * 50)
    logger.info(f": {classification_results['accuracy'] * 100:.2f}%")
    logger.info(f": {len(dataset)}")
    logger.info(f": {len(filtered_dataset)}")
    logger.info(f": {classification_results.get('actual_classes', [])}")
    logger.info("\n:")
    for class_name, metrics in classification_results["classification_report"].items():
        if class_name not in ["accuracy", "macro avg", "weighted avg"]:
            logger.info(f"  {class_name}:")
            logger.info(f"    : {metrics['precision'] * 100:.2f}%")
            logger.info(f"    : {metrics['recall'] * 100:.2f}%")
            logger.info(f"    F1: {metrics['f1-score'] * 100:.2f}%")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
