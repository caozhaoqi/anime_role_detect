#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


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
logger = logging.getLogger("test_classification_model")


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
        self.idx_to_class = {}

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
            self.idx_to_class[idx] = cls
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)
            idx += 1

        logger.info(
            f" {len(self.class_to_idx)} {len(self.images)} "
        )
        logger.info(f": {self.class_to_idx}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_model(model_path, model_type="mobilenet_v2"):
    """

    Args:
        model_path: 
        model_type: 

    Returns:
        model: 
        class_to_idx: 
    """
    logger.info(f": {model_path}")

    # 
    checkpoint = torch.load(model_path, map_location="cpu")

    # checkpoint
    if "class_to_idx" in checkpoint:
        class_to_idx = checkpoint["class_to_idx"]
        num_classes = len(class_to_idx)
    else:
        # 
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # 
        if "classifier.5.weight" in state_dict:
            num_classes = state_dict["classifier.5.weight"].shape[0]
        elif "fc.5.weight" in state_dict:
            num_classes = state_dict["fc.5.weight"].shape[0]
        elif "classifier.1.weight" in state_dict:
            num_classes = state_dict["classifier.1.weight"].shape[0]
        elif "fc.weight" in state_dict:
            num_classes = state_dict["fc.weight"].shape[0]
        else:
            num_classes = 2  # 

        class_to_idx = {}

    logger.info(f": {num_classes}")

    # 
    if model_type == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        # train_incremental.py
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.15),
            nn.Linear(512, num_classes),
        )
    elif model_type == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        # train_incremental.py
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.15),
            nn.Linear(512, num_classes),
        )
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=False)
        # train_incremental.py
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.15),
            nn.Linear(512, num_classes),
        )
    elif model_type == "resnet18":
        model = models.resnet18(pretrained=False)
        # train_incremental.py
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.15),
            nn.Linear(512, num_classes),
        )
    else:
        raise ValueError(f": {model_type}")

    # 
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info(f": {class_to_idx}")

    return model, class_to_idx


def evaluate_model(model, test_loader, device, class_names):
    """

    Args:
        model: 
        test_loader: 
        device: 
        class_names: 

    Returns:
        results: 
    """
    logger.info("...")

    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=""):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 
    accuracy = accuracy_score(all_labels, all_predictions)
    logger.info(f": {accuracy * 100:.2f}%")

    # labels
    labels = list(range(len(class_names)))

    # 
    report = classification_report(
        all_labels, all_predictions, target_names=class_names, labels=labels, output_dict=True
    )

    # 
    cm = confusion_matrix(all_labels, all_predictions, labels=labels)

    logger.info("\n:")
    logger.info(
        classification_report(all_labels, all_predictions, target_names=class_names, labels=labels)
    )

    logger.info("\n:")
    logger.info(cm)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": all_predictions,
        "labels": all_labels,
        "probabilities": [p.tolist() for p in all_probs],
    }


def test_single_image(model, image_path, transform, device, class_names):
    """

    Args:
        model: 
        image_path: 
        transform: 
        device: 
        class_names: 
    """
    logger.info(f": {image_path}")

    # 
    image = Image.open(image_path).convert("RGB")

    # 
    if transform:
        image_tensor = transform(image).unsqueeze(0).to(device)

    # 
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    confidence = probs[0][predicted.item()].item()

    logger.info(f": {predicted_class}")
    logger.info(f": {confidence * 100:.2f}%")
    logger.info(":")
    for i, class_name in enumerate(class_names):
        logger.info(f"  {class_name}: {probs[0][i].item() * 100:.2f}%")

    return predicted_class, confidence, probs[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model-path", type=str, default="models/arona_plana/model_best.pth", help=""
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "efficientnet_b0", "resnet18", "resnet50"],
        help="",
    )
    parser.add_argument(
        "--data-dir", type=str, default="../data/downloaded_images", help=""
    )
    parser.add_argument("--batch-size", type=int, default=8, help="")
    parser.add_argument("--test-image", type=str, default=None, help="")
    parser.add_argument("--output-dir", type=str, default="test_results", help="")

    args = parser.parse_args()

    # 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f": {device}")

    # 
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 
    model, loaded_class_to_idx = load_model(args.model_path, args.model_type)
    model = model.to(device)

    #  - 
    logger.info("...")
    dataset = CharacterDataset(args.data_dir, transform=transform)

    # 
    class_names = list(dataset.class_to_idx.keys())
    logger.info(f": {class_names}")

    # 
    if args.test_image:
        test_single_image(model, args.test_image, transform, device, class_names)
        return

    # 
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f": {len(dataset)}")

    # 
    results = evaluate_model(model, test_loader, device, class_names)

    # 
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "test_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        # 
        save_results = {
            "accuracy": results["accuracy"],
            "classification_report": results["classification_report"],
            "confusion_matrix": results["confusion_matrix"],
            "class_names": class_names,
        }
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n: {output_path}")

    # 
    logger.info("\n" + "=" * 50)
    logger.info("")
    logger.info("=" * 50)
    logger.info(f": {results['accuracy'] * 100:.2f}%")
    logger.info(f": {len(dataset)}")
    logger.info(f": {len(class_names)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
