#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import logging
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
from src.models.models import get_model_with_attributes

"""

"""



# 
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_with_attributes")


class CharacterAttributeDataset(torch.utils.data.Dataset):
    """"""

    def __init__(self, root_dir, annotations_file, transform=None):
        """

        Args:
            root_dir: 
            annotations_file: 
            transform: 
        """
        self.root_dir = root_dir
        self.transform = transform

        # 
        with open(annotations_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        # 
        self.class_to_idx = {}
        idx = 0
        for ann in self.annotations:
            character = ann["character"]
            if character not in self.class_to_idx:
                self.class_to_idx[character] = idx
                idx += 1

        logger.info(
            f" {len(self.class_to_idx)} {len(self.annotations)} "
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.root_dir, ann["image_path"])
        image = Image.open(img_path).convert("RGB")

        # 
        character = ann["character"]
        label = self.class_to_idx[character]

        # 
        attribute_labels = ann["attribute_labels"]
        attribute_labels = torch.tensor(attribute_labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label, attribute_labels


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
        if "classifier.weight" in state_dict:
            num_classes = state_dict["classifier.weight"].shape[0]
        else:
            num_classes = 5  # 

        class_to_idx = {}

    logger.info(f": {num_classes}")

    # 
    num_attributes = 6  # 6
    model = get_model_with_attributes(model_type, num_classes, num_attributes)

    # 
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info(f": {class_to_idx}")

    return model, class_to_idx


def evaluate_model(model, test_loader, device, class_names, attribute_names):
    """

    Args:
        model: 
        test_loader: 
        device: 
        class_names: 
        attribute_names: 

    Returns:
        results: 
    """
    logger.info("...")

    model.eval()
    all_predictions = []
    all_labels = []
    all_attribute_predictions = []
    all_attribute_labels = []

    with torch.no_grad():
        for images, labels, attribute_labels in tqdm(test_loader, desc=""):
            images, labels, attribute_labels = (
                images.to(device),
                labels.to(device),
                attribute_labels.to(device),
            )

            class_output, attribute_output = model(images)
            _, predicted = torch.max(class_output, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attribute_predictions.extend(attribute_output.cpu().numpy())
            all_attribute_labels.extend(attribute_labels.cpu().numpy())

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

    # 
    attribute_accuracy = []
    for i in range(len(attribute_names)):
        attr_labels = [lbl[i] for lbl in all_attribute_labels]
        attr_preds = [pred[i] for pred in all_attribute_predictions]
        # 
        attr_acc = accuracy_score(attr_labels, [round(p) for p in attr_preds])
        attribute_accuracy.append(attr_acc)
        logger.info(f"{attribute_names[i]} : {attr_acc * 100:.2f}%")

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": all_predictions,
        "labels": all_labels,
        "attribute_predictions": all_attribute_predictions,
        "attribute_labels": all_attribute_labels,
        "attribute_accuracy": attribute_accuracy,
    }


def test_single_image(model, image_path, transform, device, class_names, attribute_names):
    """

    Args:
        model: 
        image_path: 
        transform: 
        device: 
        class_names: 
        attribute_names: 
    """
    logger.info(f": {image_path}")

    # 
    image = Image.open(image_path).convert("RGB")

    # 
    if transform:
        image = transform(image).unsqueeze(0).to(device)

    # 
    model.eval()
    with torch.no_grad():
        class_output, attribute_output = model(image)

    # 
    class_prob = torch.softmax(class_output, dim=1)
    class_idx = torch.argmax(class_prob, dim=1).item()
    class_name = class_names[class_idx]
    class_confidence = class_prob[0, class_idx].item()

    # 
    attribute_preds = attribute_output.squeeze().cpu().numpy()
    attribute_results = {}
    for i, attr_name in enumerate(attribute_names):
        attribute_results[attr_name] = round(attribute_preds[i])

    logger.info(f": {class_name} (: {class_confidence:.4f})")
    logger.info(f": {attribute_results}")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/arona_plana_with_attributes/model_best.pth",
        help="",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "efficientnet_b0", "resnet18"],
        help="",
    )
    parser.add_argument(
        "--data-dir", type=str, default="../data/downloaded_images", help=""
    )
    parser.add_argument(
        "--annotations-file",
        type=str,
        default="../config/attribute_annotations.json",
        help="",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="")
    parser.add_argument(
        "--output-dir", type=str, default="test_results_with_attributes", help=""
    )
    parser.add_argument("--test-image", type=str, default=None, help="")
    parser.add_argument("--config", type=str, default=None, help="")

    args = parser.parse_args()

    # 
    if args.config is None:
        possible_configs = [
            "../config/character_attributes.json",
            "../../config/character_attributes.json",
            os.path.join(os.path.dirname(__file__), "..", "config", "character_attributes.json"),
        ]
        for config_path in possible_configs:
            if os.path.exists(config_path):
                args.config = config_path
                break

    if args.config:
        logger.info(f": {args.config}")

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

    # 
    attribute_names = ["hair_color", "eye_color", "has_halo", "outfit", "hair_style", "accessories"]
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)
                if "attribute_order" in config:
                    attribute_names = config["attribute_order"]
                    logger.info(f": {attribute_names}")
        except Exception as e:
            logger.warning(f": {e}")

    # 
    logger.info("...")
    dataset = CharacterAttributeDataset(args.data_dir, args.annotations_file, transform=transform)

    # 
    class_names = list(dataset.class_to_idx.keys())
    logger.info(f": {class_names}")

    # 
    if args.test_image:
        test_single_image(model, args.test_image, transform, device, class_names, attribute_names)
        return

    # 
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f": {len(dataset)}")

    # 
    results = evaluate_model(model, test_loader, device, class_names, attribute_names)

    # 
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "test_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        # 
        save_results = {
            "accuracy": results["accuracy"],
            "classification_report": results["classification_report"],
            "confusion_matrix": results["confusion_matrix"],
            "attribute_accuracy": results["attribute_accuracy"],
            "attribute_names": attribute_names,
            "class_names": class_names,
        }
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n: {output_path}")

    # 
    logger.info("\n==================================================")
    logger.info("")
    logger.info("==================================================")
    logger.info(f": {results['accuracy'] * 100:.2f}%")
    logger.info(f": {len(dataset)}")
    logger.info(f": {len(class_names)}")
    logger.info(f": {len(attribute_names)}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
