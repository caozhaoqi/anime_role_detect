#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import logging
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import get_model_with_attributes

# 
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("predict_with_attributes")


def load_model(model_path, model_type="mobilenet_v2"):
    """"""
    logger.info(f": {model_path}")

    # 
    checkpoint = torch.load(model_path, map_location="cpu")

    # checkpoint
    if "class_to_idx" in checkpoint:
        class_to_idx = checkpoint["class_to_idx"]
        num_classes = len(class_to_idx)
    else:
        num_classes = 5

    logger.info(f": {num_classes}")

    # 
    num_attributes = 6
    model = get_model_with_attributes(model_type, num_classes, num_attributes)

    # 
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info("")

    return model, class_to_idx


def load_attribute_config(config_path):
    """"""
    if not os.path.exists(config_path):
        logger.warning(f": {config_path}")
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_with_attributes(
    model, image_path, transform, device, class_to_idx, attribute_config=None
):
    """

    Args:
        model: 
        image_path: 
        transform: 
        device: 
        class_to_idx: 
        attribute_config: 

    Returns:
        dict: 
    """
    logger.info(f": {image_path}")

    # 
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 
    model.eval()
    with torch.no_grad():
        class_output, attribute_output = model(image_tensor)

    # 
    class_prob = torch.softmax(class_output, dim=1)
    class_idx = torch.argmax(class_prob, dim=1).item()
    class_confidence = class_prob[0, class_idx].item()

    # 
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predicted_class = idx_to_class.get(class_idx, "unknown")

    # 
    attribute_preds = attribute_output.squeeze().cpu().numpy()

    # 
    attribute_order = ["hair_color", "eye_color", "has_halo", "outfit", "hair_style", "accessories"]
    attribute_mappings = {}

    if attribute_config:
        attribute_order = attribute_config.get("attribute_order", attribute_order)
        attribute_mappings = attribute_config.get("attribute_mappings", {})

    # 
    predicted_attributes = {}
    for i, attr_name in enumerate(attribute_order):
        pred_idx = round(attribute_preds[i])

        # 
        mapping = attribute_mappings.get(attr_name, {})
        # 
        idx_to_attr = {v: k for k, v in mapping.items()}
        predicted_attributes[attr_name] = idx_to_attr.get(pred_idx, f"unknown_{pred_idx}")

    # 
    result = {
        "character": predicted_class,
        "confidence": class_confidence,
        "attributes": predicted_attributes,
        "attribute_confidences": attribute_preds.tolist(),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model-path", type=str, required=True, help="")
    parser.add_argument(
        "--model-type",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "efficientnet_b0", "resnet18"],
        help="",
    )
    parser.add_argument("--image", type=str, required=True, help="")
    parser.add_argument(
        "--config", type=str, default="../config/character_attributes.json", help=""
    )
    parser.add_argument("--output", type=str, default=None, help="")

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
    model, class_to_idx = load_model(args.model_path, args.model_type)
    model = model.to(device)

    # 
    attribute_config = None
    if os.path.exists(args.config):
        attribute_config = load_attribute_config(args.config)
        logger.info(f": {args.config}")
    else:
        logger.warning(f": {args.config}")

    # 
    result = predict_with_attributes(
        model, args.image, transform, device, class_to_idx, attribute_config
    )

    # 
    logger.info("\n" + "=" * 50)
    logger.info(":")
    logger.info("=" * 50)
    logger.info(f": {result['character']}")
    logger.info(f": {result['confidence']:.4f}")
    logger.info("\n:")
    for attr, value in result["attributes"].items():
        logger.info(f"  {attr}: {value}")
    logger.info("=" * 50)

    # 
    if args.output:
        os.makedirs(
            os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"\n: {args.output}")

    return result


if __name__ == "__main__":
    main()
