#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from arona.training.train_incremental import get_model, CharacterDataset
from src.core.logging.global_logger import get_logger

"""


"""


# Python

logger = get_logger("model_evaluation")


class ModelEvaluator:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # 
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def load_dataset(self):
        """"""
        dataset = CharacterDataset(root_dir=self.data_dir, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return dataloader, list(dataset.class_to_idx.keys())

    def load_model(self, model_path, model_type, num_classes):
        """"""
        model = get_model(model_type, num_classes, dropout_rate=0.3)
        checkpoint = torch.load(model_path, map_location=self.device)

        # 
        pretrained_dict = checkpoint["model_state_dict"]
        model_dict = model.state_dict()

        # 
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if not k.startswith("classifier") and not k.startswith("fc")
        }

        # 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        model.to(self.device)
        model.eval()
        return model

    def evaluate_model(self, model, dataloader):
        """"""
        correct = 0
        total = 0
        start_time = time.time()

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        end_time = time.time()
        accuracy = 100 * correct / total
        inference_time = end_time - start_time

        return accuracy, inference_time, total

    def benchmark_model(self, model_path, model_type, num_classes):
        """"""
        logger.info(f": {model_path}")

        # 
        model = self.load_model(model_path, model_type, num_classes)

        # 
        dataloader, classes = self.load_dataset()

        # 
        accuracy, inference_time, total = self.evaluate_model(model, dataloader)

        # 
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

        # 
        avg_inference_time = inference_time / total * 1000  # ms per image

        logger.info(f": {model_type}")
        logger.info(f": {accuracy:.2f}%")
        logger.info(f": {inference_time:.2f} ({total})")
        logger.info(f": {avg_inference_time:.4f}/")
        logger.info(f": {model_size:.2f}MB")
        logger.info("-" * 50)

        return {
            "model_type": model_type,
            "accuracy": accuracy,
            "inference_time": inference_time,
            "avg_inference_time": avg_inference_time,
            "total_images": total,
            "model_size": model_size,
        }


def main():
    """"""
    data_dir = "data/downloaded_images"
    models = [
        {
            "path": "models/incremental/model_best.pth",
            "type": "mobilenet_v2",
            "name": "MobileNetV2",
        },
        {
            "path": "models/incremental_efficientnet_b0/model_best.pth",
            "type": "efficientnet_b0",
            "name": "EfficientNet-B0",
        },
        {
            "path": "models/incremental_efficientnet_b3/model_best.pth",
            "type": "efficientnet_b3",
            "name": "EfficientNet-B3",
        },
        {
            "path": "models/incremental_resnet50/model_best.pth",
            "type": "resnet50",
            "name": "ResNet50",
        },
    ]

    evaluator = ModelEvaluator(data_dir)

    # 
    _, classes = evaluator.load_dataset()
    num_classes = len(classes)

    results = []

    for model_info in models:
        if os.path.exists(model_info["path"]):
            result = evaluator.benchmark_model(model_info["path"], model_info["type"], num_classes)
            result["name"] = model_info["name"]
            results.append(result)
        else:
            logger.warning(f": {model_info['path']}")

    # 
    output_file = "models/evaluation_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f": {output_file}")

    # 
    logger.info("\n===  ===")
    for result in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        logger.info(
            f"{result['name']}: {result['accuracy']:.2f}% (: {result['avg_inference_time']:.4f}ms/, : {result['model_size']:.2f}MB)"
        )


if __name__ == "__main__":
    main()
