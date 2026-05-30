#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本使用本地模型实现对采集数据分类
每个文件夹下分为"是"和"否"两个子文件夹
"""

import os
import argparse
import logging
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("classify_collection.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class LocalCollectionClassifier:
    def __init__(self, model_path, max_workers=4):
        """
        初始化本地分类器

        Args:
            model_path: 模型路径
            max_workers: 最大线程数
        """
        self.model_path = model_path
        self.max_workers = max_workers
        self.model = None
        self.class_to_idx = None
        self.transform = None

    def _load_model(self):
        """
        加载模型
        """
        try:
            logger.info(f"加载模型: {self.model_path}")

            # 加载模型
            model_data = torch.load(
                self.model_path,
                map_location=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
            )

            # 提取模型和类别映射
            if "model_state_dict" in model_data:
                from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

                self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
                self.model.classifier[1] = torch.nn.Linear(
                    self.model.classifier[1].in_features, len(model_data["class_to_idx"])
                )
                self.model.load_state_dict(model_data["model_state_dict"])
                self.class_to_idx = model_data["class_to_idx"]
            elif "model" in model_data:
                self.model = model_data["model"]
                self.class_to_idx = model_data.get("class_to_idx", {})
            else:
                # 尝试直接加载完整模型
                self.model = model_data
                # 从目录中加载class_to_idx
                model_dir = os.path.dirname(self.model_path)
                class_to_idx_path = os.path.join(model_dir, "class_to_idx.json")
                if os.path.exists(class_to_idx_path):
                    import json

                    with open(class_to_idx_path, "r", encoding="utf-8") as f:
                        self.class_to_idx = json.load(f)

            # 设置模型为评估模式
            self.model.eval()

            # 定义预处理变换
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            logger.info(f"模型加载成功，类别数: {len(self.class_to_idx)}")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False

    def classify_image(self, image_path):
        """
        使用本地模型分类单张图片

        Args:
            image_path: 图片路径

        Returns:
            tuple: (角色名称, 置信度)
        """
        if not self.model or not self.transform:
            if not self._load_model():
                return "unknown", 0.0

        try:
            # 加载和预处理图片
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0)

            # 移动到适当的设备
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.model.to(device)
            input_tensor = input_tensor.to(device)

            # 模型推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_prob, predicted_idx = torch.max(probabilities, 1)

            # 转换为角色名称
            idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            role = idx_to_class.get(predicted_idx.item(), "unknown")
            confidence = max_prob.item()

            return role, float(confidence)
        except Exception as e:
            logger.error(f"分类图片失败 {image_path}: {e}")
            return "unknown", 0.0

    def classify_collection(self, collection_dir, threshold=0.5):
        """
        分类整个采集集

        Args:
            collection_dir: 采集集目录
            threshold: 置信度阈值
        """
        # 加载模型
        if not self._load_model():
            logger.error("无法加载模型，分类失败")
            return

        # 遍历每个角色文件夹
        for role_dir in os.listdir(collection_dir):
            role_path = os.path.join(collection_dir, role_dir)

            if not os.path.isdir(role_path):
                continue

            logger.info(f"开始分类角色: {role_dir}")

            # 创建"是"和"否"子文件夹
            yes_dir = os.path.join(role_path, "是")
            no_dir = os.path.join(role_path, "否")

            os.makedirs(yes_dir, exist_ok=True)
            os.makedirs(no_dir, exist_ok=True)

            # 获取所有图片文件
            image_files = []
            for file in os.listdir(role_path):
                file_path = os.path.join(role_path, file)
                if os.path.isfile(file_path) and file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp")
                ):
                    image_files.append(file_path)

            if not image_files:
                logger.info(f"角色 {role_dir} 没有图片")
                continue

            logger.info(f"找到 {len(image_files)} 张图片")

            # 使用线程池并行处理
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_image = {
                    executor.submit(self.classify_image, image_path): image_path
                    for image_path in image_files
                }

                for future in tqdm(as_completed(future_to_image), total=len(image_files)):
                    image_path = future_to_image[future]
                    try:
                        predicted_role, confidence = future.result()
                        results.append((image_path, predicted_role, confidence))
                    except Exception as e:
                        logger.error(f"处理图片失败 {image_path}: {e}")

            # 处理分类结果
            yes_count = 0
            no_count = 0

            for image_path, predicted_role, confidence in results:
                image_name = os.path.basename(image_path)

                # 检查预测的角色是否与文件夹名称匹配（支持拼音和汉字）
                is_match = False
                if predicted_role == role_dir:
                    is_match = True
                elif predicted_role in role_dir or role_dir in predicted_role:
                    # 处理部分匹配的情况
                    is_match = True

                if is_match and confidence >= threshold:
                    # 移动到"是"文件夹
                    dest_path = os.path.join(yes_dir, image_name)
                    os.rename(image_path, dest_path)
                    yes_count += 1
                else:
                    # 移动到"否"文件夹
                    dest_path = os.path.join(no_dir, image_name)
                    os.rename(image_path, dest_path)
                    no_count += 1

            logger.info(f"角色 {role_dir} 分类完成:")
            logger.info(f"  - 是: {yes_count} 张")
            logger.info(f"  - 否: {no_count} 张")
            logger.info(f"  - 总: {yes_count + no_count} 张")


def main():
    parser = argparse.ArgumentParser(description="对采集数据进行分类")
    parser.add_argument(
        "--collection_dir",
        type=str,
        default="/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/role_images",
        help="采集数据目录",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/Users/caozhaoqi/PycharmProjects/anime_role_detect/models/efficientnet_b3_loli_incremental_20260425_191252/model_best.pth",
        help="模型路径",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--max_workers", type=int, default=4, help="最大线程数")

    args = parser.parse_args()

    logger.info(f"开始分类采集数据")
    logger.info(f"采集目录: {args.collection_dir}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"置信度阈值: {args.threshold}")

    classifier = LocalCollectionClassifier(model_path=args.model_path, max_workers=args.max_workers)

    classifier.classify_collection(args.collection_dir, args.threshold)

    logger.info("分类完成！")


if __name__ == "__main__":
    main()
