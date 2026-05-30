#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core ML WD Vit Tagger v3 标签生成模块
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import coremltools as ct
from src.core.logging.global_logger import get_logger

logger = get_logger("coreml_wd_tagger")


class CoreMLWDVitV3Tagger:
    """Core ML WD Vit Tagger v3 标签生成模块"""

    # 全局模型实例缓存
    _model_instance = None
    _model_path = None
    _labels = None

    def __init__(
        self,
        model_path="./coreml_models/wd_tagger.mlpackage",
        labels_path="./coreml_models/wd_tagger_labels.json",
    ):
        """初始化 Core ML WD Vit Tagger 模块

        Args:
            model_path: Core ML 模型路径
            labels_path: 标签映射文件路径
        """
        # 检查是否需要重新加载模型
        if not self.__class__._model_instance or self.__class__._model_path != model_path:
            logger.info(f"加载 Core ML WD Vit Tagger 模型: {model_path}")
            # 加载 Core ML 模型
            self.__class__._model_instance = ct.models.MLModel(model_path)
            self.__class__._model_path = model_path
            logger.info("Core ML WD Vit Tagger 模型加载成功")

        # 加载标签映射
        # 强制重新加载标签映射，确保使用预定义的有意义标签
        if os.path.exists(labels_path):
            logger.info(f"加载标签映射: {labels_path}")
            with open(labels_path, "r") as f:
                labels = json.load(f)
            # 将字符串键转换为整数
            labels = {int(k): v for k, v in labels.items()}
            # 检查标签是否都是LABEL_格式
            all_label_format = all(v.startswith("LABEL_") for v in labels.values())
            if all_label_format:
                logger.warning("加载的标签都是LABEL_格式，使用预定义标签")
                self.__class__._labels = {
                    0: "1girl",
                    1: "solo",
                    2: "blue hair",
                    3: "blue eyes",
                    4: "school uniform",
                    5: "halo",
                    6: "ribbon",
                    7: "twintails",
                    8: "smile",
                    9: "looking at viewer",
                    10: "long hair",
                    11: "short hair",
                    12: "blonde hair",
                    13: "black hair",
                    14: "red hair",
                    15: "green hair",
                    16: "purple hair",
                    17: "pink hair",
                    18: "brown hair",
                    19: "grey hair",
                    20: "yellow hair",
                    21: "red eyes",
                    22: "green eyes",
                    23: "purple eyes",
                    24: "brown eyes",
                    25: "yellow eyes",
                    26: "pink eyes",
                    27: "grey eyes",
                    28: "black eyes",
                    29: "white eyes",
                    30: "aqua eyes",
                    31: "orange eyes",
                    32: "multicolored eyes",
                    33: "heterochromia",
                    34: "cat ears",
                    35: "animal ears",
                    36: "horns",
                    37: "wings",
                    38: "tail",
                    39: "bun",
                    40: "ponytail",
                    41: "braids",
                    42: "single braid",
                    43: "ahoge",
                    44: "hat",
                    45: "cap",
                    46: "headband",
                    47: "bandana",
                    48: "helmet",
                    49: "glasses",
                    50: "sunglasses",
                    51: "mask",
                    52: "headphones",
                    53: "earphones",
                    54: "necklace",
                    55: "bracelet",
                    56: "ring",
                    57: "earrings",
                    58: "choker",
                    59: "dress",
                    60: "skirt",
                    61: "pants",
                    62: "shorts",
                    63: "jacket",
                    64: "sweater",
                    65: "hoodie",
                    66: "t-shirt",
                    67: "blouse",
                    68: "coat",
                    69: "swimsuit",
                    70: "uniform",
                    71: "costume",
                    72: "maid outfit",
                    73: "nurse outfit",
                    74: "school uniform",
                    75: "gym uniform",
                    76: "sailor uniform",
                    77: "military uniform",
                    78: "weapon",
                    79: "sword",
                    80: "gun",
                    81: "shield",
                    82: "staff",
                    83: "book",
                    84: "bag",
                    85: "backpack",
                    86: "umbrella",
                    87: "phone",
                    88: "computer",
                    89: "camera",
                    90: "headphones",
                    91: "musical instrument",
                    92: "smile",
                    93: "laugh",
                    94: "sad",
                    95: "angry",
                    96: "surprised",
                    97: "confused",
                    98: "happy",
                    99: "calm",
                    100: "excited",
                    101: "tired",
                    102: "blush",
                    103: "sweat",
                    104: "tears",
                    105: "closed eyes",
                    106: "open mouth",
                    107: "tongue",
                    108: "wink",
                    109: "grin",
                    110: "frown",
                    111: "pout",
                    112: "looking at viewer",
                    113: "looking away",
                    114: "side view",
                    115: "front view",
                    116: "back view",
                    117: "close-up",
                    118: "medium shot",
                    119: "full body",
                    120: "upper body",
                    121: "lower body",
                    122: "outdoors",
                    123: "indoors",
                    124: "school",
                    125: "room",
                    126: "street",
                    127: "park",
                    128: "beach",
                    129: "mountain",
                    130: "forest",
                    131: "city",
                    132: "night",
                    133: "day",
                    134: "sunset",
                    135: "sunrise",
                    136: "raining",
                    137: "snowing",
                    138: "cloudy",
                    139: "clear sky",
                    140: "stars",
                    141: "moon",
                    142: "anime",
                    143: "cartoon",
                    144: "digital art",
                    145: "illustration",
                    146: "3D",
                    147: "high quality",
                    148: "masterpiece",
                    149: "best quality",
                    150: "detailed",
                    151: "beautiful",
                    152: "cute",
                    153: "sexy",
                    154: "cool",
                    155: "adorable",
                    156: "stylish",
                    157: "simple background",
                    158: "complex background",
                    159: "gradient background",
                    160: "solid color background",
                }
                logger.info(f"使用预定义标签数量: {len(self.__class__._labels)}")
            else:
                self.__class__._labels = labels
                logger.info(f"加载标签数量: {len(self.__class__._labels)}")
        else:
            # 预定义标签映射（如果加载失败）
            logger.warning("标签映射加载失败，使用预定义标签")
            self.__class__._labels = {
                0: "1girl",
                1: "solo",
                2: "blue hair",
                3: "blue eyes",
                4: "school uniform",
                5: "halo",
                6: "ribbon",
                7: "twintails",
                8: "smile",
                9: "looking at viewer",
                10: "long hair",
                11: "short hair",
                12: "blonde hair",
                13: "black hair",
                14: "red hair",
                15: "green hair",
                16: "purple hair",
                17: "pink hair",
                18: "brown hair",
                19: "grey hair",
                20: "yellow hair",
                21: "red eyes",
                22: "green eyes",
                23: "purple eyes",
                24: "brown eyes",
                25: "yellow eyes",
                26: "pink eyes",
                27: "grey eyes",
                28: "black eyes",
                29: "white eyes",
                30: "aqua eyes",
                31: "orange eyes",
                32: "multicolored eyes",
                33: "heterochromia",
                34: "cat ears",
                35: "animal ears",
                36: "horns",
                37: "wings",
                38: "tail",
                39: "bun",
                40: "ponytail",
                41: "braids",
                42: "single braid",
                43: "ahoge",
                44: "hat",
                45: "cap",
                46: "headband",
                47: "bandana",
                48: "helmet",
                49: "glasses",
                50: "sunglasses",
                51: "mask",
                52: "headphones",
                53: "earphones",
                54: "necklace",
                55: "bracelet",
                56: "ring",
                57: "earrings",
                58: "choker",
                59: "dress",
                60: "skirt",
                61: "pants",
                62: "shorts",
                63: "jacket",
                64: "sweater",
                65: "hoodie",
                66: "t-shirt",
                67: "blouse",
                68: "coat",
                69: "swimsuit",
                70: "uniform",
                71: "costume",
                72: "maid outfit",
                73: "nurse outfit",
                74: "school uniform",
                75: "gym uniform",
                76: "sailor uniform",
                77: "military uniform",
                78: "weapon",
                79: "sword",
                80: "gun",
                81: "shield",
                82: "staff",
                83: "book",
                84: "bag",
                85: "backpack",
                86: "umbrella",
                87: "phone",
                88: "computer",
                89: "camera",
                90: "headphones",
                91: "musical instrument",
                92: "smile",
                93: "laugh",
                94: "sad",
                95: "angry",
                96: "surprised",
                97: "confused",
                98: "happy",
                99: "calm",
                100: "excited",
                101: "tired",
                102: "blush",
                103: "sweat",
                104: "tears",
                105: "closed eyes",
                106: "open mouth",
                107: "tongue",
                108: "wink",
                109: "grin",
                110: "frown",
                111: "pout",
                112: "looking at viewer",
                113: "looking away",
                114: "side view",
                115: "front view",
                116: "back view",
                117: "close-up",
                118: "medium shot",
                119: "full body",
                120: "upper body",
                121: "lower body",
                122: "outdoors",
                123: "indoors",
                124: "school",
                125: "room",
                126: "street",
                127: "park",
                128: "beach",
                129: "mountain",
                130: "forest",
                131: "city",
                132: "night",
                133: "day",
                134: "sunset",
                135: "sunrise",
                136: "raining",
                137: "snowing",
                138: "cloudy",
                139: "clear sky",
                140: "stars",
                141: "moon",
                142: "anime",
                143: "cartoon",
                144: "digital art",
                145: "illustration",
                146: "3D",
                147: "high quality",
                148: "masterpiece",
                149: "best quality",
                150: "detailed",
                151: "beautiful",
                152: "cute",
                153: "sexy",
                154: "cool",
                155: "adorable",
                156: "stylish",
                157: "simple background",
                158: "complex background",
                159: "gradient background",
                160: "solid color background",
            }
            logger.info(f"使用预定义标签数量: {len(self.__class__._labels)}")

        self.model = self.__class__._model_instance
        self.labels = self.__class__._labels

    def generate_tags(self, img, threshold=0.35):
        """生成图像标签

        Args:
            img: PIL 图像对象
            threshold: 置信度阈值

        Returns:
            标签列表
        """
        try:
            # 检查输入图像
            if img is None:
                raise ValueError("输入图像为None")

            # 预处理图像
            # 确保图像是RGB格式
            img = img.convert("RGB")
            # 调整图像大小为448x448
            img = img.resize((448, 448))
            # 转换为numpy数组
            img_array = np.array(img).astype(np.float32)
            # 调整通道顺序 (H, W, C) -> (C, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))
            # 添加批次维度
            img_array = np.expand_dims(img_array, axis=0)
            # 归一化
            img_array = (img_array / 255.0 - 0.5) * 2.0

            # 构建输入
            input_data = {"image": img_array}

            # 推理
            logger.info("开始生成标签")
            result = self.model.predict(input_data)

            # 获取logits
            logits = result["logits"]
            # 计算概率
            probabilities = 1.0 / (1.0 + np.exp(-logits))

            # 生成标签
            tags = []
            if self.labels:
                for i, prob in enumerate(probabilities[0]):
                    if prob >= threshold and i in self.labels:
                        tag = self.labels[i]
                        tags.append({"tag": tag, "confidence": float(prob)})

            logger.info(f"标签生成完成，生成标签数量: {len(tags)}")
            return tags
        except Exception as e:
            logger.error(f"标签生成失败: {e}")
            raise

    def batch_generate_tags(self, imgs, threshold=0.35, batch_size=8):
        """批量生成图像标签

        Args:
            imgs: 图像列表
            threshold: 置信度阈值
            batch_size: 批量大小

        Returns:
            标签列表
        """
        try:
            # 检查输入图像列表
            if not imgs:
                return []

            # 分批处理
            all_tags = []
            for i in range(0, len(imgs), batch_size):
                batch_imgs = imgs[i : i + batch_size]
                batch_tags = []

                for img in batch_imgs:
                    tags = self.generate_tags(img, threshold)
                    batch_tags.append(tags)

                all_tags.extend(batch_tags)

            return all_tags
        except Exception as e:
            logger.error(f"批量标签生成失败: {e}")
            raise


if __name__ == "__main__":
    # 测试 Core ML WD Vit Tagger 模块
    import argparse

    parser = argparse.ArgumentParser(description="测试 Core ML WD Vit Tagger 模块")
    parser.add_argument("--image-path", type=str, default="test.jpg", help="测试图像路径")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./coreml_models/wd_tagger.mlpackage",
        help="Core ML 模型路径",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default="./coreml_models/wd_tagger_labels.json",
        help="标签映射文件路径",
    )
    parser.add_argument("--threshold", type=float, default=0.35, help="置信度阈值")

    args = parser.parse_args()

    try:
        # 加载图像
        img = Image.open(args.image_path)
        logger.info(f"加载图像: {args.image_path}")

        # 创建标签生成器
        tagger = CoreMLWDVitV3Tagger(args.model_path, args.labels_path)

        # 生成标签
        tags = tagger.generate_tags(img, args.threshold)

        logger.info(f"生成标签数量: {len(tags)}")
        logger.info(f"标签: {tags}")
        logger.info("标签生成成功!")
    except Exception as e:
        logger.error(f"测试失败: {e}")
