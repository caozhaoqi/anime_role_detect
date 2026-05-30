#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WD Vit Tagger v3 模型集成
"""

import os
import argparse
from PIL import Image
import json
from tqdm import tqdm
import requests

# 禁用MPS，避免锁竞争问题
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# 导入torch并禁用MPS
import torch

torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

# 设置单线程模式
torch.set_num_threads(1)

# 延迟导入transformers模块
AutoProcessor = None
AutoModelForImageClassification = None
CLIPProcessor = None
CLIPModel = None

# 设置Hugging Face缓存目录为项目目录
os.environ["HF_HOME"] = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "huggingface_cache"
)

from src.core.logging.global_logger import get_logger

logger = get_logger("wd_vit_v3_tagger")


# 动态导入函数
def import_torch_modules():
    global torch, AutoProcessor, AutoModelForImageClassification, CLIPProcessor, CLIPModel

    # 导入模块
    from transformers import (
        AutoProcessor,
        AutoModelForImageClassification,
        CLIPProcessor,
        CLIPModel,
    )


class WDViTV3Tagger:
    """WD Vit Tagger v3 标签生成器"""

    # 全局Core ML标签生成器实例缓存
    _coreml_tagger = None

    def __init__(self, device=None):
        # 禁用Core ML模式，避免锁竞争问题
        self.coreml_mode = False

        # 不使用torch，直接设置设备为CPU
        self.device = "cpu"
        self.logger = get_logger("wd_vit_v3_tagger")
        self.logger.info(f"WD Vit Tagger 使用设备: {self.device}")
        self.wd_model = None
        self.wd_processor = None
        self.clip_model = None
        self.clip_processor = None
        self.id2label = {}
        self.num_id2label = {}

        self.logger.info("WD Vit Tagger 模块初始化完成，使用扩展标签列表")
        self.tags = [
            # 角色数量
            "1girl",
            "2girls",
            "3girls",
            "4+girls",
            "1boy",
            "2boys",
            "3boys",
            "4+boys",
            "solo",
            "group",
            # 头发特征
            "long hair",
            "medium hair",
            "short hair",
            "very short hair",
            "twintails",
            "ponytail",
            "bun",
            "braids",
            "single braid",
            "ahoge",
            "blue hair",
            "blonde hair",
            "black hair",
            "red hair",
            "green hair",
            "purple hair",
            "pink hair",
            "brown hair",
            "grey hair",
            "yellow hair",
            "white hair",
            "silver hair",
            "aqua hair",
            "orange hair",
            "multicolored hair",
            "gradient hair",
            "streaked hair",
            # 眼睛特征
            "blue eyes",
            "red eyes",
            "green eyes",
            "purple eyes",
            "brown eyes",
            "yellow eyes",
            "pink eyes",
            "grey eyes",
            "black eyes",
            "white eyes",
            "aqua eyes",
            "orange eyes",
            "multicolored eyes",
            "heterochromia",
            "large eyes",
            "small eyes",
            "closed eyes",
            "wink",
            # 面部特征
            "smile",
            "laugh",
            "sad",
            "angry",
            "surprised",
            "confused",
            "happy",
            "calm",
            "excited",
            "tired",
            "blush",
            "sweat",
            "tears",
            "open mouth",
            "tongue",
            "grin",
            "frown",
            "pout",
            "looking at viewer",
            "looking away",
            # 头部装饰
            "cat ears",
            "animal ears",
            "horns",
            "wings",
            "tail",
            "hat",
            "cap",
            "headband",
            "bandana",
            "helmet",
            "glasses",
            "sunglasses",
            "mask",
            "headphones",
            "earphones",
            "ribbon",
            "bow",
            "flower",
            # 服装
            "dress",
            "skirt",
            "pants",
            "shorts",
            "jacket",
            "sweater",
            "hoodie",
            "t-shirt",
            "blouse",
            "coat",
            "swimsuit",
            "uniform",
            "costume",
            "maid outfit",
            "nurse outfit",
            "school uniform",
            "gym uniform",
            "sailor uniform",
            "military uniform",
            "casual",
            "formal",
            "traditional",
            "fantasy",
            "sci-fi",
            # 配饰
            "necklace",
            "bracelet",
            "ring",
            "earrings",
            "choker",
            "scarf",
            "gloves",
            "bag",
            "backpack",
            "umbrella",
            "weapon",
            "sword",
            "gun",
            "shield",
            "staff",
            "book",
            "phone",
            "computer",
            "camera",
            "musical instrument",
            # 姿势和视角
            "standing",
            "sitting",
            "lying",
            "kneeling",
            "walking",
            "running",
            "jumping",
            "dancing",
            "fighting",
            "side view",
            "front view",
            "back view",
            "close-up",
            "medium shot",
            "full body",
            "upper body",
            "lower body",
            # 场景
            "outdoors",
            "indoors",
            "school",
            "room",
            "street",
            "park",
            "beach",
            "mountain",
            "forest",
            "city",
            "night",
            "day",
            "sunset",
            "sunrise",
            "raining",
            "snowing",
            "cloudy",
            "clear sky",
            "stars",
            "moon",
            # 艺术风格
            "anime",
            "cartoon",
            "digital art",
            "illustration",
            "3D",
            "manga style",
            "realistic",
            "chibi",
            # 质量标签
            "high quality",
            "masterpiece",
            "best quality",
            "detailed",
            "beautiful",
            "cute",
            "sexy",
            "cool",
            "adorable",
            "stylish",
            # 背景
            "simple background",
            "complex background",
            "gradient background",
            "solid color background",
            "scenery",
            "urban",
            "natural",
        ]

    def load_model(self, model_name="SmilingWolf/wd-vit-tagger-v3"):
        """加载WD Vit Tagger v3模型

        Args:
            model_name: 模型名称

        Returns:
            bool: 模型加载是否成功
        """
        import platform
        import os

        # 检查是否是 macOS 环境
        if platform.system() == "Darwin":
            self.logger.info("检测到 macOS 环境，跳过 PyTorch 模型加载，使用默认标签")
            return False

        try:
            import_torch_modules()
            global torch, AutoProcessor, AutoModelForImageClassification

            self.logger.info(f"加载WD Vit Tagger v3模型: {model_name}")
            self.logger.info(f"使用设备: {self.device}")

            # 加载处理器和模型
            self.wd_processor = AutoProcessor.from_pretrained(model_name)
            self.wd_model = AutoModelForImageClassification.from_pretrained(model_name)

            # 将模型移到指定设备
            self.wd_model.to(self.device)
            self.wd_model.eval()

            # 获取标签映射
            if hasattr(self.wd_model.config, "id2label"):
                self.id2label = self.wd_model.config.id2label
                # 转换为数字索引映射
                self.num_id2label = {int(k): v for k, v in self.id2label.items()}

            self.logger.info("WD Vit Tagger v3模型加载完成")
            return True
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            # 加载失败时使用简单标签生成方法
            self.logger.info("加载模型失败，使用简单标签生成方法")
            return False

    def _filter_tags(self, tags):
        """过滤标签，去除冗余和低质量的标签

        Args:
            tags: 原始标签列表

        Returns:
            list: 过滤后的标签列表
        """
        if not tags:
            return []

        # 过滤规则
        filtered_tags = []
        tag_set = set()

        # 标签类别分组
        tag_categories = {
            "character_count": {
                "1girl",
                "2girls",
                "3girls",
                "4+girls",
                "1boy",
                "2boys",
                "3boys",
                "4+boys",
                "solo",
                "group",
            },
            "hair_length": {"long hair", "medium hair", "short hair", "very short hair"},
            "hair_color": {
                "blue hair",
                "blonde hair",
                "black hair",
                "red hair",
                "green hair",
                "purple hair",
                "pink hair",
                "brown hair",
                "grey hair",
                "yellow hair",
                "white hair",
                "silver hair",
                "aqua hair",
                "orange hair",
            },
            "eye_color": {
                "blue eyes",
                "red eyes",
                "green eyes",
                "purple eyes",
                "brown eyes",
                "yellow eyes",
                "pink eyes",
                "grey eyes",
                "black eyes",
                "white eyes",
                "aqua eyes",
                "orange eyes",
            },
            "expression": {
                "smile",
                "laugh",
                "sad",
                "angry",
                "surprised",
                "confused",
                "happy",
                "calm",
                "excited",
                "tired",
            },
            "pose": {"standing", "sitting", "lying", "kneeling", "walking", "running", "jumping"},
            "view": {
                "side view",
                "front view",
                "back view",
                "close-up",
                "medium shot",
                "full body",
            },
            "scene": {"outdoors", "indoors", "school", "room", "street", "park", "beach"},
            "time": {"night", "day", "sunset", "sunrise"},
            "quality": {"high quality", "masterpiece", "best quality", "detailed", "beautiful"},
        }

        # 已选择的类别标签
        selected_categories = {}

        # 移除重复标签
        for tag_info in tags:
            tag = tag_info["tag"].strip().lower()
            confidence = tag_info["confidence"]

            # 跳过空标签
            if not tag:
                continue

            # 跳过LABEL_前缀的标签
            if tag.startswith("label_"):
                continue

            # 跳过置信度过低的标签
            if confidence < 0.2:
                continue

            # 保留通用标签，不进行过滤
            # generic_tags = {'anime', 'cartoon', 'digital art', 'illustration', '3d'}
            # if tag in generic_tags and confidence < 0.6:
            #     continue

            # 类别冲突处理
            skip_tag = False
            for category, category_tags in tag_categories.items():
                if tag in category_tags:
                    if category in selected_categories:
                        # 只保留置信度更高的标签
                        existing_confidence = selected_categories[category]["confidence"]
                        if confidence <= existing_confidence:
                            skip_tag = True
                        else:
                            # 移除旧标签
                            old_tag = selected_categories[category]["tag"]
                            if old_tag in tag_set:
                                tag_set.remove(old_tag)
                                filtered_tags = [t for t in filtered_tags if t["tag"] != old_tag]
                    selected_categories[category] = {"tag": tag, "confidence": confidence}
                    break

            if skip_tag:
                continue

            # 去重
            if tag not in tag_set:
                tag_set.add(tag)
                filtered_tags.append(tag_info)

        # 按置信度排序
        filtered_tags.sort(key=lambda x: x["confidence"], reverse=True)

        # 平衡不同类别的标签
        category_count = {}
        balanced_tags = []

        for tag_info in filtered_tags:
            tag = tag_info["tag"]
            category = None

            for cat, cat_tags in tag_categories.items():
                if tag in cat_tags:
                    category = cat
                    break

            if category:
                if category not in category_count:
                    category_count[category] = 0
                if category_count[category] < 3:  # 每个类别最多3个标签
                    balanced_tags.append(tag_info)
                    category_count[category] += 1
            else:
                # 非分类标签直接添加
                balanced_tags.append(tag_info)

        # 确保至少有10个标签
        if len(balanced_tags) < 10:
            # 如果标签不足10个，添加一些通用标签
            generic_tags = [
                "anime",
                "cartoon",
                "digital art",
                "illustration",
                "character",
                "high quality",
                "detailed",
                "beautiful",
                "stylish",
                "colorful",
            ]
            for tag in generic_tags:
                if tag not in tag_set and len(balanced_tags) < 10:
                    balanced_tags.append({"tag": tag, "confidence": 0.5})
                    tag_set.add(tag)

        # 限制标签数量
        max_tags = 30
        if len(balanced_tags) > max_tags:
            balanced_tags = balanced_tags[:max_tags]

        return balanced_tags

    def generate_tags(self, image, threshold=0.2):
        """生成图像标签

        Args:
            image: 图像路径或Image对象
            threshold: 置信度阈值

        Returns:
            list: 标签列表
        """
        try:
            # 检查image是否为Image对象
            if isinstance(image, Image.Image):
                # 已经是Image对象，直接使用
                self.logger.debug("使用传入的Image对象")
            elif isinstance(image, str):
                # 是图像路径，加载图像
                image = Image.open(image).convert("RGB")
                self.logger.debug(f"加载图像成功: {image}")
            else:
                # 其他类型，尝试直接使用
                self.logger.debug(f"使用传入的对象，类型: {type(image)}")

            # 检查是否加载了模型
            if self.wd_model is not None and self.wd_processor is not None:
                # 使用PyTorch模型生成标签
                self.logger.debug("使用PyTorch模型生成标签")

                # 预处理图像
                inputs = self.wd_processor(images=image, return_tensors="pt").to(self.device)

                # 模型推理
                with torch.no_grad():
                    outputs = self.wd_model(**inputs)

                # 获取预测结果
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()

                # 生成标签
                tags = []
                for i, prob in enumerate(probabilities):
                    if prob >= threshold:
                        if i in self.num_id2label:
                            tag = self.num_id2label[i]
                        else:
                            tag = f"LABEL_{i}"
                        tags.append({"tag": tag, "confidence": float(prob)})

                # 过滤标签
                filtered_tags = self._filter_tags(tags)
                # 打印前10个标签
                self.logger.info(
                    f"PyTorch模型 前10个标签: {[t['tag'] for t in filtered_tags[:10]]}"
                )
            else:
                # 使用简单标签生成方法
                self.logger.debug("使用简单标签生成方法")
                # 返回默认标签
                tags = [{"tag": tag, "confidence": 0.5} for tag in self.tags[:10]]
                # 过滤标签
                filtered_tags = self._filter_tags(tags)
                # 打印前10个标签
                self.logger.info(
                    f"简单标签生成方法 前10个标签: {[t['tag'] for t in filtered_tags[:10]]}"
                )

            return filtered_tags
        except Exception as e:
            self.logger.error(f"生成标签失败: {e}")
            # 发生错误时，返回默认标签
            return [{"tag": tag, "confidence": 0.5} for tag in self.tags[:10]]

    def batch_generate_tags(self, image_dir, output_file, threshold=0.2):
        """批量生成标签

        Args:
            image_dir: 图像目录
            output_file: 输出文件路径
            threshold: 置信度阈值
        """
        results = []

        # 遍历图像目录
        for root, _, files in os.walk(image_dir):
            for file in tqdm(files, desc="批量生成标签"):
                if file.endswith((".jpg", ".jpeg", ".png", ".webp")):
                    image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(image_path, image_dir)

                    # 生成标签
                    tags = self.generate_tags(image_path, threshold)

                    # 保存结果
                    results.append({"image_path": relative_path, "tags": tags})

        # 保存结果
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"批量生成标签完成，保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="WD Vit Tagger v3 标签生成脚本")
    parser.add_argument("--image-path", type=str, default=None, help="单张图像路径")
    parser.add_argument("--image-dir", type=str, default=None, help="图像目录")
    parser.add_argument("--output-file", type=str, default="tags.json", help="输出文件路径")
    parser.add_argument("--threshold", type=float, default=0.05, help="置信度阈值")
    parser.add_argument(
        "--model-name", type=str, default="SmilingWolf/wd-vit-tagger-v3", help="模型名称"
    )

    args = parser.parse_args()

    # 创建标签生成器
    tagger = WDViTV3Tagger()
    tagger.load_model(args.model_name)

    # 处理单张图像
    if args.image_path:
        tags = tagger.generate_tags(args.image_path, args.threshold)
        logger.info(f"图像: {args.image_path}")
        logger.info(f"生成的标签: {tags}")

    # 处理图像目录
    elif args.image_dir:
        tagger.batch_generate_tags(args.image_dir, args.output_file, args.threshold)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
