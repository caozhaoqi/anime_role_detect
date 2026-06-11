#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WD Vit Tagger v3 模型集成
支持多平台加速方案：
- macOS (Apple Silicon): CoreML (ANE加速，避开PyTorch MPS锁竞争问题)
- Linux/Windows: PyTorch (CUDA/MPS/CPU自动选择)
"""

import os
import sys
import argparse
import json
import gc
from PIL import Image
from tqdm import tqdm
import requests
import multiprocessing
import platform

# ==================== 【第一步：环境与多线程限制】 ====================
# 禁用多线程环境下的 OpenMP 锁争抢（针对 macOS M系列芯片优化）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# macOS 平台下导入 CoreML 相关模块（提前到这里，避免多进程设置导致死锁）
# 注意：由于 coremltools 和 scikit-learn 版本不兼容会导致锁阻塞，暂时禁用 CoreML
USE_COREML = False  # platform.system() == "Darwin"

# ==================== MPS加速开关 ====================
# 设置为True以启用MPS加速（适用于单进程场景，如model-service）
# 设置为False以禁用MPS（适用于多进程场景，避免锁竞争）
ENABLE_MPS = False  # model-service使用线程池，MPS非线程安全需禁用
# ====================================================

# 设置Hugging Face缓存目录为项目目录
os.environ["HF_HOME"] = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "huggingface_cache"
)

# ==================== 【第二步：平台检测与加速方案选择】 ====================
# macOS (Apple Silicon) 使用 CoreML，其他平台使用 PyTorch
# 只有非 macOS 平台才需要导入 torch 并封锁 MPS
if not USE_COREML:
    # 核心注入 - 彻底封死 MPS 探测（非 macOS 平台备用）
    torch = None
    try:
        import torch
        
        # 根据ENABLE_MPS开关决定是否启用MPS
        if ENABLE_MPS and platform.system() == "Darwin" and hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
            # 启用MPS加速
            if torch.backends.mps.is_available():
                print("✅ WDViTV3Tagger: MPS加速已启用")
                # 不封锁MPS，允许使用
            else:
                print("⚠️ WDViTV3Tagger: MPS不可用，将使用CPU")
                torch.backends.mps.is_available = lambda: False
                torch.backends.mps.is_built = lambda: False
        else:
            # 禁用MPS（多进程场景）
            if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
                torch.backends.mps.is_available = lambda: False
                torch.backends.mps.is_built = lambda: False
                print("[SAFE_BOOT] MPS已禁用（多进程模式）")
        
        torch.set_num_threads(1)
        
    except Exception as e:
        print(f"[SAFE_BOOT] MPS 配置失败(非致命): {e}")

# macOS 平台下导入 CoreML 相关模块
if USE_COREML:
    try:
        from src.core.tagging.coreml_wd_vit_v3_tagger import CoreMLWDVitV3Tagger
        print("[SAFE_BOOT] macOS 环境检测到，将使用 CoreML 加速")
    except ImportError as e:
        print(f"[SAFE_BOOT] CoreML 导入失败，回退到 CPU 模式: {e}")
        USE_COREML = False

from src.core.logging.global_logger import get_logger

logger = get_logger("wd_vit_v3_tagger")

# 延迟导入 transformers 相关模块（非 macOS 平台使用）
AutoProcessor = None
AutoModelForImageClassification = None
CLIPProcessor = None
CLIPModel = None


class WDViTV3Tagger:
    """WD Vit Tagger v3 标签生成器
    支持多平台加速方案：
    - macOS (Apple Silicon): CoreML (ANE加速)
    - Linux/Windows: PyTorch (CUDA/MPS/CPU自动选择)
    """

    # 全局Core ML标签生成器实例缓存
    _coreml_tagger = None

    def __init__(self, device=None):
        # 先初始化logger
        self.logger = get_logger("wd_vit_v3_tagger")
        
        # macOS环境下默认使用CoreML
        if USE_COREML:
            self.coreml_mode = True
            self.device = "coreml"
            self.logger.info("macOS环境检测到，使用CoreML加速")
        else:
            self.coreml_mode = False
            # 选择设备（优先使用GPU）
            self._select_device(device)
        
        self.logger.info(f"WD Vit Tagger 使用设备: {self.device}")
        self.wd_model = None
        self.wd_processor = None
        self.clip_model = None
        self.clip_processor = None
        self.id2label = {}
        self.num_id2label = {}

        # 初始化标签列表
        if USE_COREML:
            self.logger.info("WD Vit Tagger 模块初始化完成，将使用CoreML模型")
        else:
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

    def _select_device(self, device=None):
        """自动选择最佳设备
        
        Args:
            device: 手动指定设备（None表示自动选择）
        """
        import platform
        
        # macOS环境下MPS存在锁竞争问题，默认使用CPU
        # 用户可以通过device参数手动指定'mps'来尝试使用MPS加速
        if device is not None:
            self.device = device
            if device == "mps" and platform.system() == "Darwin":
                self.logger.info("手动指定使用MPS设备（macOS环境下可能存在锁竞争风险）")
            return
        
        # 自动选择最佳设备
        if platform.system() == "Darwin":
            # 根据ENABLE_MPS开关决定
            if ENABLE_MPS:
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        self.device = "mps"
                        self.logger.info("✅ macOS环境下启用MPS加速（单进程模式）")
                        return
                except:
                    pass
            
            # 回退到CPU
            self.device = "cpu"
            self.logger.info("⚠️ macOS环境下使用CPU（MPS未启用或不可用）")
            return
        
        # 非macOS平台自动选择最佳设备
        try:
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"
                self.logger.info("MPS设备可用，将使用MPS加速")
            elif torch.cuda.is_available():
                self.device = "cuda"
                self.logger.info("CUDA设备可用，将使用CUDA加速")
            else:
                self.device = "cpu"
                self.logger.info("未检测到GPU设备，使用CPU")
        except ImportError:
            # 如果torch不可用，默认为CPU
            self.device = "cpu"
            self.logger.info("PyTorch不可用，使用CPU")

    def load_model(self, model_name="SmilingWolf/wd-vit-tagger-v3"):
        """加载WD Vit Tagger v3模型

        Args:
            model_name: 模型名称（仅非macOS平台使用）

        Returns:
            bool: 模型加载是否成功
        """
        # macOS环境下使用CoreML模型
        if USE_COREML:
            return self._load_coreml_model()
        
        # 非macOS平台使用PyTorch模型
        try:
            global AutoProcessor, AutoModelForImageClassification

            # 安全导入 transformers
            from transformers import (
                AutoProcessor,
                AutoModelForImageClassification,
                CLIPProcessor,
                CLIPModel,
            )
            print("[SAFE_BOOT] transformers 库安全导入成功")

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
            # 如果是MPS设备加载失败，尝试回退到CPU
            if self.device == "mps":
                self.logger.info("MPS设备加载失败，尝试回退到CPU")
                self.device = "cpu"
                return self.load_model(model_name)
            # 加载失败时使用简单标签生成方法
            self.logger.info("加载模型失败，使用简单标签生成方法")
            return False

    def _load_coreml_model(self):
        """加载CoreML模型（macOS专用）

        Returns:
            bool: 模型加载是否成功
        """
        try:
            # 检查CoreML标签生成器是否已初始化
            if self.__class__._coreml_tagger is None:
                # 构建CoreML模型路径（相对于项目根目录）
                coreml_model_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "coreml_models", "wd_tagger.mlpackage"
                )
                coreml_labels_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "coreml_models", "wd_tagger_labels.json"
                )
                
                self.logger.info(f"加载CoreML模型: {coreml_model_path}")
                
                # 使用超时机制加载CoreML模型，避免长时间阻塞
                import threading
                import time
                
                result = {"success": False, "tagger": None, "error": None}
                
                def load_model_thread():
                    try:
                        tagger = CoreMLWDVitV3Tagger(
                            model_path=coreml_model_path,
                            labels_path=coreml_labels_path
                        )
                        result["tagger"] = tagger
                        result["success"] = True
                    except Exception as e:
                        result["error"] = e
                        self.logger.error(f"CoreML模型加载线程失败: {e}")
                
                # 创建并启动加载线程
                thread = threading.Thread(target=load_model_thread, daemon=True)
                thread.start()
                
                # 设置最大等待时间（30秒）
                max_wait_time = 30
                wait_interval = 0.1
                elapsed_time = 0
                
                while thread.is_alive() and elapsed_time < max_wait_time:
                    time.sleep(wait_interval)
                    elapsed_time += wait_interval
                
                if thread.is_alive():
                    self.logger.error("CoreML模型加载超时，将回退到简单标签生成方法")
                    return False
                
                if result["success"] and result["tagger"]:
                    self.__class__._coreml_tagger = result["tagger"]
                    self.logger.info("CoreML模型加载成功")
                elif result["error"]:
                    self.logger.error(f"CoreML模型加载失败: {result['error']}")
                    return False
            else:
                self.logger.info("CoreML模型已缓存，复用现有实例")
            
            return True
        except Exception as e:
            self.logger.error(f"加载CoreML模型失败: {e}")
            # 回退到简单标签列表
            self.logger.info("CoreML加载失败，回退到简单标签列表")
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

            # 检查是否加载了CoreML模型（macOS专用）
            if USE_COREML and self.__class__._coreml_tagger is not None:
                # 使用CoreML模型生成标签
                self.logger.debug("使用CoreML模型生成标签")
                
                # 确保图像是RGB格式
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert("RGB")
                
                # 使用CoreML标签生成器
                tags = self.__class__._coreml_tagger.generate_tags(image, threshold)
                
                # 过滤标签
                filtered_tags = self._filter_tags(tags)
                # 打印前10个标签
                self.logger.info(
                    f"CoreML模型 前10个标签: {[t['tag'] for t in filtered_tags[:10]]}"
                )
            # 检查是否加载了PyTorch模型
            elif self.wd_model is not None and self.wd_processor is not None:
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
