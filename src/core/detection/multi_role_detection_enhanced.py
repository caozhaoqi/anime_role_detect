#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版多角色检测模块

集成 Open-set 识别（未知角色检测）和模糊样本记录功能
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional

from src.core.logging.global_logger import get_logger
from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.recognition.open_set_recognizer import OpenSetRecognizer
from src.core.feedback.ambiguous_sample_recorder import AmbiguousSampleRecorder

import torch
import torchvision.transforms as transforms
from torchvision import models

logger = get_logger("multi_role_detection_enhanced")

# 项目根目录（相对于当前文件向上三级：detection → core → src → project_root）
_project_file = os.path.abspath(__file__)
_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(_project_file)))
project_root = os.path.dirname(_src_dir)
src_root = _src_dir  # src 目录路径


class EnhancedMultiRoleDetector:
    """
    增强版多角色检测器

    集成功能：
    - YOLOv8 人体检测
    - Open-set 识别（未知角色检测）
    - 模糊样本记录
    - 特征向量相似度匹配
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        enable_open_set: bool = True,
        enable_fuzzy_record: bool = True,
        unknown_threshold: float = 0.3,
        fuzzy_threshold: float = 0.5,
    ):
        """
        初始化增强版多角色检测器

        Args:
            model_name: 模型名称
            enable_open_set: 是否启用 Open-set 识别
            enable_fuzzy_record: 是否启用模糊样本记录
            unknown_threshold: 未知角色阈值（相似度低于此值判定为未知）
            fuzzy_threshold: 模糊样本阈值（相似度低于此值记录为模糊）
        """
        self.model_name = model_name
        self.enable_open_set = enable_open_set
        self.enable_fuzzy_record = enable_fuzzy_record

        self.yolo_model = None
        self.extractor = None
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.models_initialized = False

        self.open_set_recognizer = None
        self.fuzzy_recorder = None

        self.unknown_threshold = unknown_threshold
        self.fuzzy_threshold = fuzzy_threshold

    def _lazy_initialize_models(self):
        """延迟初始化模型"""
        if not self.models_initialized:
            try:
                self._initialize_models()
                self.models_initialized = True
            except Exception as e:
                logger.error(f"模型初始化失败: {e}")

    def _initialize_models(self):
        """初始化所有模型"""
        logger.info("=" * 60)
        logger.info("🔄 初始化增强版多角色检测器...")

        # 1. 初始化 Open-set 识别器
        if self.enable_open_set:
            logger.info("初始化 Open-set 识别器...")
            try:
                MODEL_NAME = self.model_name
                MODEL_DIR = os.path.join(project_root, "models", MODEL_NAME)
                index_path = os.path.join(MODEL_DIR, "role_index_final.faiss")
                mapping_path = os.path.join(MODEL_DIR, "role_index_final_mapping.json")
                role_info_path = os.path.join(project_root, "src", "core", "data", "role_info.json")

                self.open_set_recognizer = OpenSetRecognizer(
                    index_path=index_path,
                    mapping_path=mapping_path,
                    role_info_path=role_info_path,
                    unknown_threshold=self.unknown_threshold,
                    fuzzy_threshold=self.fuzzy_threshold,
                )
                logger.info("✅ Open-set 识别器初始化成功")
            except Exception as e:
                logger.warning(f"⚠️ Open-set 识别器初始化失败: {e}")
                self.enable_open_set = False

        # 2. 初始化模糊样本记录器
        if self.enable_fuzzy_record:
            logger.info("初始化模糊样本记录器...")
            try:
                self.fuzzy_recorder = AmbiguousSampleRecorder(fuzzy_low=0.5, fuzzy_high=0.7)
                logger.info("✅ 模糊样本记录器初始化成功")
            except Exception as e:
                logger.warning(f"⚠️ 模糊样本记录器初始化失败: {e}")
                self.enable_fuzzy_record = False

        # 3. 初始化 YOLOv8
        logger.info("初始化 YOLOv8 人体检测模型...")
        try:
            from ultralytics import YOLO

            self.yolo_model = YOLO("yolov8n.pt")
            logger.info("✅ YOLOv8 模型加载成功")
        except Exception as e:
            logger.error(f"❌ YOLOv8 模型加载失败: {e}")
            self.yolo_model = None

        # 4. 加载角色分类模型
        logger.info("加载角色分类模型...")
        self._load_trained_model()

        # 5. 初始化特征提取器
        logger.info("初始化特征提取器...")
        try:
            self.extractor = FeatureExtraction()
            logger.info(f"✅ 特征提取器初始化成功 (设备: {self.extractor.device})")
        except Exception as e:
            logger.warning(f"⚠️ 特征提取器初始化失败: {e}")

        logger.info("=" * 60)
        logger.info("✅ 增强版多角色检测器初始化完成")
        logger.info("=" * 60)

    def _load_trained_model(self):
        """加载训练好的模型"""
        try:
            MODEL_NAME = self.model_name
            MODEL_DIR = os.path.join(project_root, "models", MODEL_NAME)
            model_path = os.path.join(MODEL_DIR, "model_best.pth")

            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return

            model_data = torch.load(
                model_path, map_location=torch.device("cpu"), weights_only=False
            )

            self.class_to_idx = model_data.get("class_to_idx", {})

            if not self.class_to_idx:
                training_results_path = os.path.join(MODEL_DIR, "training_results.json")
                if os.path.exists(training_results_path):
                    import json

                    with open(training_results_path, "r", encoding="utf-8") as f:
                        training_results = json.load(f)
                    class_names = training_results.get("class_names", [])
                    if class_names:
                        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            num_classes = len(self.class_to_idx) if self.class_to_idx else 74

            model_full_path = os.path.join(MODEL_DIR, "model_best.pth")
            if os.path.exists(model_full_path):
                try:
                    full_model = torch.load(
                        model_full_path, map_location=torch.device("cpu"), weights_only=False
                    )
                    if isinstance(full_model, torch.nn.Module):
                        self.model = full_model
                        logger.info("✅ 完整模型加载成功")
                        self.model.eval()
                        return
                except Exception as e:
                    logger.warning(f"完整模型加载失败: {e}")

            self.model = models.efficientnet_b3(num_classes=num_classes)

            if "model_state_dict" in model_data:
                self.model.load_state_dict(model_data["model_state_dict"], strict=False)

            self.model.eval()
            logger.info(f"✅ 角色分类模型加载成功 ({num_classes} 个类别)")

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """预处理图像"""
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform(image).unsqueeze(0)

    def _classify_with_open_set(self, role_image: Image.Image) -> Dict[str, Any]:
        """
        使用 Open-set 识别进行角色分类

        Returns:
            包含 role, similarity, decision 等信息的字典
        """
        result = {
            "role": "unknown",
            "similarity": 0.0,
            "confidence": 0.0,
            "decision": "unknown",
            "is_unknown": True,
            "is_fuzzy": False,
        }

        # 1. 使用特征提取 + FAISS 搜索
        if self.open_set_recognizer and self.extractor:
            try:
                feature = self.extractor.extract_features(role_image)
                faiss_result = self.open_set_recognizer.recognize(feature, top_k=1)

                if faiss_result["predictions"]:
                    top_pred = faiss_result["predictions"][0]
                    result["similarity"] = top_pred["similarity"]
                    result["role"] = top_pred["role"]
                    result["confidence"] = top_pred["similarity"]
                    result["decision"] = faiss_result["decision"]
                    result["is_unknown"] = faiss_result["is_unknown"]
                    result["is_fuzzy"] = faiss_result["is_fuzzy"]

                    return result
            except Exception as e:
                logger.warning(f"Open-set 识别失败: {e}")

        # 2. 回退到模型分类
        if self.model and self.class_to_idx:
            try:
                input_tensor = self._preprocess_image(role_image)

                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    max_prob, predicted_idx = torch.max(probabilities, 1)

                idx = predicted_idx.item()
                result["role"] = self.idx_to_class.get(idx, "unknown")
                result["similarity"] = max_prob.item()
                result["confidence"] = max_prob.item()

                if result["similarity"] < self.unknown_threshold:
                    result["decision"] = "unknown"
                    result["is_unknown"] = True
                elif result["similarity"] < self.fuzzy_threshold:
                    result["decision"] = "fuzzy"
                    result["is_fuzzy"] = True
                else:
                    result["decision"] = "known"

            except Exception as e:
                logger.error(f"模型分类失败: {e}")

        return result

    def _classify_role(self, role_image: Image.Image) -> Tuple[str, float]:
        """使用训练模型分类角色（兼容性方法）"""
        result = self._classify_with_open_set(role_image)
        return result["role"], result["similarity"]

    def detect_roles(
        self, image_path: str, max_characters: int = 10, person_conf_threshold: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        检测图像中的多个角色

        Args:
            image_path: 图像路径
            max_characters: 最大检测数量
            person_conf_threshold: 人体检测置信度阈值

        Returns:
            角色检测和分类结果列表
        """
        results = []

        try:
            logger.info(f"🔍 开始检测角色，图像路径: {image_path}")

            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)

            self._lazy_initialize_models()

            detected_count = 0

            # 1. 使用 YOLOv8 检测人体
            if self.yolo_model:
                yolo_results = self.yolo_model(image, verbose=False)

                for result in yolo_results:
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0:
                        continue

                    for box in boxes:
                        # 安全检查
                        if box.cls is None or len(box.cls) == 0:
                            continue
                        if box.conf is None or len(box.conf) == 0:
                            continue
                        if box.xyxy is None or len(box.xyxy) == 0:
                            continue
                            
                        try:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                        except (TypeError, IndexError):
                            continue
                            
                        if cls_id != 0:
                            continue

                        if conf < person_conf_threshold:
                            continue

                        try:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = [float(x1), float(y1), float(x2), float(y2)]
                        except (TypeError, IndexError):
                            continue

                        cropped = image.crop(bbox)

                        classification = self._classify_with_open_set(cropped)

                        detection = {
                            "role": classification["role"],
                            "role_cn": classification.get("role_cn", classification["role"]),
                            "role_jp": classification.get("role_jp", ""),
                            "role_anime": classification.get("role_anime", ""),
                            "similarity": float(classification["similarity"]),
                            "confidence": float(classification["confidence"]),
                            "decision": classification["decision"],
                            "is_unknown": classification["is_unknown"],
                            "is_fuzzy": classification["is_fuzzy"],
                            "bbox": {
                                "x1": int(bbox[0]),
                                "y1": int(bbox[1]),
                                "x2": int(bbox[2]),
                                "y2": int(bbox[3]),
                            },
                            "box": bbox,
                            "attributes": [],
                        }

                        results.append(detection)
                        detected_count += 1

                        if detected_count >= max_characters:
                            break

                    if detected_count >= max_characters:
                        break

            # 2. 如果没有检测到，使用整图
            if len(results) == 0:
                logger.info("未检测到人体，使用整图进行分类")
                classification = self._classify_with_open_set(image)

                results.append(
                    {
                        "role": classification["role"],
                        "role_cn": classification.get("role_cn", classification["role"]),
                        "role_jp": classification.get("role_jp", ""),
                        "role_anime": classification.get("role_anime", ""),
                        "similarity": float(classification["similarity"]),
                        "confidence": float(classification["confidence"]),
                        "decision": classification["decision"],
                        "is_unknown": classification["is_unknown"],
                        "is_fuzzy": classification["is_fuzzy"],
                        "bbox": {
                            "x1": 0,
                            "y1": 0,
                            "x2": int(image_np.shape[1]),
                            "y2": int(image_np.shape[0]),
                        },
                        "box": [0, 0, image_np.shape[1], image_np.shape[0]],
                        "attributes": [],
                    }
                )

            # 3. 记录模糊样本
            if self.enable_fuzzy_record and self.fuzzy_recorder:
                for detection in results:
                    if detection["is_fuzzy"]:
                        self.fuzzy_recorder.record_sample(
                            image=image,
                            prediction={
                                "role": detection["role"],
                                "confidence": detection["similarity"],
                                "decision": detection["decision"],
                            },
                            metadata={"bbox": detection["bbox"], "source": "multi_role_detection"},
                        )

            logger.info(f"✅ 角色检测完成，检测到 {len(results)} 个角色")

            # 统计识别结果
            unknown_count = sum(1 for r in results if r["is_unknown"])
            fuzzy_count = sum(1 for r in results if r["is_fuzzy"])
            known_count = len(results) - unknown_count - fuzzy_count

            logger.info(f"   已知角色: {known_count}")
            logger.info(f"   模糊样本: {fuzzy_count}")
            logger.info(f"   未知角色: {unknown_count}")

        except Exception as e:
            logger.error(f"❌ 检测角色失败: {e}")
            import traceback

            logger.error(traceback.format_exc())

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """获取识别统计信息"""
        stats = {
            "open_set_enabled": self.enable_open_set,
            "fuzzy_record_enabled": self.enable_fuzzy_record,
            "unknown_threshold": self.unknown_threshold,
            "fuzzy_threshold": self.fuzzy_threshold,
        }

        if self.fuzzy_recorder:
            stats["fuzzy_samples"] = self.fuzzy_recorder.get_statistics()

        return stats


def main():
    """测试增强版多角色检测器"""
    print("=" * 60)
    print("🔍 增强版多角色检测器测试")
    print("=" * 60)

    detector = EnhancedMultiRoleDetector(
        enable_open_set=True, enable_fuzzy_record=True, unknown_threshold=0.3, fuzzy_threshold=0.5
    )

    test_images_dir = os.path.join(project_root, "data", "test_images")
    if os.path.exists(test_images_dir):
        image_files = [f for f in os.listdir(test_images_dir) if f.endswith((".jpg", ".png"))]
        if image_files:
            test_image = os.path.join(test_images_dir, image_files[0])
            results = detector.detect_roles(test_image)

            print(f"\n📋 检测结果 ({len(results)} 个角色):")
            for i, r in enumerate(results):
                decision_emoji = "❓" if r["is_unknown"] else ("❔" if r["is_fuzzy"] else "✅")
                print(
                    f"   [{i+1}] {decision_emoji} {r['role']} (相似度: {r['similarity']:.3f}, 决策: {r['decision']})"
                )
        else:
            print("\n⚠️ 测试图像目录为空")
    else:
        print("\n⚠️ 测试图像目录不存在")


if __name__ == "__main__":
    main()
