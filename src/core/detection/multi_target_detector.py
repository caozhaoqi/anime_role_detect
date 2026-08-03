#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标检测模块 - YOLOv8 人体检测 + 角色识别
实现"人脸/人体框选 + 角色识别"，支持一张图里同时识别多个角色
"""

import os
import sys
import json
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torchvision import transforms
from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("multi_target_detector")

# 添加项目根目录
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

class MultiTargetDetector:
    """多目标检测器 - YOLOv8 人体检测 + 角色分类"""

    def __init__(
        self, yolo_model: str = "yolov8n.pt", role_model_path: str = None, device: str = "mps"
    ):
        """
        初始化多目标检测器

        Args:
            yolo_model: YOLOv8 模型名称或路径
            role_model_path: 角色分类模型路径
            device: 推理设备
        """
        self.device = device

        # 加载 YOLOv8
        print("=" * 60)
        print("🔄 加载 YOLOv8 人体检测模型...")
        from ultralytics import YOLO

        self.yolo = YOLO(yolo_model)
        print("✅ YOLOv8 加载成功")

        # 加载角色分类模型
        print("\n🔄 加载角色分类模型...")
        self.role_model, self.class_names = self._load_role_model(role_model_path)
        print(f"✅ 角色分类模型加载成功: {len(self.class_names)} 个角色")

        # 图像预处理
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_role_model(self, model_path: str = None):
        """加载角色分类模型"""
        from torchvision import models

        if model_path is None:
            MODEL_NAME = "efficientnet_b3"
            MODEL_DIR = os.path.join(project_root, "models", MODEL_NAME)
            model_path = os.path.join(MODEL_DIR, "model_best.pth")

        with open(os.path.join(os.path.dirname(model_path), "training_results.json"), "r") as f:
            config = json.load(f)

        num_classes = config.get("num_classes", 74)

        model = models.efficientnet_b3(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            classifier_keys = [k for k in state_dict.keys() if k.startswith('classifier.')]
            if classifier_keys:
                classifiers = {}
                for key in classifier_keys:
                    parts = key.split('.')
                    layer_idx = int(parts[1])
                    param_name = parts[2]
                    if layer_idx not in classifiers:
                        classifiers[layer_idx] = {}
                    classifiers[layer_idx][param_name] = state_dict[key]
                
                layers = []
                layer_mapping = {}
                new_idx = 0
                for idx in sorted(classifiers.keys()):
                    params = classifiers[idx]
                    if 'running_mean' in params and 'running_var' in params:
                        layers.append(torch.nn.BatchNorm1d(params['weight'].shape[0]))
                    elif 'weight' in params:
                        weight_shape = params['weight'].shape
                        layers.append(torch.nn.Linear(weight_shape[1], weight_shape[0]))
                    layer_mapping[idx] = new_idx
                    new_idx += 1
                
                model.classifier = torch.nn.Sequential(*layers)
                
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('classifier.'):
                        parts = key.split('.')
                        old_idx = int(parts[1])
                        if old_idx in layer_mapping:
                            new_key = f'classifier.{layer_mapping[old_idx]}.{parts[2]}'
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    else:
                        new_state_dict[key] = value
                
                state_dict = new_state_dict
            
            model.load_state_dict(state_dict, strict=False)

        model = model.to("cpu")
        model.eval()

        class_names = config.get("class_names", [f"class_{i}" for i in range(num_classes)])

        return model, class_names

    def detect_and_classify(
        self,
        image: Image.Image,
        person_conf_threshold: float = 0.2,
        crop_size: Tuple[int, int] = (224, 224),
        debug: bool = False,
    ) -> dict:
        """
        检测人体并识别角色

        Args:
            image: PIL Image
            person_conf_threshold: 人体检测置信度阈值
            crop_size: 裁剪目标尺寸

        Returns:
            检测结果字典
        """
        results = {"image_size": image.size, "total_detections": 0, "detections": []}
        if debug:
            results["debug_boxes"] = []
            results["yolo_total_boxes"] = 0

        # YOLOv8 人体检测
        # workers=0 禁用多进程数据加载，避免 macOS 上 "invalid low watermark ratio" 错误
        yolo_results = self.yolo(image, verbose=False, workers=0)

        for result in yolo_results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            if debug:
                results["yolo_total_boxes"] += len(boxes)

            for box in boxes:
                # 安全检查
                if box.cls is None or len(box.cls) == 0:
                    continue
                if box.conf is None or len(box.conf) == 0:
                    continue
                if box.xyxy is None or len(box.xyxy) == 0:
                    continue

                # 获取类别 ID 和置信度
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                except (TypeError, IndexError):
                    continue

                # debug 模式：为每个原始框（过滤前）记录候选信息
                if debug:
                    dbg = {
                        "bbox": [0.0, 0.0, 0.0, 0.0],
                        "class_id": cls_id,
                        "raw_confidence": conf,
                        "passed_conf_threshold": False,
                        "cropped_role": False,
                        "is_known_character": False,
                        "kept": False,
                        "discard_reason": None,
                        "candidates": [],
                    }

                # 放宽类别过滤：COCO 预训练的 person 类对动漫角色漏检严重，
                # 因此接受所有非背景检测框（任何 cls_id），交由 EfficientNet 判定是否为已知角色。
                # 仍保留置信度阈值，过滤低质量框。
                if conf < person_conf_threshold:
                    if debug:
                        dbg["discard_reason"] = "below_conf_threshold"
                        results["debug_boxes"].append(dbg)
                    continue

                # 获取边界框坐标
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                except (TypeError, IndexError):
                    continue

                # 裁剪人体区域
                cropped = image.crop(bbox)

                # 调整大小并分类
                role_result = self._classify_crop(cropped)

                is_known = role_result.get("role", "unknown") != "unknown"

                # 仅保留被识别为已知角色的裁剪框，避免把家具/背景算进角色数
                if role_result.get("role", "unknown") == "unknown":
                    if debug:
                        dbg["bbox"] = bbox
                        dbg["passed_conf_threshold"] = True
                        dbg["cropped_role"] = True
                        dbg["is_known_character"] = False
                        dbg["discard_reason"] = "unknown_role_filtered"
                        dbg["candidates"] = role_result.get("candidates", [])
                        results["debug_boxes"].append(dbg)
                    continue

                # 添加检测结果
                detection = {
                    "bbox": bbox,
                    "person_confidence": conf,
                    "role_prediction": role_result,
                }

                results["detections"].append(detection)
                results["total_detections"] += 1

                if debug:
                    dbg["bbox"] = bbox
                    dbg["passed_conf_threshold"] = True
                    dbg["cropped_role"] = True
                    dbg["is_known_character"] = True
                    dbg["kept"] = True
                    dbg["discard_reason"] = None
                    dbg["candidates"] = role_result.get("candidates", [])
                    results["debug_boxes"].append(dbg)

        return results

    def _classify_crop(self, crop: Image.Image) -> dict:
        """对裁剪区域进行角色分类（返回 top-1 决策 + top-3 候选，用于 debug 模式）"""
        # 调整大小
        crop_resized = crop.resize((224, 224), Image.BILINEAR)

        # 转换为 tensor
        input_tensor = self.transform(crop_resized).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.role_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            num_classes = probs.shape[1]
            k = min(3, num_classes)
            topk_prob, topk_idx = torch.topk(probs, k=k, dim=1)

        # top-1 决策（与历史行为一致，argmax 不变）
        top_idx = topk_idx[0, 0].item()
        confidence = topk_prob[0, 0].item()
        role_name = (
            self.class_names[top_idx] if top_idx < len(self.class_names) else f"unknown_{top_idx}"
        )

        # top-k 候选（debug 用）
        candidates = []
        for j in range(k):
            c_idx = topk_idx[0, j].item()
            c_prob = float(topk_prob[0, j].item())
            c_name = self.class_names[c_idx] if c_idx < len(self.class_names) else f"unknown_{c_idx}"
            candidates.append({"role": c_name, "prob": c_prob})

        return {"role": role_name, "confidence": confidence, "class_id": top_idx, "candidates": candidates}

    def batch_detect_and_classify(
        self, images: List[Image.Image], person_conf_threshold: float = 0.2
    ) -> List[dict]:
        """批量检测和分类"""
        results = []
        for img in images:
            result = self.detect_and_classify(img, person_conf_threshold)
            results.append(result)
        return results


def main():
    """测试多目标检测器"""
    print("=" * 60)
    print("🔍 多目标检测测试")
    print("=" * 60)

    # 创建检测器
    detector = MultiTargetDetector(
        yolo_model="yolov8n.pt",
        device=(
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        ),
    )

    # 测试图像
    test_image_path = os.path.join(project_root, "data", "test_images")
    if os.path.exists(test_image_path):
        image_files = [f for f in os.listdir(test_image_path) if f.endswith((".jpg", ".png"))]

        if image_files:
            test_image = Image.open(os.path.join(test_image_path, image_files[0]))
            print(f"\n📷 测试图像: {image_files[0]}")

            # 检测
            results = detector.detect_and_classify(test_image)

            print(f"\n📋 检测结果:")
            print(f"   检测到 {results['total_detections']} 个人体区域")

            for i, det in enumerate(results["detections"]):
                role = det["role_prediction"]["role"]
                conf = det["role_prediction"]["confidence"]
                bbox = det["bbox"]
                print(
                    f"   [{i+1}] 角色: {role} ({conf:.2f}) | bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
                )
        else:
            print("\n⚠️ 测试图像目录为空")
    else:
        print("\n⚠️ 测试图像目录不存在")


if __name__ == "__main__":
    main()
