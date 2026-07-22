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
        # 先加载到 CPU，避免 PyTorch MPS 后端 UntypedStorage 的 bug
        # ("invalid low watermark ratio")，后续由 model.to(self.device) 移入目标设备
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        # 强制使用 CPU 设备，避免 PyTorch MPS 后端内存分配器 bug
        # ("invalid low watermark ratio")。YOLO 模型内部自行管理设备不影响。
        model = model.to("cpu")
        model.eval()

        # 获取类别名
        class_names = config.get("class_names", [f"class_{i}" for i in range(num_classes)])

        return model, class_names

    def detect_and_classify(
        self,
        image: Image.Image,
        person_conf_threshold: float = 0.2,
        crop_size: Tuple[int, int] = (224, 224),
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

        # YOLOv8 人体检测
        # workers=0 禁用多进程数据加载，避免 macOS 上 "invalid low watermark ratio" 错误
        yolo_results = self.yolo(image, verbose=False, workers=0)

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
                    
                # 获取类别 ID 和置信度
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                except (TypeError, IndexError):
                    continue

                # 只处理人体 (COCO person class = 0)
                if cls_id != 0:
                    continue

                if conf < person_conf_threshold:
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

                # 添加检测结果
                detection = {
                    "bbox": bbox,
                    "person_confidence": conf,
                    "role_prediction": role_result,
                }

                results["detections"].append(detection)
                results["total_detections"] += 1

        return results

    def _classify_crop(self, crop: Image.Image) -> dict:
        """对裁剪区域进行角色分类"""
        # 调整大小
        crop_resized = crop.resize((224, 224), Image.BILINEAR)

        # 转换为 tensor
        input_tensor = self.transform(crop_resized).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.role_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)

        # 获取结果
        top_idx = top_idx[0].item()
        confidence = top_prob[0].item()
        role_name = (
            self.class_names[top_idx] if top_idx < len(self.class_names) else f"unknown_{top_idx}"
        )

        return {"role": role_name, "confidence": confidence, "class_id": top_idx}

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
