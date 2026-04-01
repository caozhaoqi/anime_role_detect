#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多角色检测模块

用于检测图像中的多个角色，并对每个角色进行分类
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from core.logging.global_logger import get_logger
from core.feature_extraction.feature_extraction import FeatureExtraction
from core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
import torch
import torchvision.transforms as transforms
from torchvision import models

logger = get_logger("multi_role_detection")

class MultiRoleDetector:
    """
    多角色检测器
    
    使用YOLOv8进行目标检测，然后对每个检测到的角色进行分类
    """
    
    def __init__(self, model_name="efficientnet_b0"):
        """
        初始化多角色检测器
        
        Args:
            model_name: 要使用的训练模型名称
        """
        self.yolo_model = None
        self.extractor = None
        self.tagger = None
        self.model = None
        self.class_to_idx = None
        self.model_name = model_name
        self._initialize_models()
        self._load_trained_model()
    
    def _load_trained_model(self):
        """
        加载训练好的模型
        """
        try:
            # 模型路径映射
            model_paths = {
                "mobilenet_v2": "models/incremental/model_best.pth",
                "efficientnet_b0": "models/incremental_efficientnet_b0/model_best.pth",
                "efficientnet_b3": "models/incremental_efficientnet_b3/model_best.pth",
                "resnet50": "models/incremental_resnet50/model_best.pth"
            }
            
            # 检查模型是否存在
            if self.model_name not in model_paths:
                logger.error(f"不支持的模型类型: {self.model_name}")
                return
            
            model_path = model_paths[self.model_name]
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return
            
            # 加载模型数据
            model_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            self.class_to_idx = model_data.get('class_to_idx', {})
            
            # 加载模型
            if self.model_name == 'mobilenet_v2':
                self.model = models.mobilenet_v2(pretrained=False)
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.3),
                    torch.nn.Linear(self.model.classifier[1].in_features, 512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.Dropout(p=0.15),
                    torch.nn.Linear(512, len(self.class_to_idx))
                )
            elif self.model_name == 'efficientnet_b0':
                self.model = models.efficientnet_b0(pretrained=False)
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.3),
                    torch.nn.Linear(self.model.classifier[1].in_features, 512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.Dropout(p=0.15),
                    torch.nn.Linear(512, len(self.class_to_idx))
                )
            elif self.model_name == 'efficientnet_b3':
                self.model = models.efficientnet_b3(pretrained=False)
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.3),
                    torch.nn.Linear(self.model.classifier[1].in_features, 768),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.BatchNorm1d(768),
                    torch.nn.Dropout(p=0.15),
                    torch.nn.Linear(768, len(self.class_to_idx))
                )
            elif self.model_name == 'resnet50':
                self.model = models.resnet50(pretrained=False)
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.class_to_idx))
            
            # 加载模型权重
            self.model.load_state_dict(model_data['model_state_dict'])
            self.model.eval()
            logger.info(f"模型 {self.model_name} 加载完成，类别数: {len(self.class_to_idx)}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
    
    def _initialize_models(self):
        """
        初始化模型
        """
        try:
            # 尝试加载YOLOv8模型
            try:
                from ultralytics import YOLO
                # 使用预训练的YOLOv8n模型
                self.yolo_model = YOLO('yolov8n.pt')
                logger.info("YOLOv8模型加载成功")
            except Exception as e:
                logger.warning(f"YOLOv8模型加载失败: {e}")
                self.yolo_model = None
            
            # 初始化标签生成器
            self.tagger = WDViTV3Tagger()
            self.tagger.load_model()
            logger.info("标签生成器初始化完成")
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
    
    def _preprocess_image(self, image):
        """
        预处理图像以输入到模型
        
        Args:
            image: PIL图像对象
        
        Returns:
            预处理后的张量
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(image).unsqueeze(0)
    
    def detect_roles(self, image_path: str) -> List[Dict[str, Any]]:
        """
        检测图像中的多个角色
        
        Args:
            image_path: 图像路径
        
        Returns:
            角色检测和分类结果列表
        """
        results = []
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            
            # 使用YOLOv8进行目标检测
            if self.yolo_model:
                yolo_results = self.yolo_model(image_path)
                
                # 处理检测结果
                for result in yolo_results:
                    boxes = result.boxes
                    for box in boxes:
                        # 只处理人物类别（COCO数据集的类别0是person）
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            
                            # 裁剪角色图像
                            role_image = image.crop((x1, y1, x2, y2))
                            
                            # 生成标签
                            attributes = self.tagger.generate_tags(role_image)
                            
                            # 分类角色
                            role, similarity = self._classify_role(role_image)
                            
                            # 添加到结果
                            results.append({
                                "role": role,
                                "similarity": float(similarity),
                                "attributes": attributes,
                                "bbox": {
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2
                                },
                                "confidence": confidence
                            })
            else:
                # 如果YOLO模型加载失败，使用人脸检测
                results = self._detect_roles_with_face_detection(image_path, image_np)
        except Exception as e:
            logger.error(f"检测角色失败: {e}")
        
        return results
    
    def _classify_role(self, role_image):
        """
        使用训练模型分类角色
        
        Args:
            role_image: 角色图像
        
        Returns:
            (角色名称, 相似度)
        """
        if self.model and self.class_to_idx:
            try:
                # 预处理图像
                input_tensor = self._preprocess_image(role_image)
                
                # 模型推理
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    max_prob, predicted_idx = torch.max(probabilities, 1)
                    
                # 转换为角色名称
                idx_to_class = {v: k for k, v in self.class_to_idx.items()}
                role = idx_to_class.get(predicted_idx.item(), "unknown")
                similarity = max_prob.item()
                
                return role, similarity
            except Exception as e:
                logger.error(f"模型分类失败: {e}")
                return "unknown", 0.0
        else:
            # 如果模型未加载，返回未知
            return "unknown", 0.0
    
    def _detect_roles_with_face_detection(self, image_path: str, image_np: np.ndarray) -> List[Dict[str, Any]]:
        """
        使用人脸检测检测角色
        
        Args:
            image_path: 图像路径
            image_np: 图像的numpy数组
        
        Returns:
            角色检测和分类结果列表
        """
        results = []
        
        try:
            # 尝试使用OpenCV进行人脸检测
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # 转换为灰度图
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # 检测人脸
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # 处理检测结果
            for (x, y, w, h) in faces:
                # 扩展边界框，包含更多的头部区域
                x1 = max(0, x - int(w * 0.2))
                y1 = max(0, y - int(h * 0.3))
                x2 = min(image_np.shape[1], x + w + int(w * 0.2))
                y2 = min(image_np.shape[0], y + h + int(h * 0.1))
                
                # 裁剪角色图像
                role_image = Image.fromarray(image_np[y1:y2, x1:x2])
                
                # 生成标签
                attributes = self.tagger.generate_tags(role_image)
                
                # 分类角色
                role, similarity = self._classify_role(role_image)
                
                # 添加到结果
                results.append({
                    "role": role,
                    "similarity": float(similarity),
                    "attributes": attributes,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "confidence": 0.9  # 人脸检测的置信度
                })
        except Exception as e:
            logger.error(f"人脸检测失败: {e}")
        
        return results

if __name__ == "__main__":
    # 测试多角色检测
    detector = MultiRoleDetector()
    test_image = "test_image.jpg"
    if os.path.exists(test_image):
        results = detector.detect_roles(test_image)
        print(f"检测到 {len(results)} 个角色")
        for i, result in enumerate(results):
            print(f"角色 {i+1}: {result['role']}, 相似度: {result['similarity']}")
            print(f"边界框: {result['bbox']}")
    else:
        print(f"测试图像不存在: {test_image}")
