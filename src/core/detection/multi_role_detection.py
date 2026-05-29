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
        self.models_initialized = False
    
    def _lazy_initialize_models(self):
        """
        延迟初始化模型
        """
        if not self.models_initialized:
            # 先检查内存使用情况
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 70:
                logger.warning(f"内存使用率过高: {memory.percent:.2f}%，跳过模型初始化")
                return
            
            try:
                # 只初始化必要的模型，减少内存使用
                self._initialize_minimal_models()
                self.models_initialized = True
            except Exception as e:
                logger.error(f"模型初始化失败: {e}")
    
    def _initialize_minimal_models(self):
        """
        初始化最小化的模型集，减少内存使用
        """
        try:
            # 只初始化YOLOv8模型，用于角色检测
            from ultralytics import YOLO
            # 使用预训练的YOLOv8n模型（最小的模型）
            self.yolo_model = YOLO('yolov8n.pt')
            logger.info("YOLOv8模型加载成功")
            
            # 跳过标签生成器和分类模型的初始化，减少内存使用
            self.tagger = None
            self.model = None
            self.class_to_idx = None
            logger.info("使用最小化模型集，跳过标签生成器和分类模型的初始化")
        except Exception as e:
            logger.error(f"最小化模型初始化失败: {e}")
    
    def _load_trained_model(self):
        """
        加载训练好的模型
        """
        try:
            # 模型路径映射
            model_paths = {
                "mobilenet_v2": "models/mobilenet_v2/model_best.pth",
                "efficientnet_b0": "models/efficientnet_b0/model_best.pth",
                "resnet18": "models/resnet18/model_best.pth",
                "resnet18_loli8": "models/resnet18_loli8/model_best.pth",
                "mobilenet_v2_loli8": "models/mobilenet_v2_loli8/model_best.pth",
                "efficientnet_b0_loli8": "models/efficientnet_b0_loli8/model_best.pth",
                "efficientnet_b3_loli_optimized_v2_20260529_133654": "models/efficientnet_b3_loli_optimized_v2_20260529_133654/model_best.pth",
                "efficientnet_b3_loli_optimized_v2_20260522_165046": "models/efficientnet_b3_loli_optimized_v2_20260522_165046/model_best.pth"
            }
            
            # 处理默认模型
            model_name = self.model_name
            if model_name == "default":
                model_name = "efficientnet_b3_loli_optimized_v2_20260529_133654"
                logger.info(f"使用默认模型: {model_name}")
            
            # 检查模型是否存在
            if model_name not in model_paths:
                # 尝试动态构造路径
                dynamic_path = f"models/{model_name}/model_best.pth"
                if os.path.exists(dynamic_path):
                    logger.info(f"使用动态路径: {dynamic_path}")
                    model_path = dynamic_path
                else:
                    # 尝试提取基础模型名称
                    base_model_name = model_name.split('_loli8')[0]
                    if base_model_name in model_paths:
                        logger.info(f"使用基础模型 {base_model_name} 替代 {model_name}")
                        model_name = base_model_name
                        model_path = model_paths[model_name]
                    else:
                        logger.error(f"不支持的模型类型: {model_name}")
                        return
            else:
                model_path = model_paths[model_name]
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return
            
            # 加载模型数据
            model_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            
            # 尝试从模型数据中获取 class_to_idx
            self.class_to_idx = model_data.get('class_to_idx', {})
            
            # 如果 class_to_idx 为空，尝试从 training_results.json 文件中读取类别信息
            if not self.class_to_idx:
                model_dir = os.path.dirname(model_path)
                training_results_path = os.path.join(model_dir, 'training_results.json')
                if os.path.exists(training_results_path):
                    logger.info(f"从训练结果文件加载类别信息: {training_results_path}")
                    import json
                    with open(training_results_path, 'r', encoding='utf-8') as f:
                        training_results = json.load(f)
                    class_names = training_results.get('class_names', [])
                    if class_names:
                        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
                        logger.info(f"成功加载 {len(self.class_to_idx)} 个类别")
                else:
                    logger.error(f"训练结果文件不存在: {training_results_path}")
            
            # 尝试直接加载完整模型
            model_full_path = model_path.replace('model_best.pth', 'model_full.pth')
            if os.path.exists(model_full_path):
                logger.info(f"尝试加载完整模型文件: {model_full_path}")
                try:
                    full_model = torch.load(model_full_path, map_location=torch.device('cpu'), weights_only=False)
                    if isinstance(full_model, torch.nn.Module):
                        self.model = full_model
                        logger.info("成功加载完整模型")
                        self.model.eval()
                        
                        # 优化模型推理
                        if torch.cuda.is_available():
                            self.model = self.model.cuda()
                            logger.info(f"模型 {model_name} 已移至GPU")
                        
                        logger.info(f"模型 {model_name} 加载成功")
                        return
                    else:
                        logger.error(f"完整模型文件格式不正确: {type(full_model)}")
                except Exception as e:
                    logger.error(f"加载完整模型失败: {e}")
            
            # 如果无法加载完整模型，则尝试创建模型并加载权重
            logger.info(f"创建模型并加载权重: {model_name}")
            
            # 加载模型
            if 'mobilenet_v2' in model_name:
                self.model = models.mobilenet_v2(pretrained=False)
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.3),
                    torch.nn.Linear(self.model.classifier[1].in_features, 512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.Dropout(p=0.15),
                    torch.nn.Linear(512, len(self.class_to_idx))
                )
            elif 'efficientnet_b0' in model_name:
                self.model = models.efficientnet_b0(pretrained=False)
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.3),
                    torch.nn.Linear(self.model.classifier[1].in_features, 512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.Dropout(p=0.15),
                    torch.nn.Linear(512, len(self.class_to_idx))
                )
            elif model_name == 'efficientnet_b3':
                self.model = models.efficientnet_b3(pretrained=False)
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.3),
                    torch.nn.Linear(self.model.classifier[1].in_features, 768),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.BatchNorm1d(768),
                    torch.nn.Dropout(p=0.15),
                    torch.nn.Linear(768, len(self.class_to_idx))
                )
            elif model_name == 'resnet50':
                self.model = models.resnet50(pretrained=False)
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.class_to_idx))
            elif 'resnet18' in model_name:
                self.model = models.resnet18(pretrained=False)
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.class_to_idx))
            
            # 加载模型权重
            if 'model_state_dict' in model_data:
                self.model.load_state_dict(model_data['model_state_dict'])
                logger.info("成功加载模型权重")
            else:
                logger.error(f"模型文件中没有model_state_dict键")
            self.model.eval()
            
            # 优化模型推理
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info(f"模型 {model_name} 已移至GPU")
            
            logger.info(f"模型 {model_name} 加载完成，类别数: {len(self.class_to_idx)}")
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
            logger.info(f"开始检测角色，图像路径: {image_path}")
            
            # 加载图像
            logger.info("加载图像")
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            logger.info(f"图像加载成功，大小: {image.size}, 形状: {image_np.shape}")
            
            # 初始化模型
            logger.info("初始化模型")
            self._load_trained_model()
            logger.info(f"模型加载状态: model={self.model is not None}, class_to_idx={self.class_to_idx is not None}")
            
            # 直接使用人脸检测，避免YOLOv8模型的段错误问题
            logger.info("使用人脸检测")
            results = self._detect_roles_with_face_detection(image_path, image_np)
            
            # 如果人脸检测失败，尝试直接返回一个默认结果
            if len(results) == 0:
                logger.info("人脸检测未发现角色，尝试使用整个图像进行检测")
                # 使用整个图像作为角色
                role_image = image
                
                # 分类角色
                if self.model is not None and self.class_to_idx is not None:
                    role, similarity = self._classify_role(role_image)
                else:
                    role = "Unknown"
                    similarity = 0.0
                
                # 添加到结果
                results.append({
                    "role": role,
                    "similarity": float(similarity),
                    "attributes": [],
                    "bbox": {
                        "x1": 0,
                        "y1": 0,
                        "x2": int(image_np.shape[1]),
                        "y2": int(image_np.shape[0])
                    },
                    "confidence": 0.5  # 默认置信度
                })
                logger.info(f"添加默认角色检测结果: {role}, 相似度: {similarity}")
            
            # 确保至少返回一个结果
            if len(results) == 0:
                logger.info("所有检测方法都失败，返回默认角色")
                results.append({
                    "role": "Unknown",
                    "similarity": 0.0,
                    "attributes": [],
                    "bbox": {
                        "x1": 0,
                        "y1": 0,
                        "x2": int(image_np.shape[1]),
                        "y2": int(image_np.shape[0])
                    },
                    "confidence": 0.1  # 最低置信度
                })
        except Exception as e:
            logger.error(f"检测角色失败: {e}")
        
        logger.info(f"角色检测完成，检测到 {len(results)} 个角色")
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
                
                # 使用GPU进行推理（如果可用）
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                
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
            
            # 检测人脸 - 调整参数以提高检测率
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20), maxSize=(200, 200))
            
            logger.info(f"人脸检测完成，检测到 {len(faces)} 个人脸")
            
            # 处理检测结果
            for i, (x, y, w, h) in enumerate(faces):
                logger.info(f"检测到人脸 {i+1}: x={x}, y={y}, w={w}, h={h}")
                
                # 扩展边界框，包含更多的头部区域
                x1 = max(0, x - int(w * 0.2))
                y1 = max(0, y - int(h * 0.3))
                x2 = min(image_np.shape[1], x + w + int(w * 0.2))
                y2 = min(image_np.shape[0], y + h + int(h * 0.1))
                
                # 裁剪角色图像
                role_image = Image.fromarray(image_np[y1:y2, x1:x2])
                
                # 生成标签
                if self.tagger is not None:
                    attributes = self.tagger.generate_tags(role_image)
                else:
                    attributes = []
                
                # 分类角色
                if self.model is not None and self.class_to_idx is not None:
                    role, similarity = self._classify_role(role_image)
                else:
                    role = "Unknown"
                    similarity = 0.0
                
                # 添加到结果
                results.append({
                    "role": role,
                    "similarity": float(similarity),
                    "attributes": attributes,
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
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
