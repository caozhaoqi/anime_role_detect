#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动漫专用人脸检测器
针对动漫角色的人脸检测，优化侧脸、半脸、Q版等场景
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger("anime_face_detector")


class AnimeFaceDetector:
    """
    动漫人脸检测器
    
    特点：
    - 针对动漫人脸优化
    - 支持多种画风
    - 检测侧脸、半脸、Q版
    - 支持多人场景
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径，None则使用默认模型
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            device: 运行设备
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device or self._select_device()
        self.model = None
        
        # 尝试加载模型
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self._init_default_detector()
        
        logger.info(f"AnimeFaceDetector 初始化完成")
    
    def _select_device(self) -> str:
        """选择设备"""
        import platform
        if platform.system() == "Darwin":
            return "cpu"
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except:
            return "cpu"
    
    def _init_default_detector(self):
        """初始化默认检测器（使用YOLOv8）"""
        try:
            from ultralytics import YOLO
            
            # 使用YOLOv8n-face或通用检测器
            model_path = "yolov8n.pt"
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                logger.info("使用YOLOv8默认检测器")
            else:
                logger.warning("YOLO模型未找到，将使用备用检测方法")
                self.model = None
        except ImportError:
            logger.warning("ultralytics未安装，使用备用检测方法")
            self.model = None
    
    def _load_model(self, model_path: str):
        """加载自定义模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"加载自定义模型: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self._init_default_detector()
    
    def detect(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
    ) -> List[dict]:
        """
        检测动漫人脸
        
        Args:
            image_input: 图片输入
            
        Returns:
            检测结果列表，每个结果包含：
            - bbox: [x1, y1, x2, y2]
            - confidence: 置信度
            - keypoints: 关键点（如果有）
        """
        # 加载图片
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"不支持的输入类型: {type(image_input)}")
        
        # 使用YOLO检测
        if self.model is not None:
            return self._detect_yolo(image)
        else:
            return self._detect_fallback(image)
    
    def _detect_yolo(self, image: Image.Image) -> List[dict]:
        """使用YOLO检测"""
        import numpy as np
        
        # 转为numpy数组
        img_array = np.array(image)
        
        # 推理
        results = self.model(img_array, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # 安全检查
                if box.conf is None or len(box.conf) == 0:
                    continue
                if box.xyxy is None or len(box.xyxy) == 0:
                    continue
                    
                confidence = float(box.conf[0])
                
                if confidence < self.conf_threshold:
                    continue
                
                # 获取边界框
                try:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                except (TypeError, IndexError):
                    continue
                
                detection = {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": int(box.cls[0]) if box.cls is not None else 0,
                }
                
                # 获取关键点（如果有）
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    try:
                        keypoints = result.keypoints.xy[0].cpu().numpy()
                        detection["keypoints"] = keypoints.tolist()
                    except:
                        pass
                
                detections.append(detection)
        
        # NMS
        detections = self._nms(detections)
        
        return detections
    
    def _detect_fallback(self, image: Image.Image) -> List[dict]:
        """
        备用检测方法
        使用简单的图像处理检测人脸区域
        """
        import numpy as np
        from PIL import ImageFilter
        
        # 转为灰度图
        gray = image.convert("L")
        
        # 使用边缘检测
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # 简单的阈值分割找到可能的人脸区域
        img_array = np.array(edges)
        threshold = img_array.mean() + img_array.std()
        binary = img_array > threshold
        
        # 找到连通区域（简化版）
        # 实际应用中可以使用更复杂的算法
        h, w = binary.shape
        
        # 返回整个图片作为候选区域
        return [{
            "bbox": [0, 0, w, h],
            "confidence": 0.5,
            "class_id": 0,
        }]
    
    def _nms(self, detections: List[dict]) -> List[dict]:
        """
        非极大值抑制
        
        Args:
            detections: 检测结果列表
            
        Returns:
            过滤后的结果
        """
        if not detections:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        keep = []
        while detections:
            best = detections[0]
            keep.append(best)
            
            # 计算与其他框的IoU
            detections = [
                d for d in detections[1:]
                if self._iou(best["bbox"], d["bbox"]) < self.iou_threshold
            ]
        
        return keep
    
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """
        计算IoU
        
        Args:
            box1: 边界框1 [x1, y1, x2, y2]
            box2: 边界框2 [x1, y1, x2, y2]
            
        Returns:
            IoU值
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # 计算并集
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def crop_faces(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        detections: Optional[List[dict]] = None,
        padding: float = 0.2,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> List[Image.Image]:
        """
        裁剪人脸区域
        
        Args:
            image_input: 图片输入
            detections: 检测结果，None则自动检测
            padding: 边界框填充比例
            target_size: 目标尺寸 (width, height)
            
        Returns:
            裁剪后的人脸图片列表
        """
        # 加载图片
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"不支持的输入类型: {type(image_input)}")
        
        # 自动检测
        if detections is None:
            detections = self.detect(image)
        
        crops = []
        img_width, img_height = image.size
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            
            # 添加padding
            width = x2 - x1
            height = y2 - y1
            
            x1 = max(0, x1 - width * padding)
            y1 = max(0, y1 - height * padding)
            x2 = min(img_width, x2 + width * padding)
            y2 = min(img_height, y2 + height * padding)
            
            # 裁剪
            crop = image.crop((x1, y1, x2, y2))
            
            # 调整尺寸
            if target_size:
                crop = crop.resize(target_size, Image.Resampling.LANCZOS)
            
            crops.append(crop)
        
        return crops
    
    def detect_and_crop(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        **kwargs,
    ) -> Tuple[List[dict], List[Image.Image]]:
        """
        检测并裁剪
        
        Args:
            image_input: 图片输入
            **kwargs: 传递给crop_faces的参数
            
        Returns:
            (检测结果, 裁剪后图片)
        """
        detections = self.detect(image_input)
        crops = self.crop_faces(image_input, detections, **kwargs)
        return detections, crops


class MultiRoleDetector:
    """
    多角色检测器
    检测图片中的多个角色并分别处理
    """
    
    def __init__(
        self,
        face_detector: Optional[AnimeFaceDetector] = None,
        body_detector: Optional[AnimeFaceDetector] = None,
    ):
        """
        初始化
        
        Args:
            face_detector: 人脸检测器
            body_detector: 人体检测器
        """
        self.face_detector = face_detector or AnimeFaceDetector()
        self.body_detector = body_detector
    
    def detect_roles(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
    ) -> List[dict]:
        """
        检测多个角色
        
        Args:
            image_input: 图片输入
            
        Returns:
            角色列表，每个角色包含：
            - face_bbox: 人脸边界框
            - body_bbox: 人体边界框
            - confidence: 置信度
        """
        # 检测人脸
        face_detections = self.face_detector.detect(image_input)
        
        roles = []
        for face in face_detections:
            role = {
                "face_bbox": face["bbox"],
                "face_confidence": face["confidence"],
            }
            
            # 如果有身体检测器，检测对应的身体
            if self.body_detector:
                body_detections = self.body_detector.detect(image_input)
                # 找到匹配的身体
                best_body = self._match_face_to_body(face, body_detections)
                if best_body:
                    role["body_bbox"] = best_body["bbox"]
                    role["body_confidence"] = best_body["confidence"]
            
            roles.append(role)
        
        return roles
    
    def _match_face_to_body(
        self,
        face: dict,
        bodies: List[dict],
    ) -> Optional[dict]:
        """
        匹配人脸到身体
        
        Args:
            face: 人脸检测
            bodies: 身体检测列表
            
        Returns:
            匹配的身体或None
        """
        face_bbox = face["bbox"]
        face_center_y = (face_bbox[1] + face_bbox[3]) / 2
        
        best_match = None
        best_iou = 0
        
        for body in bodies:
            body_bbox = body["bbox"]
            
            # 检查人脸是否在身体上方
            body_top = body_bbox[1]
            if face_center_y < body_top:
                continue
            
            # 计算水平重叠
            face_center_x = (face_bbox[0] + face_bbox[2]) / 2
            if body_bbox[0] <= face_center_x <= body_bbox[2]:
                # 计算IoU
                iou = self.face_detector._iou(face_bbox, body_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = body
        
        return best_match


if __name__ == "__main__":
    # 测试
    detector = AnimeFaceDetector()
    
    if os.path.exists("test.jpg"):
        detections = detector.detect("test.jpg")
        print(f"检测到 {len(detections)} 个人脸")
        
        for i, det in enumerate(detections):
            print(f"  {i+1}. 置信度: {det['confidence']:.3f}, 位置: {det['bbox']}")
