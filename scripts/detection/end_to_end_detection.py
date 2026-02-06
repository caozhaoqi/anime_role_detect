#!/usr/bin/env python3
"""
端到端角色检测与识别系统
集成YOLO目标检测和角色分类模型
"""
import os
import sys
import argparse
import logging
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('end_to_end_detection')

class YOLODetector:
    """
    YOLO目标检测器
    """
    
    def __init__(self, model_path='models/yolov8s.pt', conf_threshold=0.5):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
        """
        try:
            from ultralytics import YOLO
            
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            logger.info(f"YOLO模型加载成功: {model_path}")
            
        except ImportError as e:
            logger.error(f"缺少ultralytics库: {e}")
            logger.error("请运行: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"YOLO模型加载失败: {e}")
            raise
    
    def detect(self, image_path):
        """
        检测图像中的角色
        
        Args:
            image_path: 图像路径
            
        Returns:
            detections: 检测结果列表，每个元素包含边界框和置信度
        """
        try:
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return []
            
            # 检测
            results = self.model(image)
            
            # 打印原始结果
            logger.info(f"YOLO原始检测结果: {results}")
            
            # 处理结果
            detections = []
            for result in results:
                logger.info(f"结果类型: {type(result)}")
                logger.info(f"结果属性: {dir(result)}")
                
                if hasattr(result, 'boxes'):
                    logger.info(f"检测到 {len(result.boxes)} 个目标")
                    
                    # 使用更简单的方式访问边界框信息
                    boxes = result.boxes.data.cpu().numpy()
                    logger.info(f"边界框数据形状: {boxes.shape}")
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2, confidence, class_id = box
                        class_id = int(class_id)
                        
                        # 检查置信度
                        if confidence < self.conf_threshold:
                            continue
                        
                        # 打印类别信息
                        logger.info(f"检测到目标 {i}: 类别ID={class_id}, 置信度={confidence:.4f}, 位置=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                        
                        # 保留所有类别
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id
                        })
            
            logger.info(f"在图像中检测到 {len(detections)} 个角色")
            return detections
            
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return []

class RoleClassifier:
    """
    角色分类器
    """
    
    def __init__(self, model_path='models/character_classifier_best_improved.pth'):
        """
        初始化角色分类器
        
        Args:
            model_path: 分类模型路径
        """
        try:
            from torchvision import models
            
            # 加载模型
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 获取类别信息
            if 'class_to_idx' in checkpoint:
                self.class_to_idx = checkpoint['class_to_idx']
                self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
                num_classes = len(self.class_to_idx)
            else:
                logger.error("模型中未找到class_to_idx信息")
                raise ValueError("模型中未找到class_to_idx信息")
            
            # 创建模型
            self.model = models.efficientnet_b0(pretrained=False)
            self.model.classifier[1] = torch.nn.Linear(
                self.model.classifier[1].in_features,
                num_classes
            )
            
            # 加载权重
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 处理键名不匹配
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    name = k[9:]  # 移除 'backbone.'
                else:
                    name = k
                new_state_dict[name] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()
            
            # 图像预处理
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info(f"角色分类模型加载成功，包含 {num_classes} 个类别")
            
        except Exception as e:
            logger.error(f"角色分类模型加载失败: {e}")
            raise
    
    def classify(self, image):
        """
        分类角色
        
        Args:
            image: 裁剪后的角色图像
            
        Returns:
            class_name: 角色名称
            confidence: 置信度
        """
        try:
            # 预处理图像
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_tensor = self.transform(image).unsqueeze(0)
            
            # 预测
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)
                
            # 获取类别名称
            class_name = self.idx_to_class.get(predicted.item(), 'unknown')
            
            return class_name, confidence.item()
            
        except Exception as e:
            logger.error(f"分类失败: {e}")
            return 'unknown', 0.0

class EndToEndSystem:
    """
    端到端角色检测与识别系统
    """
    
    def __init__(self, yolo_model='models/yolov8s.pt', classifier_model='models/character_classifier_best_improved.pth', conf_threshold=0.5):
        """
        初始化端到端系统
        
        Args:
            yolo_model: YOLO模型路径
            classifier_model: 分类模型路径
            conf_threshold: 置信度阈值
        """
        self.detector = YOLODetector(yolo_model, conf_threshold)
        self.classifier = RoleClassifier(classifier_model)
    
    def process(self, image_path, output_path=None):
        """
        处理图像，检测并识别角色
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            
        Returns:
            results: 识别结果列表
        """
        try:
            # 加载原始图像
            original_image = cv2.imread(image_path)
            if original_image is None:
                logger.error(f"无法读取图像: {image_path}")
                return []
            
            # 检测角色
            detections = self.detector.detect(image_path)
            
            # 识别每个角色
            results = []
            for i, detection in enumerate(detections):
                # 获取边界框
                x1, y1, x2, y2 = detection['bbox']
                
                # 裁剪角色图像
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 确保边界框有效
                if x2 > x1 and y2 > y1:
                    # 裁剪
                    cropped = original_image[y1:y2, x1:x2]
                    
                    # 识别角色
                    class_name, confidence = self.classifier.classify(cropped)
                    
                    # 保存结果
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_name': class_name,
                        'confidence': confidence
                    })
                    
                    # 在图像上绘制结果
                    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(original_image, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 保存结果图像
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, original_image)
                logger.info(f"结果已保存到: {output_path}")
            
            logger.info(f"识别完成，共识别出 {len(results)} 个角色")
            return results
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            return []

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='端到端角色检测与识别系统')
    
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--output', type=str, default='output/end_to_end_result.jpg',
                       help='输出图像路径')
    parser.add_argument('--yolo-model', type=str, default='models/yolov8s.pt',
                       help='YOLO模型路径')
    parser.add_argument('--classifier-model', type=str, default='models/character_classifier_best_improved.pth',
                       help='角色分类模型路径')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='置信度阈值')
    
    args = parser.parse_args()
    
    # 初始化系统
    logger.info("初始化端到端系统...")
    system = EndToEndSystem(
        yolo_model=args.yolo_model,
        classifier_model=args.classifier_model,
        conf_threshold=args.conf_threshold
    )
    
    # 处理图像
    logger.info(f"处理图像: {args.image}")
    results = system.process(args.image, args.output)
    
    # 打印结果
    print("\n" + "="*60)
    print("角色检测与识别结果")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        bbox = result['bbox']
        print(f"{i}. 角色: {result['class_name']}")
        print(f"   置信度: {result['confidence']:.4f}")
        print(f"   位置: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        print()
    
    print(f"总计识别出 {len(results)} 个角色")
    print(f"结果图像已保存到: {args.output}")
    print("="*60)

if __name__ == '__main__':
    main()
