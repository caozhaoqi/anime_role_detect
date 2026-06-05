"""
YOLO自动标注系统
YOLO-based Automatic Annotation System
"""
# 必须在导入任何其他模块之前设置环境变量
import os
import sys
import platform
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Mac平台禁用CUDA，避免mutex错误
if platform.system() == "Darwin":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["FORCE_CPU"] = "1"

import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️ 未安装ultralytics库，请运行: pip install ultralytics")
    raise


class YOLODetector:
    """YOLO角色检测器"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型路径或名称
        """
        import torch
        # Mac平台优先使用MPS，否则使用CPU
        if platform.system() == "Darwin":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"📥 正在加载YOLO模型: {model_path}")
        self.model = YOLO(model_path)
        print(f"✅ YOLO模型加载完成，运行设备: {self.device}")
    
    def detect(self, image_path: str, conf_threshold: float = 0.5, 
               nms_threshold: float = 0.45) -> List[Dict]:
        """
        检测图片中的角色
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
        
        Returns:
            检测结果列表，每个结果包含:
            - bbox: [x1, y1, x2, y2]
            - confidence: 置信度
            - class_id: 类别ID
            - class_name: 类别名称
        """
        results = []
        
        try:
            # 执行检测
            detections = self.model.predict(
                source=image_path,
                conf=conf_threshold,
                iou=nms_threshold,
                device=self.device,
                verbose=False
            )
            
            for result in detections:
                for box in result.boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # 获取置信度和类别
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names.get(class_id, str(class_id))
                    
                    results.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name
                    })
            
        except Exception as e:
            print(f"⚠️ 检测失败 {image_path}: {str(e)}")
        
        return results
    
    def detect_batch(self, image_paths: List[str], conf_threshold: float = 0.5, 
                    nms_threshold: float = 0.45) -> Dict[str, List[Dict]]:
        """
        批量检测图片
        
        Args:
            image_paths: 图片路径列表
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
        
        Returns:
            检测结果字典，key为图片路径
        """
        results = {}
        
        for path in image_paths:
            detections = self.detect(path, conf_threshold, nms_threshold)
            results[path] = detections
        
        return results
    
    def crop_and_save(self, image_path: str, bbox: List[float], 
                      output_path: str) -> bool:
        """
        根据边界框裁剪图片并保存
        
        Args:
            image_path: 原始图片路径
            bbox: 边界框 [x1, y1, x2, y2]
            output_path: 输出路径
        
        Returns:
            是否成功
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️ 无法读取图片: {image_path}")
                return False
            
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
            
            # 裁剪
            cropped = image[y1:y2, x1:x2]
            
            # 保存
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, cropped)
            
            return True
        
        except Exception as e:
            print(f"⚠️ 裁剪失败 {image_path}: {str(e)}")
            return False
    
    def draw_bboxes(self, image_path: str, detections: List[Dict], 
                    output_path: str) -> bool:
        """
        在图片上绘制边界框并保存
        
        Args:
            image_path: 原始图片路径
            detections: 检测结果列表
            output_path: 输出路径
        
        Returns:
            是否成功
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️ 无法读取图片: {image_path}")
                return False
            
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                confidence = det["confidence"]
                class_name = det["class_name"]
                
                # 绘制边界框
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # 绘制标签
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(image, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 保存
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, image)
            
            return True
        
        except Exception as e:
            print(f"⚠️ 绘制边界框失败 {image_path}: {str(e)}")
            return False
    
    def get_person_count(self, image_path: str, conf_threshold: float = 0.5) -> int:
        """
        获取图片中的人数
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
        
        Returns:
            人数
        """
        detections = self.detect(image_path, conf_threshold)
        
        # 统计person类别（通常class_id为0）
        person_count = 0
        for det in detections:
            if det["class_name"].lower() == "person" or det["class_id"] == 0:
                person_count += 1
        
        return person_count
    
    def get_bbox_area_ratio(self, image_path: str, conf_threshold: float = 0.5) -> float:
        """
        获取最大检测框面积占图片面积的比例
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
        
        Returns:
            面积比例
        """
        detections = self.detect(image_path, conf_threshold)
        
        if not detections:
            return 0.0
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 0.0
            
            image_area = image.shape[0] * image.shape[1]
            
            max_bbox_area = 0
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                bbox_area = (x2 - x1) * (y2 - y1)
                max_bbox_area = max(max_bbox_area, bbox_area)
            
            return max_bbox_area / image_area
        
        except Exception as e:
            print(f"⚠️ 计算面积比例失败 {image_path}: {str(e)}")
            return 0.0


# 示例用法
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO角色检测工具")
    parser.add_argument("-i", "--input", required=True, help="输入图片路径或目录")
    parser.add_argument("-o", "--output", help="输出目录")
    parser.add_argument("-m", "--model", default="yolov8n.pt", help="YOLO模型路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS阈值")
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = YOLODetector(args.model)
    
    # 获取图片列表
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [str(input_path)]
    else:
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
        image_paths = [str(p) for p in input_path.rglob('*') if p.suffix.lower() in image_extensions]
    
    print(f"📁 找到 {len(image_paths)} 张图片")
    
    # 检测
    results = detector.detect_batch(image_paths, args.conf, args.nms)
    
    # 处理结果
    for path, detections in results.items():
        print(f"\n📊 {path}:")
        if detections:
            for i, det in enumerate(detections):
                print(f"   检测 {i+1}: {det['class_name']} ({det['confidence']:.2f})")
                print(f"      边界框: {det['bbox']}")
            
            # 如果指定了输出目录，绘制边界框
            if args.output:
                output_path = Path(args.output) / Path(path).name
                detector.draw_bboxes(path, detections, str(output_path))
                print(f"      已保存到: {output_path}")
        else:
            print("   未检测到任何目标")