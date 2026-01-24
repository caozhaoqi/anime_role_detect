import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

class Preprocessing:
    # 全局模型实例缓存
    _model_instance = None
    
    def __init__(self, model_path=None):
        """初始化预处理模块"""
        # 使用全局模型实例避免重复加载
        if not self.__class__._model_instance:
            if model_path:
                self.__class__._model_instance = YOLO(model_path)
            else:
                # 尝试加载YOLOv8-anime模型，如果没有则使用默认模型
                try:
                    self.__class__._model_instance = YOLO('yolov8s-anime.pt')
                except:
                    self.__class__._model_instance = YOLO('yolov8s.pt')
        
        self.model = self.__class__._model_instance
        # 图像标准化参数
        self.img_size = (224, 224)
    
    def detect_character(self, image_path):
        """使用YOLOv8检测图像中的角色主体"""
        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 使用YOLOv8检测
        results = self.model(img_rgb)
        
        # 提取检测结果
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                
                # 只保留人物或类似人物的检测结果
                # 注意：YOLOv8默认类别中，人物是类别0
                if cls == 0 and conf > 0.5:
                    boxes.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf
                    })
        
        return boxes
    
    def crop_character(self, image_path, boxes):
        """根据检测结果裁剪角色主体"""
        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 如果没有检测到角色，返回原始图像
        if not boxes:
            return img_rgb
        
        # 选择置信度最高的检测结果
        best_box = max(boxes, key=lambda x: x['confidence'])
        x1, y1, x2, y2 = best_box['bbox']
        
        # 确保坐标是整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 确保坐标在图像范围内
        h, w = img_rgb.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 裁剪图像
        cropped_img = img_rgb[y1:y2, x1:x2]
        
        return cropped_img
    
    def normalize_image(self, img):
        """标准化图像"""
        # 调整图像大小
        resized_img = cv2.resize(img, self.img_size)
        
        # 转换为PIL Image格式
        pil_img = Image.fromarray(resized_img)
        
        return pil_img
    
    def process(self, image_path):
        """完整的预处理流程"""
        try:
            # 检测角色
            boxes = self.detect_character(image_path)
            
            # 裁剪角色
            cropped_img = self.crop_character(image_path, boxes)
            
            # 标准化图像
            normalized_img = self.normalize_image(cropped_img)
            
            return normalized_img, boxes
        except Exception as e:
            print(f"预处理失败: {e}")
            # 如果处理失败，返回原始图像的标准化版本
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return self.normalize_image(img_rgb), []
            else:
                raise
    
    def process_multiple_characters(self, image_path, max_characters=5):
        """处理图片中的多个角色"""
        try:
            # 检测角色
            boxes = self.detect_character(image_path)
            
            # 按置信度排序，选择前max_characters个角色
            boxes.sort(key=lambda x: x['confidence'], reverse=True)
            boxes = boxes[:max_characters]
            
            # 对每个角色进行处理
            processed_characters = []
            for box in boxes:
                # 裁剪单个角色
                cropped_img = self.crop_single_character(image_path, box)
                
                # 标准化图像
                normalized_img = self.normalize_image(cropped_img)
                
                processed_characters.append({
                    'box': box['bbox'],
                    'confidence': box['confidence'],
                    'image': normalized_img
                })
            
            return processed_characters
        except Exception as e:
            print(f"多角色处理失败: {e}")
            return []
    
    def crop_single_character(self, image_path, box):
        """裁剪单个角色"""
        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 获取边界框坐标
        x1, y1, x2, y2 = box['bbox']
        
        # 确保坐标是整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 确保坐标在图像范围内
        h, w = img_rgb.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 裁剪图像
        cropped_img = img_rgb[y1:y2, x1:x2]
        
        return cropped_img

if __name__ == "__main__":
    # 测试预处理模块
    preprocessor = Preprocessing()
    
    # 测试图像路径（需要根据实际情况修改）
    test_image = "test.jpg"
    
    try:
        normalized_img, boxes = preprocessor.process(test_image)
        print(f"检测到 {len(boxes)} 个角色")
        print(f"预处理后的图像大小: {normalized_img.size}")
        
        # 保存预处理后的图像
        normalized_img.save("preprocessed_test.jpg")
        print("预处理后的图像已保存为 preprocessed_test.jpg")
    except Exception as e:
        print(f"测试失败: {e}")
