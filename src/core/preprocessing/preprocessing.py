import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 使用全局日志系统
from src.core.logging.global_logger import get_logger, log_system, log_error
logger = get_logger("preprocessing")

class Preprocessing:
    # 全局模型实例缓存
    _model_instance = None
    
    def __init__(self, model_path=None):
        """初始化预处理模块"""
        # 使用全局模型实例避免重复加载
        if not self.__class__._model_instance:
            if model_path:
                logger.info(f"加载指定模型: {model_path}")
                self.__class__._model_instance = YOLO(model_path)
                logger.info("模型加载成功")
            else:
                # 尝试加载YOLOv8-anime模型，如果没有则使用默认模型
                try:
                    logger.info("尝试加载YOLOv8-anime模型...")
                    self.__class__._model_instance = YOLO('yolov8s-anime.pt')
                    logger.info("YOLOv8-anime模型加载成功")
                except Exception as e:
                    logger.warning(f"YOLOv8-anime模型加载失败: {e}，使用默认模型")
                    self.__class__._model_instance = YOLO('yolov8s.pt')
                    logger.info("默认YOLOv8s模型加载成功")
        
        self.model = self.__class__._model_instance
        # 图像标准化参数
        self.img_size = (224, 224)
        logger.debug(f"预处理模块初始化完成，图像大小: {self.img_size}")
    
    def detect_character(self, image_path):
        """使用YOLOv8检测图像中的角色主体"""
        logger.debug(f"开始检测角色: {image_path}")
        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法加载图像: {image_path}")
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logger.debug(f"图像加载成功，大小: {img_rgb.shape}")
        
        # 使用YOLOv8检测
        results = self.model(img_rgb)
        
        # 提取检测结果
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                
                # 修改：降低置信度阈值到 0.25，提高召回率
                # 修改：放宽类别限制，不仅限于 person (0)，防止二次元角色被误识别
                if conf > 0.25:
                    boxes.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf
                    })
        
        logger.info(f"角色检测完成，检测到 {len(boxes)} 个角色")
        return boxes
    
    def crop_character(self, image_path, boxes):
        """根据检测结果裁剪角色主体"""
        logger.debug(f"开始裁剪角色: {image_path}")
        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法加载图像: {image_path}")
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 如果没有检测到角色，返回原始图像
        if not boxes:
            logger.warning("没有检测到角色，返回原始图像")
            return img_rgb
        
        # 选择置信度最高的检测结果
        best_box = max(boxes, key=lambda x: x['confidence'])
        x1, y1, x2, y2 = best_box['bbox']
        logger.debug(f"选择置信度最高的检测结果: 置信度={best_box['confidence']:.4f}, 边界框={[x1, y1, x2, y2]}")
        
        # 确保坐标是整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 确保坐标在图像范围内
        h, w = img_rgb.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        logger.debug(f"调整边界框坐标: [{x1}, {y1}, {x2}, {y2}]")
        
        # 裁剪图像
        cropped_img = img_rgb[y1:y2, x1:x2]
        logger.debug(f"角色裁剪完成，裁剪后大小: {cropped_img.shape}")
        
        return cropped_img
    
    def normalize_image(self, img):
        """标准化图像"""
        logger.debug(f"开始标准化图像，原始大小: {img.shape}")
        # 调整图像大小
        resized_img = cv2.resize(img, self.img_size)
        logger.debug(f"调整图像大小到: {self.img_size}")
        
        # 图像增强：增加对比度和亮度
        # 对比度增强
        alpha = 1.1  # 对比度因子
        beta = 10    # 亮度偏移
        enhanced_img = cv2.convertScaleAbs(resized_img, alpha=alpha, beta=beta)
        logger.debug(f"图像增强完成，对比度因子: {alpha}, 亮度偏移: {beta}")
        
        # 转换为PIL Image格式
        pil_img = Image.fromarray(enhanced_img)
        logger.debug(f"转换为PIL Image格式，最终大小: {pil_img.size}")
        
        return pil_img
    
    def process(self, image_path):
        """完整的预处理流程"""
        logger.info(f"开始完整预处理流程: {image_path}")
        try:
            # 检测角色
            boxes = self.detect_character(image_path)
            
            # 裁剪角色
            cropped_img = self.crop_character(image_path, boxes)
            
            # 标准化图像
            normalized_img = self.normalize_image(cropped_img)
            
            logger.info(f"预处理流程完成，检测到 {len(boxes)} 个角色")
            return normalized_img, boxes
        except Exception as e:
            logger.error(f"预处理失败: {e}")
            # 如果处理失败，返回原始图像的标准化版本
            img = cv2.imread(image_path)
            if img is not None:
                logger.warning("返回原始图像的标准化版本")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return self.normalize_image(img_rgb), []
            else:
                logger.error("无法加载图像，抛出异常")
                raise
    
    def process_multiple_characters(self, image_path, max_characters=5):
        """处理图片中的多个角色"""
        logger.info(f"开始多角色处理: {image_path}, max_characters={max_characters}")
        try:
            # 检测角色
            boxes = self.detect_character(image_path)
            
            # 按置信度排序，选择前max_characters个角色
            boxes.sort(key=lambda x: x['confidence'], reverse=True)
            selected_boxes = boxes[:max_characters]
            logger.info(f"选择前 {len(selected_boxes)} 个角色进行处理")
            
            # 对每个角色进行处理
            processed_characters = []
            for i, box in enumerate(selected_boxes):
                logger.debug(f"处理角色 {i+1}/{len(selected_boxes)}，置信度: {box['confidence']:.4f}")
                # 裁剪单个角色
                cropped_img = self.crop_single_character(image_path, box)
                
                # 标准化图像
                normalized_img = self.normalize_image(cropped_img)
                
                processed_characters.append({
                    'box': box['bbox'],
                    'confidence': box['confidence'],
                    'image': normalized_img
                })
            
            logger.info(f"多角色处理完成，成功处理 {len(processed_characters)} 个角色")
            return processed_characters
        except Exception as e:
            logger.error(f"多角色处理失败: {e}")
            return []
    
    def crop_single_character(self, image_path, box):
        """裁剪单个角色"""
        logger.debug(f"开始裁剪单个角色: {image_path}")
        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法加载图像: {image_path}")
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 获取边界框坐标
        x1, y1, x2, y2 = box['bbox']
        logger.debug(f"原始边界框: [{x1}, {y1}, {x2}, {y2}]")
        
        # 确保坐标是整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 确保坐标在图像范围内
        h, w = img_rgb.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        logger.debug(f"调整后边界框: [{x1}, {y1}, {x2}, {y2}]")
        
        # 裁剪图像
        cropped_img = img_rgb[y1:y2, x1:x2]
        logger.debug(f"单个角色裁剪完成，裁剪后大小: {cropped_img.shape}")
        
        return cropped_img

if __name__ == "__main__":
    # 测试预处理模块
    logger.info("开始测试预处理模块...")
    preprocessor = Preprocessing()
    
    # 测试图像路径（需要根据实际情况修改）
    test_image = "test.jpg"
    logger.info(f"测试图像路径: {test_image}")
    
    try:
        normalized_img, boxes = preprocessor.process(test_image)
        logger.info(f"检测到 {len(boxes)} 个角色")
        logger.info(f"预处理后的图像大小: {normalized_img.size}")
        
        # 保存预处理后的图像
        normalized_img.save("preprocessed_test.jpg")
        logger.info("预处理后的图像已保存为 preprocessed_test.jpg")
        logger.info("预处理模块测试成功!")
    except Exception as e:
        logger.error(f"测试失败: {e}")
