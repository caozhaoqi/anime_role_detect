import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

# 使用全局日志系统
from core.logging.global_logger import get_logger, log_system, log_error
logger = get_logger("feature_extraction")

class FeatureExtraction:
    # 全局模型实例缓存
    _model_instance = None
    _processor_instance = None
    _device = None
    _quantized = False
    _model_name = None
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", quantize=False):
        """初始化特征提取模块
        
        Args:
            model_name: 模型名称
            quantize: 是否使用量化模型
        """
        # 检查是否需要重新加载模型
        if not self.__class__._model_instance or self.__class__._model_name != model_name:
            logger.info(f"加载特征提取模型: {model_name}")
            self.__class__._model_instance = CLIPModel.from_pretrained(model_name)
            self.__class__._processor_instance = CLIPProcessor.from_pretrained(model_name)
            self.__class__._device = "cuda" if torch.cuda.is_available() else "cpu"
            self.__class__._model_instance.to(self.__class__._device)
            self.__class__._model_name = model_name
            
            # 量化模型以减少内存使用和提高推理速度
            if quantize:
                logger.info("开始模型量化...")
                self.__class__._model_instance = torch.quantization.quantize_dynamic(
                    self.__class__._model_instance,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                self.__class__._quantized = True
                logger.info("模型量化完成")
        
        self.model = self.__class__._model_instance
        self.processor = self.__class__._processor_instance
        self.device = self.__class__._device
        
        # 设置模型为评估模式
        self.model.eval()
    
    def extract_features(self, img):
        """提取图像特征"""
        try:
            # 检查输入图像
            if img is None:
                raise ValueError("输入图像为None")
            
            # 预处理图像
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            
            # 提取特征
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            
            # 归一化特征向量
            norm = features.norm(dim=-1, keepdim=True)
            # 防止除以零
            if norm.item() > 1e-10:
                features = features / norm
            else:
                # 如果范数为零，使用随机向量
                logger.warning("特征向量范数为零，使用随机向量")
                features = torch.randn_like(features)
                features = features / features.norm(dim=-1, keepdim=True)
            
            # 转换为numpy数组
            features_np = features.cpu().numpy().squeeze()
            
            return features_np
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            raise
    
    def batch_extract_features(self, imgs, batch_size=8):
        """批量提取图像特征
        
        Args:
            imgs: 图像列表
            batch_size: 批量大小
            
        Returns:
            特征向量列表
        """
        try:
            # 检查输入图像列表
            if not imgs:
                return []
            
            # 分批处理
            all_features = []
            for i in range(0, len(imgs), batch_size):
                batch_imgs = imgs[i:i+batch_size]
                
                # 预处理图像
                inputs = self.processor(images=batch_imgs, return_tensors="pt", padding=True).to(self.device)
                
                # 提取特征
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                
                # 归一化特征向量
                features = features / features.norm(dim=-1, keepdim=True)
                
                # 转换为numpy数组
                features_np = features.cpu().numpy()
                all_features.append(features_np)
            
            # 合并所有批次的特征
            if all_features:
                return np.vstack(all_features)
            else:
                return np.array([])
        except Exception as e:
            logger.error(f"批量特征提取失败: {e}")
            raise
    
    def extract_features_from_multiple_characters(self, characters, batch_size=8):
        """从多个角色中提取特征
        
        Args:
            characters: 角色字典列表
            batch_size: 批量大小
            
        Returns:
            添加了特征的角色字典列表
        """
        try:
            # 检查输入角色列表
            if not characters:
                return []
            
            # 提取所有角色的图像
            imgs = [char['image'] for char in characters if 'image' in char]
            
            # 批量提取特征
            features = self.batch_extract_features(imgs, batch_size=batch_size)
            
            # 将特征与角色信息关联
            feature_idx = 0
            for char in characters:
                if 'image' in char:
                    char['feature'] = features[feature_idx]
                    feature_idx += 1
            
            return characters
        except Exception as e:
            logger.error(f"多角色特征提取失败: {e}")
            return []

if __name__ == "__main__":
    # 测试特征提取模块
    extractor = FeatureExtraction()
    
    # 测试图像路径（需要根据实际情况修改）
    test_image = "test.jpg"
    
    try:
        # 加载图像
        img = Image.open(test_image)
        
        # 提取特征
        features = extractor.extract_features(img)
        
        logger.info(f"特征向量维度: {features.shape}")
        logger.info(f"特征向量前10个元素: {features[:10]}")
        logger.info("特征提取成功!")
    except Exception as e:
        logger.error(f"测试失败: {e}")
