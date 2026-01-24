import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class FeatureExtraction:
    # 全局模型实例缓存
    _model_instance = None
    _processor_instance = None
    _device = None
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """初始化特征提取模块"""
        # 使用全局模型实例避免重复加载
        if not self.__class__._model_instance:
            self.__class__._model_instance = CLIPModel.from_pretrained(model_name)
            self.__class__._processor_instance = CLIPProcessor.from_pretrained(model_name)
            self.__class__._device = "cuda" if torch.cuda.is_available() else "cpu"
            self.__class__._model_instance.to(self.__class__._device)
        
        self.model = self.__class__._model_instance
        self.processor = self.__class__._processor_instance
        self.device = self.__class__._device
        
        # 设置模型为评估模式
        self.model.eval()
    
    def extract_features(self, img):
        """提取图像特征"""
        try:
            # 预处理图像
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            
            # 提取特征
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            
            # 归一化特征向量
            features = features / features.norm(dim=-1, keepdim=True)
            
            # 转换为numpy数组
            features_np = features.cpu().numpy().squeeze()
            
            return features_np
        except Exception as e:
            print(f"特征提取失败: {e}")
            raise
    
    def batch_extract_features(self, imgs):
        """批量提取图像特征"""
        try:
            # 预处理图像
            inputs = self.processor(images=imgs, return_tensors="pt", padding=True).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            
            # 归一化特征向量
            features = features / features.norm(dim=-1, keepdim=True)
            
            # 转换为numpy数组
            features_np = features.cpu().numpy()
            
            return features_np
        except Exception as e:
            print(f"批量特征提取失败: {e}")
            raise
    
    def extract_features_from_multiple_characters(self, characters):
        """从多个角色中提取特征"""
        try:
            # 提取所有角色的图像
            imgs = [char['image'] for char in characters]
            
            # 批量提取特征
            features = self.batch_extract_features(imgs)
            
            # 将特征与角色信息关联
            for i, char in enumerate(characters):
                char['feature'] = features[i]
            
            return characters
        except Exception as e:
            print(f"多角色特征提取失败: {e}")
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
        
        print(f"特征向量维度: {features.shape}")
        print(f"特征向量前10个元素: {features[:10]}")
        print("特征提取成功!")
    except Exception as e:
        print(f"测试失败: {e}")
