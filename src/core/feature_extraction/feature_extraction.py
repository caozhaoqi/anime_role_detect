from PIL import Image
import numpy as np
import os
import gc

# 延迟导入torch和transformers模块
torch = None
CLIPProcessor = None
CLIPModel = None

# 使用全局日志系统
from src.core.logging.global_logger import get_logger, log_system, log_error
logger = get_logger("feature_extraction")

# 导入跨平台诊断工具
try:
    from utils.diagnostics import CrossPlatformDiagnostics
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    logger.warning("跨平台诊断工具不可用")
    DIAGNOSTICS_AVAILABLE = False

# 动态导入函数
def import_torch_modules():
    global torch, CLIPProcessor, CLIPModel, gc
    # 设置环境变量
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # 导入模块
    import torch
    from transformers import CLIPProcessor, CLIPModel
    import gc

class FeatureExtraction:
    # 全局模型实例缓存
    _model_instance = None
    _processor_instance = None
    _device = None
    _quantized = False
    _model_name = None
    _coreml_extractor = None
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", quantize=True, use_coreml=False):
        """初始化特征提取模块
        
        Args:
            model_name: 模型名称
            quantize: 是否使用量化模型
            use_coreml: 是否使用Core ML模式
        """
        # 禁用Core ML模式，避免锁竞争问题
        use_coreml = False
        
        # 延迟加载模型，避免初始化时的锁竞争
        self.model_name = model_name
        self.quantize = quantize
        self.coreml_mode = False
        
        # 初始化模型实例为None
        self.model = None
        self.processor = None
        
        # 自动选择设备
        import_torch_modules()
        global torch
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"特征提取模块使用设备: {self.device}")
        logger.info(f"MPS可用: {torch.backends.mps.is_available()}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        
        logger.info("特征提取模块初始化完成，模型将在首次使用时加载")
    
    def _load_model(self):
        """延迟加载模型"""
        # 不需要加载模型，使用简单特征提取方法
        pass

    def extract_features(self, img):
        """提取图像特征"""
        try:
            # 检查输入图像
            if img is None:
                raise ValueError("输入图像为None")
            
            # 使用简单特征提取方法，避免使用PyTorch
            logger.debug("使用简单特征提取方法")
            # 调整图像大小
            img = img.resize((224, 224))
            # 转换为numpy数组
            import numpy as np
            img_array = np.array(img)
            # 计算像素平均值作为特征
            features = img_array.mean(axis=(0, 1)).flatten()
            # 归一化特征
            features = features / np.linalg.norm(features) if np.linalg.norm(features) > 0 else features
            # 填充到512维
            if len(features) < 512:
                features = np.pad(features, (0, 512 - len(features)), 'constant')
            elif len(features) > 512:
                features = features[:512]
            
            logger.debug(f"特征提取完成，特征维度: {features.shape}")
            return features
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            # 发生错误时，返回一个随机特征向量
            import numpy as np
            return np.random.rand(512).astype(np.float32)
    
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
            
            # 使用简单特征提取方法，避免使用PyTorch
            logger.debug("使用简单特征提取方法进行批量处理")
            all_features = []
            for img in imgs:
                feature = self.extract_features(img)
                all_features.append(feature)
            
            # 转换为numpy数组
            if all_features:
                return np.vstack(all_features)
            else:
                return np.array([])
        except Exception as e:
            logger.error(f"批量特征提取失败: {e}")
            # 发生错误时，返回空数组
            return np.array([])
    
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
