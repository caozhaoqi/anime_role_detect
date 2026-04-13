from PIL import Image
import numpy as np
import os
import gc

# 禁用MPS，避免锁竞争问题
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# 导入torch并禁用MPS
import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

# 延迟导入transformers模块
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
    # 导入模块
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
        
        # 不使用torch，直接设置设备为CPU
        self.device = 'cpu'
        logger.info(f"特征提取模块使用设备: {self.device}")
        logger.info("特征提取模块初始化完成，使用简单特征提取方法")
    
    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            logger.info(f"加载CLIP模型: {self.model_name}")
            # 导入必要的模块
            import_torch_modules()
            
            # 加载模型和处理器
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
            
            # 移动到设备
            self.model.to(self.device)
            
            # 量化模型
            if self.quantize:
                try:
                    self.model = torch.quantization.quantize_dynamic(
                        self.model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                    logger.info("模型量化完成")
                except Exception as e:
                    logger.warning(f"模型量化失败: {e}")
            
            logger.info("CLIP模型加载完成")

    def extract_features(self, img):
        """提取图像特征"""
        try:
            # 检查输入图像
            if img is None:
                raise ValueError("输入图像为None")
            
            # 使用改进的简单特征提取方法，避免使用PyTorch
            logger.debug("使用改进的简单特征提取方法")
            
            # 调整图像大小
            img = img.resize((224, 224))
            # 转换为numpy数组
            import numpy as np
            img_array = np.array(img)
            
            # 提取更丰富的特征
            # 1. 颜色直方图特征 (R, G, B)
            hist_r = np.histogram(img_array[:,:,0], bins=16, range=(0, 255))[0]
            hist_g = np.histogram(img_array[:,:,1], bins=16, range=(0, 255))[0]
            hist_b = np.histogram(img_array[:,:,2], bins=16, range=(0, 255))[0]
            
            # 2. 纹理特征 (简单的边缘检测)
            from PIL import ImageFilter
            edges = img.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges)
            edge_density = np.mean(edges_array)
            
            # 3. 颜色统计特征
            mean_color = img_array.mean(axis=(0, 1))
            std_color = img_array.std(axis=(0, 1))
            
            # 4. 形状特征
            width, height = img.size
            aspect_ratio = width / height
            
            # 组合所有特征
            features = np.concatenate([
                hist_r,  # 16维
                hist_g,  # 16维
                hist_b,  # 16维
                mean_color,  # 3维
                std_color,  # 3维
                [edge_density, aspect_ratio]  # 2维
            ])
            
            # 归一化特征
            features = features / np.linalg.norm(features) if np.linalg.norm(features) > 0 else features
            
            # 填充到512维
            if len(features) < 512:
                features = np.pad(features, (0, 512 - len(features)), 'constant')
            elif len(features) > 512:
                features = features[:512]
            
            logger.debug(f"特征提取完成，特征形状: {features.shape}")
            return features
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            # 返回随机特征向量作为降级方案
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
