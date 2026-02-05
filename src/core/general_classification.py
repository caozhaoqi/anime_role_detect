#!/usr/bin/env python3
"""
通用分类模块
封装核心分类逻辑，提供简洁的API接口
支持网页应用和其他脚本调用
"""
import os
import sys
import numpy as np
from PIL import Image

# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 计算项目根目录的绝对路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入日志记录模块
try:
    from src.core.log_fusion.log_recorder import record_classification_log
except Exception as e:
    # print(f"导入日志记录模块失败: {e}") # 暂时屏蔽，避免干扰
    record_classification_log = None


class GeneralClassification:
    """通用分类器类"""
    
    # 全局缓存
    _instance_cache = None
    
    def __new__(cls, threshold=0.7, index_path=None):
        """单例模式，避免重复初始化"""
        if cls._instance_cache is None:
            cls._instance_cache = super(GeneralClassification, cls).__new__(cls)
        return cls._instance_cache
    
    def __init__(self, threshold=0.7, index_path=None):
        """初始化通用分类器"""
        if hasattr(self, 'is_initialized') and self.is_initialized:
            return
        
        self.threshold = threshold
        self.index_path = index_path
        self.extractor = None
        self.classifier = None
        self.preprocessor = None
        self.model_inference = None # 新增：模型推理器
        self.is_initialized = False
        
    def initialize(self):
        """初始化所有模块"""
        if self.is_initialized:
            return True
        
        try:
            # 延迟导入，避免启动时加载所有模块
            from src.core.preprocessing.preprocessing import Preprocessing
            from src.core.feature_extraction.feature_extraction import FeatureExtraction
            from src.core.classification.classification import Classification
            from src.core.classification.efficientnet_inference import EfficientNetInference
            
            # 初始化各模块
            print("初始化预处理模块...")
            self.preprocessor = Preprocessing()
            
            print("初始化特征提取模块...")
            self.extractor = FeatureExtraction()
            
            print("初始化分类模块...")
            self.classifier = Classification(threshold=self.threshold)
            
            # 尝试初始化EfficientNet推理器
            try:
                print("初始化EfficientNet推理模型...")
                self.model_inference = EfficientNetInference()
            except Exception as e:
                print(f"EfficientNet模型初始化失败 (非致命): {e}")
            
            # 如果指定了索引路径，尝试加载
            if self.index_path:
                if os.path.exists(f"{self.index_path}.faiss") and os.path.exists(f"{self.index_path}_mapping.json"):
                    print(f"加载索引文件: {self.index_path}")
                    self.classifier.load_index(self.index_path)
                else:
                    print(f"警告: 索引文件不存在: {self.index_path}")
            
            self.is_initialized = True
            print("所有模块初始化成功!")
            return True
        except Exception as e:
            print(f"模块初始化失败: {e}")
            self.is_initialized = False
            return False
    
    def build_index_from_directory(self, data_dir):
        """从目录构建特征索引"""
        if not self.initialize():
            return False
        
        try:
            print(f"从目录构建索引: {data_dir}")
            
            # 收集所有角色目录，过滤掉非角色目录
            exclude_dirs = ['shuffled', 'classification_results', 'downloaded', '.DS_Store']
            role_dirs = [d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d)) and d not in exclude_dirs]
            
            if not role_dirs:
                print("错误: 目录中没有角色子目录")
                return False
            
            print(f"找到 {len(role_dirs)} 个角色目录")
            
            # 为每个角色提取特征
            features = []
            role_names = []
            
            for role in role_dirs:
                role_dir = os.path.join(data_dir, role)
                
                # 获取角色目录中的图片
                image_files = [f for f in os.listdir(role_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                if not image_files:
                    print(f"警告: 角色 {role} 目录中没有图片")
                    continue
                
                print(f"处理角色: {role}, 共 {len(image_files)} 张图片")
                
                # 遍历该角色下的所有图片，提取特征
                for img_file in image_files:
                    image_path = os.path.join(role_dir, img_file)
                    try:
                        # 预处理
                        normalized_img, _ = self.preprocessor.process(image_path)
                        
                        # 提取特征
                        feature = self.extractor.extract_features(normalized_img)
                        features.append(feature)
                        role_names.append(role)
                        
                    except Exception as e:
                        print(f"  ✗ 处理图片 {img_file} 失败: {e}")
                        continue
                
                print(f"  ✓ 完成角色 {role} 的特征提取")
            
            if not features:
                print("错误: 无法提取任何特征")
                return False
            
            # 构建索引
            features_np = np.array(features).astype(np.float32)
            self.classifier.build_index(features_np, role_names)
            
            print(f"✓ 索引构建成功，包含 {len(features)} 个向量，覆盖 {len(set(role_names))} 个角色")
            return True
        except Exception as e:
            print(f"构建索引失败: {e}")
            return False
    
    def classify_image(self, image_path, use_model=False):
        """分类单个图像
        :param image_path: 图片路径
        :param use_model: 是否使用EfficientNet模型进行推理
        """
        if not self.initialize():
            return None, 0.0, None
            
        try:
            # 预处理 (YOLO检测)
            # 无论使用哪种方法，先用YOLO裁剪出人物主体总是好的
            normalized_img, boxes = self.preprocessor.process(image_path)
            
            # 如果指定使用模型且模型已加载
            if use_model and self.model_inference:
                # 注意：EfficientNetInference 内部会再次读取图片并进行自己的预处理
                # 这里我们传入原始路径，或者我们可以优化让它接受 PIL Image
                # 目前为了简单，直接传路径
                role, similarity, _ = self.model_inference.predict(image_path)
                feature = None # 模型推理不产生CLIP特征向量
            else:
                # 使用默认的 CLIP + FAISS 索引
                if self.classifier.index is None:
                    raise ValueError("索引尚未构建或加载。请先运行 scripts/build_faiss_index.py 构建索引。")
                
                # 提取特征
                feature = self.extractor.extract_features(normalized_img)
                
                # 分类
                role, similarity = self.classifier.classify(feature)
            
            # 记录分类日志
            if record_classification_log is not None:
                record_classification_log(
                    image_path=image_path,
                    role=role,
                    similarity=similarity,
                    feature=feature if feature is not None else [],
                    boxes=boxes
                )
            
            return role, similarity, boxes
        except Exception as e:
            print(f"分类图像失败: {e}")
            raise e # 抛出异常以便上层捕获
    
    def classify_pil_image(self, pil_image, use_model=False):
        """分类PIL图像对象"""
        if not self.initialize():
            return None, 0.0
        
        try:
            # 保存为临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # 保存图像
            pil_image.save(temp_path)
            
            # 调用分类方法
            role, similarity, _ = self.classify_image(temp_path, use_model=use_model)
            
            # 删除临时文件
            os.unlink(temp_path)
            
            return role, similarity
        except Exception as e:
            print(f"分类PIL图像失败: {e}")
            return None, 0.0
    
    def batch_classify(self, image_paths, use_model=False):
        """批量分类图像"""
        if not self.initialize():
            return []
        
        results = []
        for image_path in image_paths:
            try:
                role, similarity, boxes = self.classify_image(image_path, use_model=use_model)
                results.append({
                    'image_path': image_path,
                    'role': role,
                    'similarity': similarity,
                    'boxes': boxes
                })
            except Exception as e:
                print(f"处理图片 {image_path} 失败: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        return results
    
    def get_available_roles(self):
        """获取可用角色列表"""
        if not self.initialize():
            return []
        
        if hasattr(self.classifier, 'role_mapping'):
            return self.classifier.role_mapping
        return []
    
    def save_index(self, index_path):
        """保存索引"""
        if not self.initialize():
            return False
        
        try:
            self.classifier.save_index(index_path)
            print(f"索引已保存到: {index_path}")
            return True
        except Exception as e:
            print(f"保存索引失败: {e}")
            return False
    
    def load_index(self, index_path):
        """加载索引"""
        if not self.initialize():
            return False
        
        try:
            self.classifier.load_index(index_path)
            print(f"索引已从: {index_path} 加载")
            print(f"包含角色: {self.classifier.role_mapping}")
            return True
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False

# 全局分类器实例
_global_classifier = None

def get_classifier(index_path="role_index"):
    """获取全局分类器实例"""
    global _global_classifier
    if _global_classifier is None:
        # 构造索引文件的绝对路径
        abs_index_path = os.path.join(PROJECT_ROOT, index_path)
        _global_classifier = GeneralClassification(index_path=abs_index_path)
    return _global_classifier

def classify_image(image_path, use_model=False):
    """快速分类图像的便捷函数"""
    classifier = get_classifier()
    return classifier.classify_image(image_path, use_model=use_model)

def classify_pil_image(pil_image, use_model=False):
    """快速分类PIL图像的便捷函数"""
    classifier = get_classifier()
    return classifier.classify_pil_image(pil_image, use_model=use_model)

def build_index_from_directory(data_dir):
    """快速从目录构建索引的便捷函数"""
    classifier = get_classifier()
    return classifier.build_index_from_directory(data_dir)

if __name__ == "__main__":
    """测试通用分类模块"""
    # 测试初始化
    print("测试模块初始化...")
    classifier = GeneralClassification()
    classifier.initialize()
    
    # 测试从目录构建索引
    test_dir = "data/blue_archive_optimized"
    if os.path.exists(test_dir):
        print(f"\n测试从目录构建索引: {test_dir}")
        classifier.build_index_from_directory(test_dir)
        
        # 测试分类
        test_image = os.path.join(test_dir, "蔚蓝档案_星野", 
                                 [f for f in os.listdir(os.path.join(test_dir, "蔚蓝档案_星野")) 
                                  if f.endswith('.jpg')][0])
        
        print(f"\n测试分类图像 (索引模式): {test_image}")
        role, similarity, boxes = classifier.classify_image(test_image, use_model=False)
        print(f"分类结果: 角色={role}, 相似度={similarity:.4f}, 检测框={boxes}")
        
        print(f"\n测试分类图像 (模型模式): {test_image}")
        role, similarity, boxes = classifier.classify_image(test_image, use_model=True)
        print(f"分类结果: 角色={role}, 相似度={similarity:.4f}, 检测框={boxes}")
    else:
        print(f"测试目录不存在: {test_dir}")
