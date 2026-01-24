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

# 导入日志记录模块
try:
    from src.core.log_fusion.log_recorder import record_classification_log
except Exception as e:
    print(f"导入日志记录模块失败: {e}")
    record_classification_log = None


class GeneralClassification:
    """通用分类器类"""
    
    # 全局缓存
    _instance_cache = None
    _index_cache = {}
    
    def __new__(cls, threshold=0.7):
        """单例模式，避免重复初始化"""
        if cls._instance_cache is None:
            cls._instance_cache = super(GeneralClassification, cls).__new__(cls)
        return cls._instance_cache
    
    def __init__(self, threshold=0.7):
        """初始化通用分类器"""
        if hasattr(self, 'is_initialized') and self.is_initialized:
            return
        
        self.threshold = threshold
        self.extractor = None
        self.classifier = None
        self.preprocessor = None
        self.is_initialized = False
        self.initialized_modules = set()
        
    def initialize(self):
        """初始化所有模块"""
        if self.is_initialized:
            return True
        
        try:
            # 延迟导入，避免启动时加载所有模块
            from src.core.preprocessing.preprocessing import Preprocessing
            from src.core.feature_extraction.feature_extraction import FeatureExtraction
            from src.core.classification.classification import Classification
            
            # 初始化各模块
            print("初始化预处理模块...")
            self.preprocessor = Preprocessing()
            
            print("初始化特征提取模块...")
            self.extractor = FeatureExtraction()
            
            print("初始化分类模块...")
            self.classifier = Classification(threshold=self.threshold)
            
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
            exclude_dirs = ['shuffled', 'classification_results', 'downloaded']
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
                
                # 为每个角色选择第一张图片作为代表
                image_path = os.path.join(role_dir, image_files[0])
                print(f"处理角色: {role}, 使用图片: {image_files[0]}")
                
                try:
                    # 预处理
                    normalized_img, _ = self.preprocessor.process(image_path)
                    
                    # 提取特征
                    feature = self.extractor.extract_features(normalized_img)
                    features.append(feature)
                    role_names.append(role)
                    
                    print(f"✓ 成功为角色 {role} 提取特征")
                except Exception as e:
                    print(f"✗ 处理角色 {role} 失败: {e}")
                    continue
            
            if not features:
                print("错误: 无法提取任何特征")
                return False
            
            # 构建索引
            features_np = np.array(features).astype(np.float32)
            self.classifier.build_index(features_np, role_names)
            
            print(f"✓ 索引构建成功，包含 {len(role_names)} 个角色")
            return True
        except Exception as e:
            print(f"构建索引失败: {e}")
            return False
    
    def classify_image(self, image_path):
        """分类单个图像"""
        if not self.initialize():
            return None, 0.0, None
        
        try:
            # 预处理
            normalized_img, boxes = self.preprocessor.process(image_path)
            
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
                    feature=feature,
                    boxes=boxes
                )
            
            return role, similarity, boxes
        except Exception as e:
            print(f"分类图像失败: {e}")
            return None, 0.0, None
    
    def classify_pil_image(self, pil_image):
        """分类PIL图像对象"""
        if not self.initialize():
            return None, 0.0
        
        try:
            # 转换PIL图像为numpy数组
            img_array = np.array(pil_image)
            
            # 保存为临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # 保存图像
            pil_image.save(temp_path)
            
            # 调用分类方法
            role, similarity, _ = self.classify_image(temp_path)
            
            # 删除临时文件
            os.unlink(temp_path)
            
            return role, similarity
        except Exception as e:
            print(f"分类PIL图像失败: {e}")
            return None, 0.0
    
    def batch_classify(self, image_paths):
        """批量分类图像"""
        if not self.initialize():
            return []
        
        results = []
        for image_path in image_paths:
            role, similarity, boxes = self.classify_image(image_path)
            results.append({
                'image_path': image_path,
                'role': role,
                'similarity': similarity,
                'boxes': boxes
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

def get_classifier():
    """获取全局分类器实例"""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = GeneralClassification()
    return _global_classifier

def classify_image(image_path):
    """快速分类图像的便捷函数"""
    classifier = get_classifier()
    return classifier.classify_image(image_path)

def classify_pil_image(pil_image):
    """快速分类PIL图像的便捷函数"""
    classifier = get_classifier()
    return classifier.classify_pil_image(pil_image)

def build_index_from_directory(data_dir):
    """快速从目录构建索引的便捷函数"""
    classifier = get_classifier()
    return classifier.build_index_from_directory(data_dir)

if __name__ == "__main__":
    """测试通用分类模块"""
    classifier = GeneralClassification()
    
    # 测试初始化
    print("测试模块初始化...")
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
        
        print(f"\n测试分类图像: {test_image}")
        role, similarity, boxes = classifier.classify_image(test_image)
        print(f"分类结果: 角色={role}, 相似度={similarity:.4f}, 检测框={boxes}")
    else:
        print(f"测试目录不存在: {test_dir}")
