import time
import os
import numpy as np
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification

class PerformanceTest:
    def __init__(self, test_dir="test_images", index_path="role_index"):
        """初始化性能测试"""
        self.test_dir = test_dir
        self.index_path = index_path
        
        # 初始化各个模块
        self.preprocessor = Preprocessing()
        self.extractor = FeatureExtraction()
        self.classifier = Classification(index_path)
    
    def get_test_images(self):
        """获取测试图片"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = []
        
        if not os.path.exists(self.test_dir):
            print(f"测试目录不存在: {self.test_dir}")
            return []
        
        for root, dirs, files in os.walk(self.test_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        return image_files
    
    def test_preprocessing(self, image_path):
        """测试预处理性能"""
        start_time = time.time()
        normalized_img, boxes = self.preprocessor.process(image_path)
        end_time = time.time()
        
        return end_time - start_time, normalized_img, boxes
    
    def test_feature_extraction(self, normalized_img):
        """测试特征提取性能"""
        start_time = time.time()
        feature = self.extractor.extract_features(normalized_img)
        end_time = time.time()
        
        return end_time - start_time, feature
    
    def test_classification(self, feature):
        """测试分类性能"""
        start_time = time.time()
        role, similarity = self.classifier.classify(feature)
        end_time = time.time()
        
        return end_time - start_time, role, similarity
    
    def test_end_to_end(self, image_path):
        """测试端到端性能"""
        start_time = time.time()
        
        # 预处理
        normalized_img, boxes = self.preprocessor.process(image_path)
        
        # 特征提取
        feature = self.extractor.extract_features(normalized_img)
        
        # 分类
        role, similarity = self.classifier.classify(feature)
        
        end_time = time.time()
        
        return end_time - start_time, role, similarity
    
    def run_tests(self):
        """运行所有测试"""
        # 获取测试图片
        test_images = self.get_test_images()
        if not test_images:
            print("没有找到测试图片")
            return
        
        print(f"找到 {len(test_images)} 张测试图片")
        
        # 存储测试结果
        preprocessing_times = []
        feature_extraction_times = []
        classification_times = []
        end_to_end_times = []
        
        # 运行测试
        for i, image_path in enumerate(test_images):
            print(f"测试第 {i+1}/{len(test_images)} 张图片: {os.path.basename(image_path)}")
            
            try:
                # 测试预处理
                pre_time, normalized_img, boxes = self.test_preprocessing(image_path)
                preprocessing_times.append(pre_time)
                print(f"  预处理时间: {pre_time:.4f}s")
                
                # 测试特征提取
                feat_time, feature = self.test_feature_extraction(normalized_img)
                feature_extraction_times.append(feat_time)
                print(f"  特征提取时间: {feat_time:.4f}s")
                
                # 测试分类
                class_time, role, similarity = self.test_classification(feature)
                classification_times.append(class_time)
                print(f"  分类时间: {class_time:.4f}s")
                print(f"  识别结果: {role} (相似度: {similarity:.4f})")
                
                # 测试端到端
                end_to_end_time, _, _ = self.test_end_to_end(image_path)
                end_to_end_times.append(end_to_end_time)
                print(f"  端到端时间: {end_to_end_time:.4f}s")
                
            except Exception as e:
                print(f"  测试失败: {e}")
                continue
        
        # 计算平均时间
        if preprocessing_times:
            avg_pre_time = np.mean(preprocessing_times)
            print(f"\n平均预处理时间: {avg_pre_time:.4f}s")
        
        if feature_extraction_times:
            avg_feat_time = np.mean(feature_extraction_times)
            print(f"平均特征提取时间: {avg_feat_time:.4f}s")
        
        if classification_times:
            avg_class_time = np.mean(classification_times)
            print(f"平均分类时间: {avg_class_time:.4f}s")
        
        if end_to_end_times:
            avg_end_to_end_time = np.mean(end_to_end_times)
            print(f"平均端到端时间: {avg_end_to_end_time:.4f}s")
        
        # 计算FPS
        if end_to_end_times:
            avg_end_to_end_time = np.mean(end_to_end_times)
            fps = 1.0 / avg_end_to_end_time
            print(f"平均FPS: {fps:.2f}")

if __name__ == "__main__":
    # 运行性能测试
    test = PerformanceTest()
    test.run_tests()
