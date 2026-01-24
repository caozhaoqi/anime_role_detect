#!/usr/bin/env python3
"""
简化版系统评估脚本，绕过YOLO模型加载问题
"""
import os
import sys
import numpy as np
from PIL import Image

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification

class SimpleEvaluator:
    """简化版评估器"""

    def __init__(self, index_path="role_index", threshold=0.7):
        """初始化评估器"""
        self.index_path = index_path
        self.threshold = threshold
        
        # 初始化各个模块
        self.extractor = FeatureExtraction()
        self.classifier = Classification(index_path, threshold)
        
        # 评估结果
        self.results = {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "unknown": 0
        }

    def simple_process(self, image_path):
        """简化的预处理流程"""
        try:
            # 直接加载并调整图像大小
            img = Image.open(image_path)
            img = img.resize((224, 224))
            img = img.convert('RGB')
            return img
        except Exception as e:
            print(f"预处理失败: {e}")
            return None

    def evaluate_single_character(self, test_dir):
        """评估单角色识别性能"""
        print("=== 评估单角色识别 ===")
        
        # 获取所有角色目录
        role_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        if not role_dirs:
            print(f"测试目录中没有角色子目录: {test_dir}")
            return
        
        # 首先构建索引
        print("\n[步骤 1] 构建向量索引...")
        all_features = []
        all_roles = []
        
        for role in role_dirs:
            role_dir = os.path.join(test_dir, role)
            image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"处理角色 '{role}'，共 {len(image_files)} 张图片用于构建索引")
            
            for img_file in image_files[:2]:  # 每个角色使用前2张图片构建索引
                img_path = os.path.join(role_dir, img_file)
                
                try:
                    # 简化预处理
                    normalized_img = self.simple_process(img_path)
                    if normalized_img is None:
                        continue
                    
                    # 提取特征
                    feature = self.extractor.extract_features(normalized_img)
                    
                    all_features.append(feature)
                    all_roles.append(role)
                    print(f"  成功处理图片: {img_file}")
                except Exception as e:
                    print(f"  处理图片 {img_file} 失败: {e}")
                    continue
        
        if not all_features:
            print("错误: 未能提取任何特征，终止验证")
            return
        
        # 构建索引
        features_np = np.array(all_features).astype(np.float32)
        self.classifier.build_index(features_np, all_roles)
        print(f"索引构建完成，包含 {len(all_roles)} 个向量")
        
        # 然后评估测试
        print("\n[步骤 2] 评估分类性能...")
        
        for role in role_dirs:
            role_dir = os.path.join(test_dir, role)
            image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"\n评估角色 '{role}'，共 {len(image_files)} 张图片")
            
            for img_file in image_files[2:]:  # 使用剩余图片进行测试
                img_path = os.path.join(role_dir, img_file)
                
                try:
                    # 简化预处理
                    normalized_img = self.simple_process(img_path)
                    if normalized_img is None:
                        continue
                    
                    # 提取特征
                    feature = self.extractor.extract_features(normalized_img)
                    
                    # 分类
                    predicted_role, similarity = self.classifier.classify(feature)
                    
                    # 记录结果
                    self.results["total"] += 1
                    
                    if predicted_role == "unknown" or similarity < self.threshold:
                        self.results["unknown"] += 1
                        print(f"  图片 {img_file}: 无法识别 (相似度: {similarity:.4f})")
                    elif predicted_role == role:
                        self.results["correct"] += 1
                        print(f"  图片 {img_file}: 正确识别为 {predicted_role} (相似度: {similarity:.4f})")
                    else:
                        self.results["incorrect"] += 1
                        print(f"  图片 {img_file}: 错误识别为 {predicted_role} (相似度: {similarity:.4f})")
                except Exception as e:
                    print(f"  处理图片 {img_file} 失败: {e}")
                    continue

    def generate_report(self):
        """生成评估报告"""
        print("\n=== 评估报告 ===")
        
        if self.results["total"] > 0:
            accuracy = self.results["correct"] / self.results["total"] * 100
            unknown_rate = self.results["unknown"] / self.results["total"] * 100
            
            print(f"总测试数: {self.results['total']}")
            print(f"正确识别数: {self.results['correct']}")
            print(f"错误识别数: {self.results['incorrect']}")
            print(f"无法识别数: {self.results['unknown']}")
            print(f"准确率: {accuracy:.2f}%")
            print(f"无法识别率: {unknown_rate:.2f}%")
            
            if accuracy > 50:
                print("结论: 系统功能验证通过 (准确率 > 50%)")
            else:
                print("结论: 系统功能验证未通过 (准确率 <= 50%)，请检查模型或数据质量")
        else:
            print("没有进行任何测试")

    def evaluate(self, single_character_dir):
        """完整评估"""
        # 评估单角色识别
        if os.path.exists(single_character_dir):
            self.evaluate_single_character(single_character_dir)
        
        # 生成报告
        self.generate_report()


def main():
    """主函数"""
    test_dir = "tests/test_images/single_character"
    
    evaluator = SimpleEvaluator()
    evaluator.evaluate(test_dir)


if __name__ == "__main__":
    main()
