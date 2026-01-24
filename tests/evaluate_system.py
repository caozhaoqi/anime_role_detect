#!/usr/bin/env python3
"""
系统评估脚本
"""
import os
import sys
import time
import numpy as np
import argparse

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification


class SystemEvaluator:
    """系统评估器"""

    def __init__(self, index_path="role_index", threshold=0.7):
        """初始化评估器"""
        self.index_path = index_path
        self.threshold = threshold
        
        # 初始化各个模块
        self.preprocessor = Preprocessing()
        self.extractor = FeatureExtraction()
        self.classifier = Classification(index_path, threshold)
        
        # 尝试初始化标签生成器，如果缺少依赖则跳过
        try:
            from src.core.tagging.tagging import Tagging
            self.tagger = Tagging()
        except ImportError:
            print("警告: 缺少标签生成依赖，跳过标签生成功能")
            self.tagger = None
        
        # 评估结果
        self.results = {
            "single_character": {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "unknown": 0,
                "times": []
            },
            "multiple_characters": {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "unknown": 0,
                "times": []
            }
        }

    def evaluate_single_character(self, test_dir):
        """评估单角色识别性能"""
        print("=== 评估单角色识别 ===")
        
        # 获取所有角色目录
        role_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        if not role_dirs:
            print(f"测试目录中没有角色子目录: {test_dir}")
            return
        
        for role in role_dirs:
            role_dir = os.path.join(test_dir, role)
            image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"评估角色 '{role}'，共 {len(image_files)} 张图片")
            
            for img_file in image_files:
                img_path = os.path.join(role_dir, img_file)
                
                try:
                    start_time = time.time()
                    
                    # 预处理图像
                    normalized_img, boxes = self.preprocessor.process(img_path)
                    
                    # 提取特征
                    feature = self.extractor.extract_features(normalized_img)
                    
                    # 分类
                    predicted_role, similarity = self.classifier.classify(feature)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    # 记录结果
                    self.results["single_character"]["total"] += 1
                    self.results["single_character"]["times"].append(processing_time)
                    
                    if predicted_role == "unknown" or similarity < self.threshold:
                        self.results["single_character"]["unknown"] += 1
                        print(f"  图片 {img_file}: 无法识别 (相似度: {similarity:.4f}, 时间: {processing_time:.4f}s)")
                    elif predicted_role == role:
                        self.results["single_character"]["correct"] += 1
                        print(f"  图片 {img_file}: 正确识别为 {predicted_role} (相似度: {similarity:.4f}, 时间: {processing_time:.4f}s)")
                    else:
                        self.results["single_character"]["incorrect"] += 1
                        print(f"  图片 {img_file}: 错误识别为 {predicted_role} (相似度: {similarity:.4f}, 时间: {processing_time:.4f}s)")
                    
                except Exception as e:
                    print(f"  处理图片 {img_file} 失败: {e}")
                    continue

    def evaluate_multiple_characters(self, test_dir):
        """评估多角色识别性能"""
        print("\n=== 评估多角色识别 ===")
        
        # 获取所有测试图片
        image_files = []
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print(f"测试目录中没有图片文件: {test_dir}")
            return
        
        print(f"共 {len(image_files)} 张测试图片")
        
        for img_path in image_files:
            try:
                start_time = time.time()
                
                # 处理多个角色
                characters = self.preprocessor.process_multiple_characters(img_path)
                
                if characters:
                    # 提取特征
                    characters_with_features = self.extractor.extract_features_from_multiple_characters(characters)
                    
                    # 分类
                    classified_characters = self.classifier.classify_multiple_characters(characters_with_features)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    # 记录结果
                    self.results["multiple_characters"]["total"] += 1
                    self.results["multiple_characters"]["times"].append(processing_time)
                    
                    # 打印结果
                    print(f"图片 {os.path.basename(img_path)}:")
                    print(f"  检测到 {len(classified_characters)} 个角色")
                    for i, char in enumerate(classified_characters):
                        role = char.get('role', 'unknown')
                        similarity = char.get('similarity', 0.0)
                        confidence = char.get('confidence', 0.0)
                        
                        if role == "unknown" or similarity < self.threshold:
                            self.results["multiple_characters"]["unknown"] += 1
                            print(f"  角色{i+1}: 无法识别 (检测置信度: {confidence:.4f}, 相似度: {similarity:.4f})")
                        else:
                            # 这里简化处理，实际应用中需要标注多角色的真实标签
                            self.results["multiple_characters"]["correct"] += 1
                            print(f"  角色{i+1}: {role} (检测置信度: {confidence:.4f}, 相似度: {similarity:.4f})")
                else:
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    self.results["multiple_characters"]["total"] += 1
                    self.results["multiple_characters"]["times"].append(processing_time)
                    self.results["multiple_characters"]["unknown"] += 1
                    print(f"图片 {os.path.basename(img_path)}: 未检测到角色")
                    
            except Exception as e:
                print(f"处理图片 {os.path.basename(img_path)} 失败: {e}")
                continue

    def generate_report(self, report_path="tests/test_results/evaluation_report.txt"):
        """生成评估报告"""
        print("\n=== 生成评估报告 ===")
        
        # 确保报告目录存在
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("二次元角色识别与分类系统评估报告\n")
            f.write("=" * 60 + "\n")
            
            # 单角色识别评估结果
            f.write("\n1. 单角色识别评估结果\n")
            f.write("-" * 40 + "\n")
            
            single_results = self.results["single_character"]
            if single_results["total"] > 0:
                accuracy = single_results["correct"] / single_results["total"] * 100
                unknown_rate = single_results["unknown"] / single_results["total"] * 100
                avg_time = np.mean(single_results["times"]) if single_results["times"] else 0
                
                f.write(f"总测试图片数: {single_results['total']}\n")
                f.write(f"正确识别数: {single_results['correct']}\n")
                f.write(f"错误识别数: {single_results['incorrect']}\n")
                f.write(f"无法识别数: {single_results['unknown']}\n")
                f.write(f"准确率: {accuracy:.2f}%\n")
                f.write(f"无法识别率: {unknown_rate:.2f}%\n")
                f.write(f"平均处理时间: {avg_time:.4f}s\n")
            else:
                f.write("无测试数据\n")
            
            # 多角色识别评估结果
            f.write("\n2. 多角色识别评估结果\n")
            f.write("-" * 40 + "\n")
            
            multi_results = self.results["multiple_characters"]
            if multi_results["total"] > 0:
                avg_characters_per_image = (multi_results["correct"] + multi_results["unknown"]) / multi_results["total"]
                avg_time = np.mean(multi_results["times"]) if multi_results["times"] else 0
                
                f.write(f"总测试图片数: {multi_results['total']}\n")
                f.write(f"识别角色数: {multi_results['correct']}\n")
                f.write(f"无法识别角色数: {multi_results['unknown']}\n")
                f.write(f"平均每张图片识别角色数: {avg_characters_per_image:.2f}\n")
                f.write(f"平均处理时间: {avg_time:.4f}s\n")
            else:
                f.write("无测试数据\n")
            
            # 系统配置
            f.write("\n3. 系统配置\n")
            f.write("-" * 40 + "\n")
            f.write(f"索引路径: {self.index_path}\n")
            f.write(f"相似度阈值: {self.threshold}\n")
        
        print(f"评估报告已生成: {report_path}")
        return report_path

    def evaluate(self, single_character_dir, multiple_character_dir):
        """完整评估"""
        # 评估单角色识别
        if os.path.exists(single_character_dir):
            self.evaluate_single_character(single_character_dir)
        
        # 评估多角色识别
        if os.path.exists(multiple_character_dir):
            self.evaluate_multiple_characters(multiple_character_dir)
        
        # 生成报告
        self.generate_report()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="系统评估脚本")
    parser.add_argument("--single_character_dir", default="tests/test_images/single_character", help="单角色测试数据目录")
    parser.add_argument("--multiple_character_dir", default="tests/test_images/multiple_characters", help="多角色测试数据目录")
    parser.add_argument("--index_path", default="role_index", help="向量索引路径")
    parser.add_argument("--threshold", type=float, default=0.7, help="相似度阈值")
    
    args = parser.parse_args()
    
    evaluator = SystemEvaluator(args.index_path, args.threshold)
    evaluator.evaluate(args.single_character_dir, args.multiple_character_dir)


if __name__ == "__main__":
    main()
