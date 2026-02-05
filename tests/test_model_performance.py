#!/usr/bin/env python3
"""
测试最佳模型的性能，生成详细的测试报告
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.classification.efficientnet_inference import EfficientNetInference

class ModelTester:
    def __init__(self, model_path, data_dir, output_dir='outputs/analysis', batch_size=32):
        """初始化模型测试器"""
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载模型
        self.model = self.load_model()
        
        # 加载数据集信息
        self.classes = self.model.classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        print(f"测试数据集包含 {len(self.classes)} 个类别")
        
        # 收集所有测试图像路径
        self.test_images = self.collect_test_images()
        print(f"找到 {len(self.test_images)} 张测试图像")
    
    def load_model(self):
        """加载模型"""
        print(f"加载模型: {self.model_path}")
        # 启用优化
        model = EfficientNetInference(model_path=self.model_path, data_dir=self.data_dir, enable_optimizations=True)
        print("模型加载成功")
        return model
    
    def collect_test_images(self):
        """收集所有测试图像路径"""
        test_images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_dir, img_file)
                        test_images.append((img_path, class_name))
        return test_images
    
    def test_model(self):
        """测试模型性能"""
        print("开始测试模型...")
        
        all_preds = []
        all_labels = []
        all_scores = []
        
        # 提取图像路径和真实标签
        image_paths = [img_path for img_path, _ in self.test_images]
        true_classes = [true_class for _, true_class in self.test_images]
        
        # 使用批量预测提高速度
        start_time = time.time()
        batch_results = self.model.predict_batch(image_paths, batch_size=self.batch_size)
        total_time = time.time() - start_time
        
        print(f"批量预测完成，耗时: {total_time:.2f}秒")
        print(f"平均每张图像预测时间: {total_time/len(image_paths):.4f}秒")
        
        # 处理预测结果
        for i, result in enumerate(batch_results):
            if 'error' in result:
                # 处理错误情况
                all_preds.append(-1)
                all_labels.append(self.class_to_idx[true_classes[i]])
                all_scores.append(0.0)
            else:
                # 处理正常预测
                pred_class = result['best_role']
                score = result['best_score']
                true_idx = self.class_to_idx[true_classes[i]]
                pred_idx = self.class_to_idx.get(pred_class, -1) if pred_class in self.class_to_idx else -1
                
                all_preds.append(pred_idx)
                all_labels.append(true_idx)
                all_scores.append(score)
        
        # 过滤掉无效预测
        valid_indices = [i for i, pred in enumerate(all_preds) if pred != -1]
        filtered_labels = [all_labels[i] for i in valid_indices]
        filtered_preds = [all_preds[i] for i in valid_indices]
        
        # 计算指标
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        report = classification_report(filtered_labels, filtered_preds, target_names=self.classes, output_dict=True)
        cm = confusion_matrix(filtered_labels, filtered_preds, labels=list(range(len(self.classes))))
        
        print(f"测试完成！总准确率: {accuracy:.4f}")
        print(f"有效预测: {len(valid_indices)}/{len(all_preds)}")
        
        # 获取性能统计信息
        performance_stats = self.model.get_performance_stats()
        print("\n性能统计信息:")
        print(json.dumps(performance_stats, indent=2))
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'all_preds': all_preds,
            'all_labels': all_labels,
            'all_scores': all_scores,
            'valid_indices': valid_indices,
            'performance_stats': performance_stats
        }
    
    def generate_report(self, test_results):
        """生成测试报告"""
        print("生成测试报告...")
        
        # 保存详细的分类报告
        report_path = os.path.join(self.output_dir, 'classification_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(test_results['report'], f, ensure_ascii=False, indent=2)
        print(f"分类报告已保存: {report_path}")
        
        # 生成混淆矩阵热力图
        self.plot_confusion_matrix(test_results['confusion_matrix'])
        
        # 生成整体性能摘要
        summary = {
            'model_path': self.model_path,
            'accuracy': test_results['accuracy'],
            'num_classes': len(self.classes),
            'num_samples': len(self.test_images),
            'valid_predictions': len(test_results['valid_indices']),
            'performance_stats': test_results['performance_stats'],
            'class_performance': {}
        }
        
        # 添加每个类别的性能
        for cls in self.classes:
            if cls in test_results['report']:
                summary['class_performance'][cls] = {
                    'precision': test_results['report'][cls]['precision'],
                    'recall': test_results['report'][cls]['recall'],
                    'f1-score': test_results['report'][cls]['f1-score'],
                    'support': test_results['report'][cls]['support']
                }
        
        # 保存摘要
        summary_path = os.path.join(self.output_dir, 'performance_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"性能摘要已保存: {summary_path}")
        
        # 打印关键指标
        print("\n=== 模型性能摘要 ===")
        print(f"总准确率: {summary['accuracy']:.4f}")
        print(f"测试样本数: {summary['num_samples']}")
        print(f"有效预测: {summary['valid_predictions']}")
        print(f"类别数: {summary['num_classes']}")
        
        # 计算平均指标
        avg_precision = np.mean([v['precision'] for v in summary['class_performance'].values()])
        avg_recall = np.mean([v['recall'] for v in summary['class_performance'].values()])
        avg_f1 = np.mean([v['f1-score'] for v in summary['class_performance'].values()])
        
        print(f"平均精确率: {avg_precision:.4f}")
        print(f"平均召回率: {avg_recall:.4f}")
        print(f"平均F1分数: {avg_f1:.4f}")
        
        # 打印性能统计
        if 'performance_stats' in summary:
            print("\n=== 推理性能统计 ===")
            stats = summary['performance_stats']
            print(f"总推理次数: {stats['total_inferences']}")
            print(f"平均推理时间: {stats['average_time']:.4f}秒")
            print(f"最快推理时间: {stats['min_time']:.4f}秒")
            print(f"最慢推理时间: {stats['max_time']:.4f}秒")
        
        return summary
    
    def plot_confusion_matrix(self, cm):
        """绘制混淆矩阵"""
        plt.figure(figsize=(20, 20))
        sns.set(font_scale=0.8)
        
        # 计算混淆矩阵的归一化版本
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 绘制热力图
        sns.heatmap(
            cm_normalized, 
            annot=False, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes
        )
        
        plt.title('混淆矩阵 (归一化)', fontsize=16)
        plt.xlabel('预测标签', fontsize=14)
        plt.ylabel('真实标签', fontsize=14)
        plt.tight_layout()
        
        # 保存混淆矩阵
        cm_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"混淆矩阵已保存: {cm_path}")
    
    def test_single_image(self, image_path):
        """测试单个图像"""
        print(f"测试单个图像: {image_path}")
        
        # 模型预测
        pred_class, score, results = self.model.predict(image_path, top_k=5)
        
        print("预测结果:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['role']}: {result['similarity']:.4f}")
        
        return results

    def test_parallel(self, num_workers=4):
        """使用并行预测测试模型"""
        print(f"开始并行测试模型 (使用 {num_workers} 个线程)...")
        
        # 提取图像路径
        image_paths = [img_path for img_path, _ in self.test_images[:1000]]  # 测试前1000张图像
        
        start_time = time.time()
        parallel_results = self.model.predict_parallel(image_paths, num_workers=num_workers)
        total_time = time.time() - start_time
        
        print(f"并行预测完成，耗时: {total_time:.2f}秒")
        print(f"平均每张图像预测时间: {total_time/len(image_paths):.4f}秒")
        
        return parallel_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试模型性能')
    parser.add_argument('--model_path', type=str, 
                        default='models/character_classifier_best_improved.pth',
                        help='模型路径')
    parser.add_argument('--data_dir', type=str, 
                        default='data/split_dataset/val',
                        help='测试数据集目录')
    parser.add_argument('--output_dir', type=str, 
                        default='outputs/analysis',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, 
                        default=32,
                        help='批量预测大小')
    parser.add_argument('--test_parallel', action='store_true',
                        help='是否测试并行预测性能')
    
    args = parser.parse_args()
    
    # 初始化测试器
    tester = ModelTester(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    # 测试并行预测性能
    if args.test_parallel:
        tester.test_parallel()
    
    # 测试模型
    test_results = tester.test_model()
    
    # 生成报告
    summary = tester.generate_report(test_results)
    
    # 测试几个示例图像
    print("\n测试示例图像...")
    
    # 选择前5个测试图像
    test_images = tester.test_images[:5]
    for img_path, true_class in test_images:
        print(f"\n测试图像: {img_path}")
        print(f"真实类别: {true_class}")
        tester.test_single_image(img_path)

if __name__ == '__main__':
    main()
