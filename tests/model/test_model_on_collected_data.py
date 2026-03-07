#!/usr/bin/env python3
"""
模型测试脚本
在采集的数据上测试训练好的模型性能
"""
import os
import sys
import json
import time
import logging
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_tester')


class ModelTester:
    def __init__(self, model_path, data_dir, batch_size=32, num_workers=4):
        self.model_path = model_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.transform = None
        
        self.results = {
            'predictions': [],
            'true_labels': [],
            'image_paths': [],
            'inference_times': []
        }
    
    def load_model(self):
        """加载训练好的模型"""
        logger.info(f"加载模型: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 提取类别信息
        if 'class_to_idx' in checkpoint:
            self.class_to_idx = checkpoint['class_to_idx']
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            logger.info(f"加载了 {len(self.class_to_idx)} 个类别")
        else:
            logger.warning("模型中未找到class_to_idx信息")
            self.class_to_idx = None
            self.idx_to_class = None
        
        # 重建模型
        num_classes = len(self.class_to_idx) if self.class_to_idx else 131
        
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        # 加载权重
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # 修复键名不匹配问题
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                name = k[9:]  # 移除 'backbone.'
            else:
                name = k
            new_state_dict[name] = v
        
        # 加载权重
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        model = model.to(self.device)
        
        self.model = model
        logger.info(f"模型加载完成，设备: {self.device}")
        
        # 定义数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
        ])
    
    def load_data(self, sample_ratio=1.0):
        """加载测试数据
        
        Args:
            sample_ratio: 数据采样比例 (0.0-1.0)，1.0表示使用全部数据
        """
        logger.info(f"加载数据: {self.data_dir} (采样比例: {sample_ratio*100:.0f}%)")
        
        data = []
        labels = []
        
        # 遍历数据目录
        for character_dir in sorted(os.listdir(self.data_dir)):
            character_path = os.path.join(self.data_dir, character_dir)
            
            if not os.path.isdir(character_path):
                continue
            
            # 提取角色名称
            character_name = character_dir
            
            # 查找对应的类别索引
            class_idx = None
            if self.class_to_idx and character_name in self.class_to_idx:
                class_idx = self.class_to_idx[character_name]
            else:
                logger.warning(f"角色 '{character_name}' 不在类别映射中，跳过")
                continue
            
            # 加载该角色的所有图片
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
                image_files.extend(Path(character_path).glob(ext))
            
            # 根据采样比例选择图片
            if sample_ratio < 1.0:
                import random
                random.shuffle(image_files)
                num_samples = int(len(image_files) * sample_ratio)
                image_files = image_files[:num_samples]
                logger.info(f"角色 '{character_name}': 原始{len(image_files)}张，采样{num_samples}张")
            
            for img_file in image_files:
                data.append({
                    'path': str(img_file),
                    'character': character_name,
                    'class_idx': class_idx
                })
                labels.append(class_idx)
        
        logger.info(f"加载了 {len(data)} 张图片，{len(set(labels))} 个类别")
        return data, labels
    
    def test(self, sample_ratio=1.0):
        """执行测试
        
        Args:
            sample_ratio: 数据采样比例 (0.0-1.0)，1.0表示使用全部数据
        """
        if self.model is None:
            raise ValueError("模型未加载")
        
        data, true_labels = self.load_data(sample_ratio=sample_ratio)
        
        if len(data) == 0:
            logger.error("没有可用的测试数据")
            return None
        
        logger.info("开始测试...")
        
        correct = 0
        total = 0
        
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for item in data:
            try:
                # 加载和预处理图像
                image = Image.open(item['path']).convert('RGB')
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                # 推理
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                end_time = time.time()
                
                inference_time = (end_time - start_time) * 1000  # 转换为毫秒
                
                # 获取预测结果
                _, predicted_idx = torch.max(outputs, 1)
                predicted_idx = predicted_idx.item()
                true_idx = item['class_idx']
                
                # 记录结果
                self.results['predictions'].append(predicted_idx)
                self.results['true_labels'].append(true_idx)
                self.results['image_paths'].append(item['path'])
                self.results['inference_times'].append(inference_time)
                
                # 统计准确率
                if predicted_idx == true_idx:
                    correct += 1
                    class_correct[true_idx] += 1
                class_total[true_idx] += 1
                
                total += 1
                
                if total % 100 == 0:
                    logger.info(f"已处理 {total}/{len(data)} 张图片")
                    
            except Exception as e:
                logger.error(f"处理图片失败 {item['path']}: {e}")
        
        # 计算总体准确率
        accuracy = correct / total if total > 0 else 0
        
        logger.info(f"测试完成！")
        logger.info(f"总体准确率: {accuracy:.4f} ({correct}/{total})")
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for class_idx in class_total:
            acc = class_correct[class_idx] / class_total[class_idx]
            class_name = self.idx_to_class.get(class_idx, f"Class_{class_idx}")
            class_accuracies[class_name] = {
                'accuracy': acc,
                'correct': class_correct[class_idx],
                'total': class_total[class_idx]
            }
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'class_accuracies': class_accuracies,
            'avg_inference_time': np.mean(self.results['inference_times']),
            'total_inference_time': sum(self.results['inference_times'])
        }
    
    def generate_report(self, test_results, output_path='test_report.txt'):
        """生成测试报告"""
        if test_results is None:
            logger.error("测试结果为空，无法生成报告")
            return
        
        logger.info(f"生成测试报告: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("模型测试报告\n")
            f.write("="*80 + "\n\n")
            
            # 基本信息
            f.write("1. 基本信息\n")
            f.write("-"*80 + "\n")
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"数据目录: {self.data_dir}\n")
            f.write(f"测试设备: {self.device}\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 总体性能
            f.write("2. 总体性能\n")
            f.write("-"*80 + "\n")
            f.write(f"总样本数: {test_results['total']}\n")
            f.write(f"正确预测: {test_results['correct']}\n")
            f.write(f"准确率: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)\n")
            f.write(f"平均推理时间: {test_results['avg_inference_time']:.2f}ms\n")
            f.write(f"总推理时间: {test_results['total_inference_time']:.2f}ms\n\n")
            
            # 分类报告
            f.write("3. 分类报告\n")
            f.write("-"*80 + "\n")
            
            true_labels = self.results['true_labels']
            predictions = self.results['predictions']
            
            # 获取所有实际的类别索引
            actual_classes = sorted(set(true_labels))
            
            # 为每个实际类别生成名称
            target_names = []
            for idx in actual_classes:
                class_name = self.idx_to_class.get(idx, f"Class_{idx}")
                target_names.append(class_name)
            
            # 使用labels参数指定实际的类别
            report = classification_report(true_labels, predictions, 
                                      labels=actual_classes,
                                      target_names=target_names, 
                                      digits=4,
                                      zero_division=0)
            f.write(report)
            f.write("\n")
            
            # 每个类别的详细结果
            f.write("4. 各类别详细结果\n")
            f.write("-"*80 + "\n")
            f.write(f"{'角色名称':<20} {'准确率':<10} {'正确数':<10} {'总数':<10}\n")
            f.write("-"*80 + "\n")
            
            # 按准确率排序
            sorted_classes = sorted(test_results['class_accuracies'].items(), 
                               key=lambda x: x[1]['accuracy'], 
                               reverse=True)
            
            for class_name, metrics in sorted_classes:
                f.write(f"{class_name:<20} {metrics['accuracy']:<10.4f} "
                       f"{metrics['correct']:<10} {metrics['total']:<10}\n")
            
            # 性能最好的类别
            f.write("\n5. 性能最好的类别 (Top 10)\n")
            f.write("-"*80 + "\n")
            for i, (class_name, metrics) in enumerate(sorted_classes[:10], 1):
                f.write(f"{i:2d}. {class_name:<20} 准确率: {metrics['accuracy']:.4f} "
                       f"({metrics['correct']}/{metrics['total']})\n")
            
            # 性能最差的类别
            f.write("\n6. 性能最差的类别 (Top 10)\n")
            f.write("-"*80 + "\n")
            for i, (class_name, metrics) in enumerate(sorted_classes[-10:], 1):
                f.write(f"{i:2d}. {class_name:<20} 准确率: {metrics['accuracy']:.4f} "
                       f"({metrics['correct']}/{metrics['total']})\n")
            
            # 推理时间统计
            f.write("\n7. 推理时间统计\n")
            f.write("-"*80 + "\n")
            inference_times = self.results['inference_times']
            f.write(f"平均推理时间: {np.mean(inference_times):.2f}ms\n")
            f.write(f"最快推理时间: {np.min(inference_times):.2f}ms\n")
            f.write(f"最慢推理时间: {np.max(inference_times):.2f}ms\n")
            f.write(f"推理时间标准差: {np.std(inference_times):.2f}ms\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("报告生成完成\n")
            f.write("="*80 + "\n")
        
        logger.info(f"测试报告已保存: {output_path}")
    
    def save_detailed_results(self, output_path='detailed_results.json'):
        """保存详细的测试结果到JSON文件"""
        detailed_results = {
            'model_path': self.model_path,
            'data_dir': self.data_dir,
            'test_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': []
        }
        
        for i in range(len(self.results['predictions'])):
            true_idx = self.results['true_labels'][i]
            pred_idx = self.results['predictions'][i]
            
            true_name = self.idx_to_class.get(true_idx, f"Class_{true_idx}")
            pred_name = self.idx_to_class.get(pred_idx, f"Class_{pred_idx}")
            
            detailed_results['results'].append({
                'image_path': self.results['image_paths'][i],
                'true_label': true_name,
                'predicted_label': pred_name,
                'correct': true_idx == pred_idx,
                'inference_time': self.results['inference_times'][i]
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细结果已保存: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='模型测试工具')
    parser.add_argument('--model_path', type=str, 
                       default='models/character_classifier_best_improved.pth',
                       help='模型路径')
    parser.add_argument('--data_dir', type=str, 
                       default='data/all_characters',
                       help='数据目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='工作进程数')
    parser.add_argument('--output_report', type=str, 
                       default='test_report.txt',
                       help='报告输出路径')
    parser.add_argument('--output_json', type=str, 
                       default='detailed_results.json',
                       help='详细结果JSON输出路径')
    parser.add_argument('--sample_ratio', type=float, default=0.1,
                       help='数据采样比例 (0.0-1.0), 默认1.0表示使用全部数据')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ModelTester(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 加载模型
    tester.load_model()
    
    # 执行测试
    test_results = tester.test(sample_ratio=args.sample_ratio)
    
    if test_results:
        # 生成报告
        tester.generate_report(test_results, args.output_report)
        
        # 保存详细结果
        tester.save_detailed_results(args.output_json)


if __name__ == "__main__":
    main()
