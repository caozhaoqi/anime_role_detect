#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合评估脚本

评估所有改进的综合性能，包括模型性能、数据集扩充效果、在线学习能力等。
"""

import os
import json
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('comprehensive_evaluation')

class CharacterDataset:
    """角色数据集"""
    def __init__(self, data_dir, transform=None, max_images=1000):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # 遍历目录结构
        for idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                class_images = []
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_dir, img_name)
                        class_images.append(img_path)
                
                # 限制每个类别的图像数量
                class_images = class_images[:10]  # 每个类别最多10张图像
                for img_path in class_images:
                    self.image_paths.append(img_path)
                    self.labels.append(idx)
                    
                    # 达到最大图像数量时停止
                    if len(self.image_paths) >= max_images:
                        break
            
            # 达到最大图像数量时停止
            if len(self.image_paths) >= max_images:
                break
        
        logger.info(f"数据集初始化完成，包含 {len(self.class_to_idx)} 个类别，{len(self.image_paths)} 张图像")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"加载图像 {img_path} 失败: {e}")
            return torch.zeros(3, 224, 224), label

class CharacterClassifier(nn.Module):
    """角色分类器"""
    def __init__(self, num_classes=1000):
        super(CharacterClassifier, self).__init__()
        from torchvision import models
        self.backbone = models.efficientnet_b0(pretrained=False)
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features,
            num_classes
        )
    
    def forward(self, x):
        return self.backbone(x)

class ModelEvaluator:
    """模型评估器"""
    def __init__(self, device='mps'):
        self.device = device
    
    def load_model(self, model_path, num_classes=1000):
        """加载模型"""
        model = CharacterClassifier(num_classes=num_classes)
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"成功加载模型: {model_path}")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                return None
        model = model.to(self.device)
        model.eval()
        return model
    
    def evaluate_model(self, model, dataloader):
        """评估模型"""
        if not model:
            return {}
        
        correct = 0
        total = 0
        total_time = 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="评估模型"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 记录推理时间
                start_time = time.time()
                outputs = model(images)
                end_time = time.time()
                total_time += end_time - start_time
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = (correct / total) * 100
        avg_inference_time = (total_time / total) * 1000  # 毫秒
        fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'accuracy': accuracy,
            'avg_inference_time_ms': avg_inference_time,
            'fps': fps,
            'total_images': total
        }

class DatasetEvaluator:
    """数据集评估器"""
    def evaluate_dataset(self, data_dir):
        """评估数据集"""
        # 统计数据集信息
        class_counts = {}
        total_images = 0
        
        for class_name in sorted(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                image_count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                class_counts[class_name] = image_count
                total_images += image_count
        
        num_classes = len(class_counts)
        avg_images_per_class = total_images / num_classes if num_classes > 0 else 0
        
        return {
            'num_classes': num_classes,
            'total_images': total_images,
            'avg_images_per_class': avg_images_per_class,
            'class_counts': class_counts
        }

class ComprehensiveEvaluator:
    """综合评估器"""
    def __init__(self, device='cpu'):
        self.device = device
        self.model_evaluator = ModelEvaluator(device=device)
        self.dataset_evaluator = DatasetEvaluator()
    
    def evaluate_models(self, model_paths, test_data_dir):
        """评估多个模型"""
        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载测试数据集
        dataset = CharacterDataset(test_data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # 评估每个模型
        model_results = {}
        for model_name, model_path in model_paths.items():
            logger.info(f"评估模型: {model_name}")
            model = self.model_evaluator.load_model(model_path, num_classes=len(dataset.class_to_idx))
            if model:
                results = self.model_evaluator.evaluate_model(model, dataloader)
                model_results[model_name] = results
                logger.info(f"模型 {model_name} 评估结果: {results}")
            else:
                model_results[model_name] = {}
                logger.warning(f"跳过模型评估: {model_name}")
        
        return model_results
    
    def evaluate_dataset_expansion(self, original_data_dir, expanded_data_dir):
        """评估数据集扩充效果"""
        logger.info("评估数据集扩充效果...")
        
        # 评估原始数据集
        original_stats = self.dataset_evaluator.evaluate_dataset(original_data_dir)
        
        # 评估扩充后数据集
        expanded_stats = self.dataset_evaluator.evaluate_dataset(expanded_data_dir)
        
        # 计算扩充效果
        expansion_ratio = expanded_stats['total_images'] / original_stats['total_images'] if original_stats['total_images'] > 0 else 0
        class_increase = expanded_stats['num_classes'] - original_stats['num_classes']
        
        return {
            'original_dataset': original_stats,
            'expanded_dataset': expanded_stats,
            'expansion_ratio': expansion_ratio,
            'class_increase': class_increase
        }
    
    def evaluate_online_learning(self, base_model_path, new_data_dir, test_data_dir):
        """评估在线学习能力"""
        logger.info("评估在线学习能力...")
        
        # 这里简化处理，实际应该使用OnlineLearningSystem
        # 加载基础模型
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载测试数据集
        dataset = CharacterDataset(test_data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # 评估基础模型
        base_model = self.model_evaluator.load_model(base_model_path, num_classes=len(dataset.class_to_idx))
        base_results = self.model_evaluator.evaluate_model(base_model, dataloader)
        
        # 评估更新后模型（假设已存在）
        updated_model_path = base_model_path.replace('.pth', '_updated.pth')
        if os.path.exists(updated_model_path):
            updated_model = self.model_evaluator.load_model(updated_model_path, num_classes=len(dataset.class_to_idx))
            updated_results = self.model_evaluator.evaluate_model(updated_model, dataloader)
        else:
            updated_results = {}
            logger.warning(f"更新后模型不存在: {updated_model_path}")
        
        return {
            'base_model': base_results,
            'updated_model': updated_results
        }
    
    def generate_report(self, results, output_path):
        """生成综合评估报告"""
        logger.info("生成综合评估报告...")
        
        # 生成报告内容
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }
        
        # 保存报告
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"综合评估报告已保存到: {output_path}")
        
        # 打印报告摘要
        self._print_report_summary(report)
    
    def _print_report_summary(self, report):
        """打印报告摘要"""
        logger.info("\n=== 综合评估报告摘要 ===")
        
        # 模型评估结果
        if 'model_evaluation' in report['results']:
            model_results = report['results']['model_evaluation']
            logger.info("\n1. 模型评估结果:")
            for model_name, results in model_results.items():
                if results:
                    logger.info(f"   {model_name}: Accuracy={results['accuracy']:.2f}%, FPS={results['fps']:.2f}")
        
        # 数据集扩充效果
        if 'dataset_expansion' in report['results']:
            expansion_results = report['results']['dataset_expansion']
            logger.info("\n2. 数据集扩充效果:")
            logger.info(f"   原始数据集: {expansion_results['original_dataset']['num_classes']} 类, {expansion_results['original_dataset']['total_images']} 张图像")
            logger.info(f"   扩充后数据集: {expansion_results['expanded_dataset']['num_classes']} 类, {expansion_results['expanded_dataset']['total_images']} 张图像")
            logger.info(f"   扩充比例: {expansion_results['expansion_ratio']:.2f}x")
            logger.info(f"   类别增加: {expansion_results['class_increase']} 个")
        
        # 在线学习能力
        if 'online_learning' in report['results']:
            online_results = report['results']['online_learning']
            logger.info("\n3. 在线学习能力:")
            if online_results['base_model']:
                logger.info(f"   基础模型: Accuracy={online_results['base_model']['accuracy']:.2f}%")
            if online_results['updated_model']:
                logger.info(f"   更新后模型: Accuracy={online_results['updated_model']['accuracy']:.2f}%")
        
        logger.info("\n=== 评估完成 ===")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='综合评估脚本')
    parser.add_argument('--original-data-dir', type=str, default='data/original', help='原始数据集目录')
    parser.add_argument('--expanded-data-dir', type=str, default='data/train', help='扩充后数据集目录')
    parser.add_argument('--test-data-dir', type=str, default='data/split_dataset/val', help='测试数据集目录')
    parser.add_argument('--base-model', type=str, default='models/character_classifier.pth', help='基础模型路径')
    parser.add_argument('--distilled-model', type=str, default='models/distillation/student_model_best.pth', help='蒸馏模型路径')
    parser.add_argument('--multimodal-model', type=str, default='models/multimodal_model_trained.pth', help='多模态模型路径')
    parser.add_argument('--output-report', type=str, default='reports/comprehensive_evaluation_report.json', help='输出评估报告路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    args = parser.parse_args()
    
    # 初始化综合评估器
    evaluator = ComprehensiveEvaluator(device=args.device)
    
    # 评估模型
    model_paths = {
        'base_model': args.base_model,
        'distilled_model': args.distilled_model,
        'multimodal_model': args.multimodal_model
    }
    model_results = evaluator.evaluate_models(model_paths, args.test_data_dir)
    
    # 评估数据集扩充效果
    dataset_results = evaluator.evaluate_dataset_expansion(args.original_data_dir, args.expanded_data_dir)
    
    # 评估在线学习能力
    online_results = evaluator.evaluate_online_learning(args.base_model, args.expanded_data_dir, args.test_data_dir)
    
    # 整合所有结果
    results = {
        'model_evaluation': model_results,
        'dataset_expansion': dataset_results,
        'online_learning': online_results
    }
    
    # 生成综合评估报告
    evaluator.generate_report(results, args.output_report)
    
    logger.info("综合评估完成")

if __name__ == '__main__':
    main()
