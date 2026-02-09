#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型集成脚本

通过组合多个模型的预测结果来提高整体准确率
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_ensemble')


class ModelEnsemble:
    def __init__(self, model_paths, device='cpu'):
        """
        初始化模型集成器
        
        Args:
            model_paths: 模型路径列表
            device: 设备
        """
        self.model_paths = model_paths
        self.device = device
        self.models = []
        self.class_to_idx = None
        
        logger.info(f"初始化模型集成器，共 {len(model_paths)} 个模型")
        
    def load_models(self):
        """加载所有模型"""
        for i, model_path in enumerate(self.model_paths):
            if not os.path.exists(model_path):
                logger.warning(f"模型文件不存在: {model_path}")
                continue
            
            logger.info(f"加载模型 {i+1}/{len(self.model_paths)}: {model_path}")
            
            try:
                # 加载模型
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 获取类别映射
                if self.class_to_idx is None and 'class_to_idx' in checkpoint:
                    self.class_to_idx = checkpoint['class_to_idx']
                
                # 获取类别数量
                num_classes = len(self.class_to_idx) if self.class_to_idx else checkpoint.get('num_classes', 47)
                
                # 创建自定义模型结构（支持两种不同的模型结构）
                from torchvision import models
                import torch.nn as nn
                
                # 检测模型结构类型
                classifier_keys = [k for k in checkpoint['model_state_dict'].keys() if 'classifier' in k]
                
                if len(classifier_keys) == 2:  # 简单结构: Linear(1280 -> 47)
                    class SimpleEfficientNet(nn.Module):
                        def __init__(self, num_classes=47):
                            super().__init__()
                            self.backbone = models.efficientnet_b0(pretrained=False)
                            self.backbone.classifier = nn.Linear(1280, num_classes)
                        
                        def forward(self, x):
                            return self.backbone(x)
                    
                    model = SimpleEfficientNet(num_classes=num_classes)
                    
                else:  # 复杂结构: Linear(1280 -> 512) -> ReLU -> Dropout -> Linear(512 -> 256) -> ReLU -> Dropout -> Linear(256 -> 47)
                    class ComplexEfficientNet(nn.Module):
                        def __init__(self, num_classes=47, dropout_rate=0.3):
                            super().__init__()
                            self.backbone = models.efficientnet_b0(pretrained=False)
                            self.backbone.classifier = nn.Sequential(
                                nn.Linear(1280, 512),
                                nn.ReLU(),
                                nn.Dropout(dropout_rate),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(dropout_rate),
                                nn.Linear(256, num_classes)
                            )
                        
                        def forward(self, x):
                            return self.backbone(x)
                    
                    model = ComplexEfficientNet(num_classes=num_classes)
                
                # 加载权重
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                self.models.append(model)
                logger.info(f"模型 {i+1} 加载成功")
                
            except Exception as e:
                logger.error(f"加载模型 {i+1} 失败: {str(e)}")
                continue
        
        if not self.models:
            raise ValueError("没有成功加载任何模型")
        
        logger.info(f"成功加载 {len(self.models)} 个模型")
        
    def predict_ensemble(self, dataloader, method='voting'):
        """
        集成预测
        
        Args:
            dataloader: 数据加载器
            method: 集成方法 ('voting', 'average', 'weighted_average')
            
        Returns:
            预测结果和准确率
        """
        if not self.models:
            raise ValueError("没有可用的模型")
        
        logger.info(f"开始集成预测，方法: {method}")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        # 收集所有模型的预测
        for model in self.models:
            model_predictions = []
            model_probabilities = []
            
            with torch.no_grad():
                for inputs, labels in tqdm(dataloader, desc=f'模型 {self.models.index(model)+1} 预测'):
                    inputs = inputs.to(self.device)
                    outputs = model(inputs)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    model_predictions.append(predicted.cpu().numpy())
                    model_probabilities.append(probabilities.cpu().numpy())
            
            all_predictions.append(np.concatenate(model_predictions))
            all_probabilities.append(np.concatenate(model_probabilities))
            
            if len(all_labels) == 0:
                for _, labels in dataloader:
                    all_labels.append(labels.cpu().numpy())
                all_labels = np.concatenate(all_labels)
        
        # 集成预测
        if method == 'voting':
            ensemble_predictions = self._voting_ensemble(all_predictions)
        elif method == 'average':
            ensemble_predictions = self._average_ensemble(all_probabilities)
        elif method == 'weighted_average':
            ensemble_predictions = self._weighted_average_ensemble(all_probabilities)
        else:
            raise ValueError(f"未知的集成方法: {method}")
        
        # 计算准确率
        accuracy = np.mean(ensemble_predictions == all_labels)
        
        logger.info(f"集成预测完成，准确率: {accuracy:.4f}")
        
        return {
            'predictions': ensemble_predictions,
            'labels': all_labels,
            'accuracy': accuracy,
            'method': method
        }
    
    def _voting_ensemble(self, all_predictions):
        """
        投票集成
        
        Args:
            all_predictions: 所有模型的预测结果
            
        Returns:
            集成预测结果
        """
        # 转置数组，使每个样本的所有预测在一行
        predictions_array = np.array(all_predictions).T
        
        # 对每个样本进行投票
        ensemble_predictions = []
        for predictions in predictions_array:
            # 统计每个类别的得票数
            counter = Counter(predictions)
            # 选择得票最多的类别
            ensemble_predictions.append(counter.most_common(1)[0][0])
        
        return np.array(ensemble_predictions)
    
    def _average_ensemble(self, all_probabilities):
        """
        平均集成
        
        Args:
            all_probabilities: 所有模型的概率预测
            
        Returns:
            集成预测结果
        """
        # 计算平均概率
        avg_probabilities = np.mean(all_probabilities, axis=0)
        
        # 选择概率最大的类别
        ensemble_predictions = np.argmax(avg_probabilities, axis=1)
        
        return ensemble_predictions
    
    def _weighted_average_ensemble(self, all_probabilities):
        """
        加权平均集成
        
        Args:
            all_probabilities: 所有模型的概率预测
            
        Returns:
            集成预测结果
        """
        # 这里使用均等权重，实际应用中可以根据模型性能设置不同权重
        weights = np.ones(len(all_probabilities)) / len(all_probabilities)
        
        # 计算加权平均概率
        weighted_probabilities = np.zeros_like(all_probabilities[0])
        for i, probabilities in enumerate(all_probabilities):
            weighted_probabilities += weights[i] * probabilities
        
        # 选择概率最大的类别
        ensemble_predictions = np.argmax(weighted_probabilities, axis=1)
        
        return ensemble_predictions
    
    def evaluate_ensemble_methods(self, dataloader):
        """
        评估不同的集成方法
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            各种集成方法的评估结果
        """
        logger.info("开始评估不同的集成方法")
        
        methods = ['voting', 'average', 'weighted_average']
        results = {}
        
        for method in methods:
            logger.info(f"评估方法: {method}")
            result = self.predict_ensemble(dataloader, method=method)
            results[method] = result
        
        # 找出最佳方法
        best_method = max(results.keys(), key=lambda k: results[k]['accuracy'])
        logger.info(f"最佳集成方法: {best_method}, 准确率: {results[best_method]['accuracy']:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='模型集成脚本')
    
    parser.add_argument('--model-paths', type=str, nargs='+',
                       default=[
                           'models/character_classifier_optimized_best.pth',
                           'models/character_classifier_best.pth'
                       ],
                       help='模型路径列表')
    parser.add_argument('--val-dir', type=str, 
                       default='data/split_dataset/val',
                       help='验证集目录')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--method', type=str, 
                       choices=['voting', 'average', 'weighted_average', 'all'],
                       default='all',
                       help='集成方法')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # 检测设备
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    logger.info(f"使用设备: {device}")
    
    # 初始化集成器
    ensemble = ModelEnsemble(args.model_paths, device=device)
    
    # 加载模型
    ensemble.load_models()
    
    # 准备数据
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    # 评估集成方法
    if args.method == 'all':
        results = ensemble.evaluate_ensemble_methods(val_loader)
        
        # 打印结果
        print("\n集成方法评估结果:")
        print("-" * 50)
        for method, result in results.items():
            print(f"{method:20s}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print("-" * 50)
    else:
        result = ensemble.predict_ensemble(val_loader, method=args.method)
        print(f"\n集成方法: {result['method']}")
        print(f"准确率: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")


if __name__ == '__main__':
    main()