#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型融合策略
结合多个模型的预测结果，提升分类准确率
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_ensemble')


class ModelEnsemble:
    """模型集成类"""
    
    def __init__(self, model_paths, device='cpu'):
        self.models = []
        self.device = device
        
        for model_path in model_paths:
            logger.info(f"加载模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # 根据模型类型创建相应的模型架构
            if 'efficientnet' in model_path.lower():
                from torchvision import models
                model = models.efficientnet_b3(pretrained=False)
                num_classes = checkpoint['class_to_idx'].__len__() if 'class_to_idx' in checkpoint else 22
                model.classifier = nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(model.classifier[1].in_features, 768),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(768),
                    nn.Dropout(0.2),
                    nn.Linear(768, num_classes)
                )
            else:
                from torchvision import models
                model = models.mobilenet_v2(pretrained=False)
                num_classes = checkpoint['class_to_idx'].__len__() if 'class_to_idx' in checkpoint else 22
                model.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(model.classifier[1].in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.15),
                    nn.Linear(512, num_classes)
                )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            self.models.append(model)
            logger.info(f"模型加载完成，类别数: {num_classes}")
        
        logger.info(f"共加载 {len(self.models)} 个模型")
    
    def predict(self, image, transform):
        """使用所有模型进行预测
        
        Args:
            image: PIL Image对象
            transform: 图像变换
        
        Returns:
            融合后的预测结果
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # 收集所有模型的预测
        all_predictions = []
        with torch.no_grad():
            for model in self.models:
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                all_predictions.append(probs.cpu().numpy())
        
        # 融合策略
        ensemble_probs = self._ensemble_predictions(all_predictions)
        
        return ensemble_probs
    
    def _ensemble_predictions(self, predictions):
        """融合多个模型的预测结果
        
        Args:
            predictions: list of numpy arrays, shape (num_models, 1, num_classes)
        
        Returns:
            融合后的概率分布
        """
        predictions = np.array(predictions)
        
        # 策略1: 平均融合
        avg_probs = np.mean(predictions, axis=0)[0]
        
        # 策略2: 加权平均（基于模型准确率）
        weights = np.array([1.0] * len(predictions))
        weighted_probs = np.average(predictions, axis=0, weights=weights)[0]
        
        # 策略3: 最大值融合
        max_probs = np.max(predictions, axis=0)[0]
        
        # 策略4: 投票融合
        votes = np.argmax(predictions, axis=2)
        vote_probs = np.zeros_like(predictions[0][0])
        for vote in votes[0]:
            vote_probs[vote] += 1
        vote_probs = vote_probs / len(predictions)
        
        # 返回平均融合结果
        return avg_probs
    
    def predict_batch(self, images, transform):
        """批量预测"""
        if isinstance(images[0], str):
            images = [Image.open(img).convert('RGB') for img in images]
        
        batch_tensors = torch.stack([transform(img) for img in images]).to(self.device)
        
        all_predictions = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(batch_tensors)
                probs = F.softmax(outputs, dim=1)
                all_predictions.append(probs.cpu().numpy())
        
        predictions = np.array(all_predictions)
        avg_probs = np.mean(predictions, axis=0)
        
        return avg_probs


def evaluate_ensemble(ensemble, data_dir, class_to_idx, transform):
    """评估集成模型性能"""
    
    device = ensemble.device
    correct = 0
    total = 0
    per_class_correct = {cls: 0 for cls in class_to_idx}
    per_class_total = {cls: 0 for cls in class_to_idx}
    
    for character in os.listdir(data_dir):
        character_dir = os.path.join(data_dir, character)
        if not os.path.isdir(character_dir):
            continue
        
        if character not in class_to_idx:
            continue
        
        for img_name in os.listdir(character_dir):
            img_path = os.path.join(character_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                continue
            
            # 预测
            probs = ensemble.predict(img_path, transform)
            predicted_class = np.argmax(probs)
            
            # 获取真实标签
            true_class = class_to_idx[character]
            
            # 统计
            total += 1
            per_class_total[character] += 1
            
            if predicted_class == true_class:
                correct += 1
                per_class_correct[character] += 1
    
    accuracy = 100 * correct / total if total > 0 else 0
    logger.info(f"集成模型准确率: {accuracy:.2f}% ({correct}/{total})")
    
    # 打印每个类别的准确率
    logger.info("\n各类别准确率:")
    for character in sorted(class_to_idx.keys()):
        if per_class_total[character] > 0:
            class_acc = 100 * per_class_correct[character] / per_class_total[character]
            logger.info(f"  {character}: {class_acc:.2f}% ({per_class_correct[character]}/{per_class_total[character]})")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='模型集成评估')
    parser.add_argument('--model-paths', type=str, nargs='+', required=True, 
                       help='模型文件路径列表')
    parser.add_argument('--data-dir', type=str, default='../../data/train', help='数据目录')
    parser.add_argument('--device', type=str, default='mps', 
                       choices=['cpu', 'mps', 'cuda'], help='设备')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if (args.device == 'cuda' and torch.cuda.is_available()) or 
                          (args.device == 'mps' and torch.backends.mps.is_available()) else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建集成模型
    ensemble = ModelEnsemble(args.model_paths, device=device)
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.CenterCrop((288, 288)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载类别映射
    class_to_idx = {}
    for model_path in args.model_paths:
        checkpoint = torch.load(model_path, map_location=device)
        if 'class_to_idx' in checkpoint:
            class_to_idx.update(checkpoint['class_to_idx'])
            break
    
    if not class_to_idx:
        logger.error("无法找到类别映射")
        return
    
    logger.info(f"类别数: {len(class_to_idx)}")
    
    # 评估集成模型
    accuracy = evaluate_ensemble(ensemble, args.data_dir, class_to_idx, transform)
    
    logger.info(f"\n最终集成准确率: {accuracy:.2f}%")


if __name__ == '__main__':
    main()
