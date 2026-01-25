#!/usr/bin/env python3
"""
验证训练结果脚本
使用测试图像验证训练好的模型性能
"""
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('validate_training_result')


class CharacterClassifier(nn.Module):
    """角色分类器模型"""
    
    def __init__(self, num_classes):
        """初始化模型
        
        Args:
            num_classes: 类别数量
        """
        super().__init__()
        # 使用EfficientNet-B0作为基础模型
        from torchvision import models
        self.backbone = models.efficientnet_b0(pretrained=False)
        # 替换分类头
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, 
            num_classes
        )
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


def validate_model(args):
    """验证模型
    
    Args:
        args: 命令行参数
    """
    # 检测设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 构建类别映射
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
    else:
        # 如果模型中没有类别映射，从验证集目录构建
        val_dir = 'data/split_dataset/val'
        classes = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    # 初始化模型
    model = CharacterClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"模型加载成功，包含 {num_classes} 个类别")
    logger.info(f"类别列表: {list(class_to_idx.keys())}")
    
    # 测试图像
    test_images = [
        'data/all_characters/原神_丽莎/safebooru_lisa_(genshin_impact)_6269075.jpg',
        'data/all_characters/原神_凯亚/safebooru_kaeya_(genshin_impact)_6397349.jpg',
        'data/all_characters/原神_安柏/safebooru_amber_(genshin_impact)_6217488.jpg',
        'data/all_characters/原神_温迪/safebooru_venti_(genshin_impact)_6382977.png',
        'data/all_characters/原神_琴/safebooru_jean_(genshin_impact)_6381326.jpg',
    ]
    
    # 验证每个测试图像
    for img_path in test_images:
        if not os.path.exists(img_path):
            logger.warning(f"测试图像不存在: {img_path}")
            continue
        
        logger.info(f"验证图像: {img_path}")
        
        # 加载和预处理图像
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 模型预测
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, 1)[0][preds].item()
        
        predicted_class = idx_to_class[preds.item()]
        logger.info(f"预测结果: {predicted_class}, 置信度: {confidence:.4f}")
        print(f"\n验证图像: {os.path.basename(img_path)}")
        print(f"预测结果: {predicted_class}")
        print(f"置信度: {confidence:.4f}")
    
    # 批量验证验证集
    logger.info("\n开始批量验证验证集...")
    val_dir = 'data/split_dataset/val'
    correct = 0
    total = 0
    
    for class_name in os.listdir(val_dir):
        class_path = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            img_path = os.path.join(class_path, img_name)
            
            # 加载和预处理图像
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # 模型预测
            with torch.no_grad():
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)
            
            predicted_class = idx_to_class[preds.item()]
            if predicted_class == class_name:
                correct += 1
            total += 1
    
    if total > 0:
        accuracy = correct / total
        logger.info(f"批量验证完成！准确率: {accuracy:.4f}, 正确: {correct}, 总样本: {total}")
        print(f"\n批量验证结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"正确预测: {correct}")
        print(f"总样本数: {total}")
    else:
        logger.warning("没有找到验证样本")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='验证训练结果')
    parser.add_argument('--model_path', type=str, default='models/character_classifier_best.pth', help='模型路径')
    args = parser.parse_args()
    
    logger.info('开始验证训练结果...')
    logger.info(f'模型路径: {args.model_path}')
    
    validate_model(args)
    
    logger.info('验证完成！')


if __name__ == "__main__":
    main()
