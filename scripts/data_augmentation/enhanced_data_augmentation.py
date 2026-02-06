#!/usr/bin/env python3
"""
增强数据工程实现
包括画风迁移和硬样本挖掘
"""
import os
import sys
import argparse
import logging
import random
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('enhanced_data_augmentation')

class EnhancedDataAugmenter:
    """
    增强数据增强器
    实现画风迁移和硬样本挖掘
    """
    
    def __init__(self, output_dir='data/augmented'):
        """
        初始化增强器
        
        Args:
            output_dir: 增强数据输出目录
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def style_transfer(self, image_path, output_path, style_type='sketch'):
        """
        实现简单的画风迁移
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            style_type: 风格类型，可选 'sketch' (线条稿), 'paint' (油画), 'pixel' (像素风)
            
        Returns:
            bool: 是否成功
        """
        try:
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return False
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if style_type == 'sketch':
                # 线条稿风格
                # 高斯模糊
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                # Canny边缘检测
                edges = cv2.Canny(blurred, 50, 150)
                # 反转颜色
                sketch = 255 - edges
                # 转换为RGB
                result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
                
            elif style_type == 'paint':
                # 油画风格
                # 双边滤波
                colored = cv2.bilateralFilter(image, 9, 75, 75)
                # 边缘检测
                edges = cv2.Canny(gray, 50, 150)
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                
                # 确保边缘图像和彩色图像尺寸一致
                if edges.shape[:2] != colored.shape[:2]:
                    edges = cv2.resize(edges, (colored.shape[1], colored.shape[0]))
                
                # 融合边缘和彩色图像
                # 使用 mask 时，mask 必须是单通道的
                edges_mask = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
                result = cv2.bitwise_and(colored, colored, mask=255-edges_mask)
                
            elif style_type == 'pixel':
                # 像素风格
                # 缩小图像
                height, width = image.shape[:2]
                small = cv2.resize(image, (width//10, height//10), interpolation=cv2.INTER_NEAREST)
                # 放大回原始大小
                result = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
                
            else:
                logger.error(f"不支持的风格类型: {style_type}")
                return False
            
            # 保存结果
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, result)
            logger.info(f"已生成 {style_type} 风格图像: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"风格迁移失败: {e}")
            return False
    
    def random_augmentation(self, image_path, output_path):
        """
        随机数据增强
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 随机增强
            augmentations = [
                lambda img: img.rotate(random.uniform(-30, 30), expand=True),
                lambda img: ImageOps.mirror(img),
                lambda img: ImageOps.flip(img),
                lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0))),
                lambda img: img.filter(ImageFilter.Sharpen()),
                lambda img: ImageOps.autocontrast(img, cutoff=random.uniform(0, 10)),
                lambda img: ImageOps.equalize(img),
            ]
            
            # 随机选择2-3种增强
            num_aug = random.randint(2, 3)
            selected_aug = random.sample(augmentations, num_aug)
            
            # 应用增强
            augmented = image
            for aug in selected_aug:
                augmented = aug(augmented)
            
            # 调整大小
            augmented = augmented.resize((224, 224))
            
            # 保存结果
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            augmented.save(output_path)
            logger.info(f"已生成随机增强图像: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"随机增强失败: {e}")
            return False
    
    def augment_directory(self, input_dir, num_aug_per_image=3):
        """
        增强整个目录的图像
        
        Args:
            input_dir: 输入目录
            num_aug_per_image: 每个图像生成的增强样本数
            
        Returns:
            int: 成功增强的图像数量
        """
        success_count = 0
        
        # 遍历目录
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                # 计算相对路径
                relative_path = os.path.relpath(os.path.join(root, file), input_dir)
                
                for i in range(num_aug_per_image):
                    # 生成增强图像
                    if i == 0:
                        # 线条稿风格
                        output_path = os.path.join(self.output_dir, f"sketch_{i}_{relative_path}")
                        success = self.style_transfer(
                            os.path.join(root, file),
                            output_path,
                            style_type='sketch'
                        )
                    elif i == 1:
                        # 油画风格
                        output_path = os.path.join(self.output_dir, f"paint_{i}_{relative_path}")
                        success = self.style_transfer(
                            os.path.join(root, file),
                            output_path,
                            style_type='paint'
                        )
                    else:
                        # 随机增强
                        output_path = os.path.join(self.output_dir, f"random_{i}_{relative_path}")
                        success = self.random_augmentation(
                            os.path.join(root, file),
                            output_path
                        )
                    
                    if success:
                        success_count += 1
        
        logger.info(f"增强完成，成功生成 {success_count} 张增强图像")
        return success_count

class HardNegativeMiner:
    """
    硬样本挖掘
    """
    
    def __init__(self, model_path, data_dir):
        """
        初始化硬样本挖掘器
        
        Args:
            model_path: 模型权重路径
            data_dir: 数据目录
        """
        self.model_path = model_path
        self.data_dir = data_dir
    
    def load_model(self):
        """
        加载模型
        
        Returns:
            model: 加载好的模型
        """
        try:
            import torch
            from torchvision import models
            
            # 加载模型
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # 创建模型
            model = models.efficientnet_b0(pretrained=False)
            
            # 调整分类器
            if 'class_to_idx' in checkpoint:
                num_classes = len(checkpoint['class_to_idx'])
                model.classifier[1] = torch.nn.Linear(
                    model.classifier[1].in_features,
                    num_classes
                )
            
            # 加载权重
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 处理键名不匹配
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    name = k[9:]  # 移除 'backbone.'
                else:
                    name = k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            
            logger.info("模型加载成功")
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return None
    
    def find_hard_negatives(self, output_file='hard_negatives.json', confidence_threshold=0.6):
        """
        查找硬样本
        
        Args:
            output_file: 硬样本列表输出文件
            confidence_threshold: 置信度阈值，低于此值的样本被认为是硬样本
            
        Returns:
            list: 硬样本列表
        """
        try:
            import torch
            from torchvision import transforms
            
            # 加载模型
            model = self.load_model()
            if model is None:
                return []
            
            # 图像预处理
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # 获取类别信息
            checkpoint = torch.load(self.model_path, map_location='cpu')
            class_to_idx = checkpoint.get('class_to_idx', {})
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            
            # 查找硬样本
            hard_negatives = []
            
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        continue
                    
                    image_path = os.path.join(root, file)
                    
                    try:
                        # 加载和预处理图像
                        from PIL import Image
                        image = Image.open(image_path).convert('RGB')
                        image_tensor = transform(image).unsqueeze(0)
                        
                        # 预测
                        with torch.no_grad():
                            output = model(image_tensor)
                            probabilities = torch.nn.functional.softmax(output, dim=1)
                            confidence, predicted = torch.max(probabilities, dim=1)
                            
                        # 获取真实类别
                        true_class = os.path.basename(root)
                        
                        # 检查是否是硬样本
                        if confidence.item() < confidence_threshold:
                            hard_negatives.append({
                                'image_path': image_path,
                                'true_class': true_class,
                                'predicted_class': idx_to_class.get(predicted.item(), 'unknown'),
                                'confidence': confidence.item()
                            })
                            
                    except Exception as e:
                        logger.error(f"处理图像 {image_path} 失败: {e}")
                        continue
            
            # 保存硬样本列表
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(hard_negatives, f, ensure_ascii=False, indent=2)
            
            logger.info(f"硬样本挖掘完成，找到 {len(hard_negatives)} 个硬样本，已保存到 {output_file}")
            return hard_negatives
            
        except Exception as e:
            logger.error(f"硬样本挖掘失败: {e}")
            return []

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='增强数据工程工具')
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 风格迁移子命令
    style_parser = subparsers.add_parser('style_transfer', help='画风迁移')
    style_parser.add_argument('--input', type=str, required=True, help='输入图像或目录')
    style_parser.add_argument('--output', type=str, default='data/augmented', help='输出目录')
    style_parser.add_argument('--style', type=str, choices=['sketch', 'paint', 'pixel'], default='sketch', help='风格类型')
    
    # 随机增强子命令
    random_parser = subparsers.add_parser('random_augment', help='随机数据增强')
    random_parser.add_argument('--input', type=str, required=True, help='输入目录')
    random_parser.add_argument('--output', type=str, default='data/augmented', help='输出目录')
    random_parser.add_argument('--num_aug', type=int, default=3, help='每个图像生成的增强样本数')
    
    # 硬样本挖掘子命令
    hard_parser = subparsers.add_parser('find_hard', help='硬样本挖掘')
    hard_parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    hard_parser.add_argument('--data', type=str, required=True, help='数据目录')
    hard_parser.add_argument('--output', type=str, default='hard_negatives.json', help='硬样本列表输出文件')
    hard_parser.add_argument('--threshold', type=float, default=0.6, help='置信度阈值')
    
    args = parser.parse_args()
    
    if args.command == 'style_transfer':
        # 风格迁移
        augmenter = EnhancedDataAugmenter(args.output)
        
        if os.path.isdir(args.input):
            # 处理目录
            success = augmenter.augment_directory(args.input)
        else:
            # 处理单个文件
            output_path = os.path.join(args.output, f"{args.style}_" + os.path.basename(args.input))
            success = augmenter.style_transfer(args.input, output_path, args.style)
        
    elif args.command == 'random_augment':
        # 随机增强
        augmenter = EnhancedDataAugmenter(args.output)
        success = augmenter.augment_directory(args.input, args.num_aug)
        
    elif args.command == 'find_hard':
        # 硬样本挖掘
        miner = HardNegativeMiner(args.model, args.data)
        hard_negatives = miner.find_hard_negatives(args.output, args.threshold)
        
        # 打印结果
        print(f"\n硬样本挖掘结果:")
        print(f"找到 {len(hard_negatives)} 个硬样本")
        if hard_negatives:
            print("\n前5个硬样本:")
            for i, sample in enumerate(hard_negatives[:5]):
                print(f"{i+1}. {sample['image_path']}")
                print(f"   真实类别: {sample['true_class']}")
                print(f"   预测类别: {sample['predicted_class']}")
                print(f"   置信度: {sample['confidence']:.4f}")
                print()
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
