#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的标签生成模块
"""

import os
import argparse
import torch
from PIL import Image
import json
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from core.logging.global_logger import get_logger

logger = get_logger("image_tagger")


class ImageTagger:
    """图像标签生成器"""
    def __init__(self, device=None):
        self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.tags = []
        self.logger = get_logger("image_tagger")
    
    def load_model(self, model_name="openai/clip-vit-base-patch32"):
        """加载CLIP模型
        
        Args:
            model_name: 模型名称
        """
        try:
            self.logger.info(f"加载模型: {model_name}")
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # 定义动画角色相关的标签
            self.tags = [
                '1girl', 'solo', 'blue hair', 'blue eyes', 'school uniform',
                'halo', 'ribbon', 'twintails', 'smile', 'looking at viewer',
                'long hair', 'short hair', 'blonde hair', 'black hair', 'red hair',
                'green hair', 'purple hair', 'pink hair', 'brown hair', 'grey hair',
                'yellow hair', 'red eyes', 'green eyes', 'purple eyes', 'brown eyes',
                'yellow eyes', 'pink eyes', 'grey eyes', 'black eyes', 'white eyes',
                'aqua eyes', 'orange eyes', 'multicolored eyes', 'heterochromia',
                'cat ears', 'animal ears', 'horns', 'wings', 'tail',
                'bun', 'ponytail', 'braids', 'single braid', 'ahoge',
                'hat', 'cap', 'headband', 'bandana', 'helmet',
                'glasses', 'sunglasses', 'mask', 'headphones', 'earphones',
                'necklace', 'bracelet', 'ring', 'earrings', 'choker',
                'dress', 'skirt', 'pants', 'shorts', 'jacket',
                'sweater', 'hoodie', 't-shirt', 'blouse', 'coat',
                'swimsuit', 'uniform', 'costume', 'maid outfit', 'nurse outfit',
                'school uniform', 'gym uniform', 'sailor uniform', 'military uniform',
                'weapon', 'sword', 'gun', 'shield', 'staff',
                'book', 'bag', 'backpack', 'umbrella', 'phone',
                'computer', 'camera', 'headphones', 'musical instrument',
                'smile', 'laugh', 'sad', 'angry', 'surprised',
                'confused', 'happy', 'calm', 'excited', 'tired',
                'blush', 'sweat', 'tears', 'closed eyes', 'open mouth',
                'tongue', 'wink', 'grin', 'frown', 'pout',
                'looking at viewer', 'looking away', 'side view', 'front view', 'back view',
                'close-up', 'medium shot', 'full body', 'upper body', 'lower body',
                'outdoors', 'indoors', 'school', 'room', 'street',
                'park', 'beach', 'mountain', 'forest', 'city',
                'night', 'day', 'sunset', 'sunrise', 'raining',
                'snowing', 'cloudy', 'clear sky', 'stars', 'moon',
                'anime', 'cartoon', 'digital art', 'illustration', '3D',
                'high quality', 'masterpiece', 'best quality', 'detailed', 'beautiful',
                'cute', 'sexy', 'cool', 'adorable', 'stylish',
                'simple background', 'complex background', 'gradient background', 'solid color background'
            ]
            
            self.logger.info("模型加载成功！")
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def generate_tags(self, image_path, threshold=0.25, top_k=20):
        """生成图像标签
        
        Args:
            image_path: 图像路径
            threshold: 置信度阈值
            top_k: 返回前k个标签
        
        Returns:
            list: 标签列表
        """
        if not self.model:
            raise ValueError("模型未加载，请先调用load_model()")
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 处理图像和文本
            inputs = self.processor(
                text=self.tags,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # 移动到设备
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
            
            # 过滤并排序标签
            tag_probs = [(self.tags[i], probs[i].item()) for i in range(len(self.tags))]
            tag_probs = [(tag, prob) for tag, prob in tag_probs if prob > threshold]
            tag_probs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前k个标签
            return [tag for tag, prob in tag_probs[:top_k]]
        except Exception as e:
            self.logger.error(f"生成标签失败: {e}")
            return []
    
    def batch_generate_tags(self, image_dir, output_file, threshold=0.25, top_k=20):
        """批量生成标签
        
        Args:
            image_dir: 图像目录
            output_file: 输出文件路径
            threshold: 置信度阈值
            top_k: 返回前k个标签
        """
        results = []
        
        # 遍历图像目录
        for root, _, files in os.walk(image_dir):
            for file in tqdm(files, desc='批量生成标签'):
                if file.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(image_path, image_dir)
                    
                    # 生成标签
                    tags = self.generate_tags(image_path, threshold, top_k)
                    
                    # 保存结果
                    results.append({
                        'image_path': relative_path,
                        'tags': tags
                    })
        
        # 保存结果
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"批量生成标签完成，保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='统一的标签生成脚本')
    parser.add_argument('--image-path', type=str, default=None, help='单张图像路径')
    parser.add_argument('--image-dir', type=str, default=None, help='图像目录')
    parser.add_argument('--output-file', type=str, default='tags.json', help='输出文件路径')
    parser.add_argument('--threshold', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--top-k', type=int, default=20, help='返回前k个标签')
    parser.add_argument('--model-name', type=str, default='openai/clip-vit-base-patch32', help='模型名称')
    
    args = parser.parse_args()
    
    # 创建标签生成器
    tagger = ImageTagger()
    tagger.load_model(args.model_name)
    
    # 处理单张图像
    if args.image_path:
        tags = tagger.generate_tags(args.image_path, args.threshold, args.top_k)
        logger.info(f"图像: {args.image_path}")
        logger.info(f"生成的标签: {tags}")
    
    # 处理图像目录
    elif args.image_dir:
        tagger.batch_generate_tags(args.image_dir, args.output_file, args.threshold, args.top_k)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()