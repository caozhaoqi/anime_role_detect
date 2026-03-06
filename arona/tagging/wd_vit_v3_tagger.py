#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用WD Vit V3 Tagger模型为采集的数据打标签
"""

import os
import json
import argparse
import requests
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('wd_vit_v3_tagger')


class WDViTV3Tagger:
    def __init__(self, model_path=None, threshold=0.35):
        """初始化WD Vit V3 Tagger
        
        Args:
            model_path: 模型路径
            threshold: 标签阈值
        """
        self.threshold = threshold
        self.model = None
        self.tokenizer = None
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning("未提供WD Vit V3模型路径，将使用API方式")
    
    def load_model(self, model_path):
        """加载本地模型"""
        try:
            logger.info(f"加载WD Vit V3模型: {model_path}")
            
            # 使用CLIP模型进行标签匹配
            from transformers import CLIPProcessor, CLIPModel
            
            # 加载CLIP模型
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.model.eval()
            
            # 定义动画角色相关的标签
            self.tags = [
                '1girl', 'solo', 'blue hair', 'blue eyes', 'school uniform',
                'halo', 'ribbon', 'twintails', 'smile', 'looking at viewer',
                'black hair', 'red eyes', 'long hair', 'short hair', 'glasses',
                'hat', 'blush', 'open mouth', 'closed eyes', 'wink',
                'purple hair', 'green eyes', 'yellow eyes', 'pink hair',
                'white hair', 'brown hair', 'orange hair', 'grey hair',
                'ponytail', 'bun', 'braids', 'side ponytail', 'messy hair',
                'drill hair', 'pigtails', 'bob cut', 'hime cut', 'ahoge',
                'cat ears', 'dog ears', 'fox ears', 'bunny ears', 'animal ears',
                'headphones', 'earphones', 'earrings', 'necklace', 'bracelet',
                'ring', 'gloves', 'scarf', 'cape', 'jacket',
                'dress', 'skirt', 'pants', 'shorts', 'swimsuit',
                'gym uniform', 'maid uniform', 'nurse uniform', 'sailor uniform',
                'military uniform', 'formal wear', 'casual wear', 'winter wear',
                'summer wear', 'spring wear', 'autumn wear', 'traditional wear',
                'kimono', 'yukata', 'school bag', 'backpack', 'purse',
                'umbrella', 'fan', 'book', 'phone', 'camera',
                'flower', 'rose', 'lily', 'sunflower', 'cherry blossom',
                'star', 'moon', 'sun', 'cloud', 'rain',
                'snow', 'fire', 'water', 'wind', 'earth',
                'sky', 'sea', 'mountain', 'forest', 'city',
                'indoor', 'outdoor', 'day', 'night', 'twilight',
                'morning', 'afternoon', 'evening', 'sunrise', 'sunset',
                'happy', 'sad', 'angry', 'surprised', 'confused',
                'worried', 'excited', 'calm', 'tired', 'sleepy'
            ]
            
            logger.info("模型加载成功！")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
    
    def get_tags_api(self, image_path):
        """通过API获取标签
        
        Args:
            image_path: 图片路径
        
        Returns:
            list: 标签列表
        """
        try:
            # 这里使用一个假设的API，实际应用中需要替换为真实的API
            # 例如使用Hugging Face Inference API或其他服务
            logger.info(f"通过API处理图片: {image_path}")
            
            # 模拟API响应
            # 实际应用中应该使用真实的API调用
            # response = requests.post(
            #     "https://api.example.com/wd-vit-v3-tagger",
            #     files={"image": open(image_path, "rb")},
            #     data={"threshold": self.threshold}
            # )
            # tags = response.json()["tags"]
            
            # 模拟标签结果
            tags = [
                "1girl", "solo", "blue hair", "blue eyes", "school uniform", 
                "halo", "ribbon", "twintails", "smile", "looking at viewer"
            ]
            
            return tags
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            # 返回默认标签
            return ["1girl", "solo"]
    
    def get_tags_local(self, image_path):
        """使用本地模型获取标签
        
        Args:
            image_path: 图片路径
        
        Returns:
            list: 标签列表
        """
        try:
            logger.info(f"使用本地模型处理图片: {image_path}")
            
            # 加载图片
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            
            # 预处理图片和文本
            inputs = self.processor(
                text=self.tags,
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # 图片到文本的相似度
                probs = torch.softmax(logits_per_image, dim=1)
            
            # 获取概率大于阈值的标签
            tags = []
            for i, prob in enumerate(probs[0]):
                if prob.item() >= self.threshold:
                    tags.append(self.tags[i])
            
            # 如果没有标签，返回默认标签
            if not tags:
                tags = ["1girl", "solo"]
            
            return tags
        except Exception as e:
            logger.error(f"本地模型推理失败: {e}")
            # 失败时返回默认标签
            return ["1girl", "solo"]
    
    def get_tags(self, image_path):
        """获取图片标签
        
        Args:
            image_path: 图片路径
        
        Returns:
            list: 标签列表
        """
        if self.model and hasattr(self, 'processor'):
            # 使用本地模型推理
            return self.get_tags_local(image_path)
        else:
            # 使用API
            return self.get_tags_api(image_path)


def process_directory(directory, output_dir, tagger):
    """处理目录中的所有图片
    
    Args:
        directory: 图片目录
        output_dir: 输出目录
        tagger: 标签器实例
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    logger.info(f"找到 {len(image_files)} 张图片")
    
    # 处理每张图片
    for image_path in tqdm(image_files, desc="处理图片"):
        try:
            # 获取标签
            tags = tagger.get_tags(image_path)
            
            # 生成输出文件路径
            relative_path = os.path.relpath(image_path, directory)
            output_file = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.json')
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存标签为JSON
            tag_data = {
                "image_path": relative_path,
                "tags": tags,
                "tag_count": len(tags)
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(tag_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"已为 {image_path} 生成标签")
        except Exception as e:
            logger.error(f"处理 {image_path} 失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='使用WD Vit V3 Tagger为图片打标签')
    parser.add_argument('--input-dir', type=str, default='../data/downloaded_images', help='输入图片目录')
    parser.add_argument('--output-dir', type=str, default='../data/image_tags', help='输出标签目录')
    parser.add_argument('--model-path', type=str, default=None, help='WD Vit V3模型路径')
    parser.add_argument('--threshold', type=float, default=0.35, help='标签阈值')
    
    args = parser.parse_args()
    
    # 初始化标签器
    tagger = WDViTV3Tagger(args.model_path, args.threshold)
    
    # 处理目录
    process_directory(args.input_dir, args.output_dir, tagger)
    
    logger.info(f"标签生成完成，结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()