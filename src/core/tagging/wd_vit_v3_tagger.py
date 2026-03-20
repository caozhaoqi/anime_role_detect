#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WD Vit Tagger v3 模型集成
"""

import os
import argparse
import torch
from PIL import Image
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageClassification, CLIPProcessor, CLIPModel
import requests

from src.core.logging.global_logger import get_logger

logger = get_logger("wd_vit_v3_tagger")


class WDViTV3Tagger:
    """WD Vit Tagger v3 标签生成器"""
    def __init__(self, device=None):
        self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.wd_model = None
        self.wd_processor = None
        self.clip_model = None
        self.clip_processor = None
        self.id2label = {}
        self.num_id2label = {}
        self.logger = get_logger("wd_vit_v3_tagger")
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
    
    def load_model(self, model_name="SmilingWolf/wd-vit-tagger-v3"):
        """加载WD Vit Tagger v3模型
        
        Args:
            model_name: 模型名称
        """
        try:
            self.logger.info(f"加载模型: {model_name}")
            self.wd_processor = AutoProcessor.from_pretrained(model_name)
            self.wd_model = AutoModelForImageClassification.from_pretrained(model_name)
            self.wd_model.to(self.device)
            self.wd_model.eval()
            
            # 加载CLIP模型作为替代
            self.logger.info("加载CLIP模型作为标签生成的替代方案...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            self.clip_model.eval()
            
            # 获取标签映射
            self.id2label = self.wd_model.config.id2label
            
            # 尝试从Hugging Face获取标签映射
            try:
                # 下载标签映射文件 (优先尝试 selected_tags.csv)
                labels_url = f"https://huggingface.co/{model_name}/raw/main/selected_tags.csv"
                response = requests.get(labels_url)
                if response.status_code == 200:
                    # 解析CSV文件
                    import csv
                    csv_content = response.text
                    reader = csv.reader(csv_content.splitlines())
                    # 读取表头
                    header = next(reader)
                    self.logger.info(f"CSV表头: {header}")
                    # 构建数字ID到标签的映射
                    for i, row in enumerate(reader):
                        if len(row) > 1:
                            # 假设第二列是标签名称
                            label = row[1].strip()
                            self.num_id2label[i] = label
                        elif len(row) > 0:
                            # 如果只有一列，使用该列作为标签
                            label = row[0].strip()
                            self.num_id2label[i] = label
                    self.logger.info(f"从Hugging Face获取标签映射成功！标签数量: {len(self.num_id2label)}")
                    self.logger.info(f"前10个标签: {list(self.num_id2label.items())[:10]}")
                else:
                    # 尝试 labels.json
                    labels_url = f"https://huggingface.co/{model_name}/raw/main/labels.json"
                    response = requests.get(labels_url)
                    if response.status_code == 200:
                        labels_data = response.json()
                        # 构建数字ID到标签的映射
                        for i, label in enumerate(labels_data):
                            self.num_id2label[i] = label
                        self.logger.info(f"从Hugging Face获取标签映射成功！标签数量: {len(self.num_id2label)}")
                    else:
                        self.logger.warning("无法从Hugging Face获取标签映射，将使用CLIP模型生成标签")
            except Exception as e:
                self.logger.warning(f"获取标签映射失败: {e}，将使用CLIP模型生成标签")
            
            self.logger.info(f"模型加载成功！")
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def generate_tags(self, image_path, threshold=0.05):
        """生成图像标签
        
        Args:
            image_path: 图像路径
            threshold: 置信度阈值
        
        Returns:
            list: 标签列表
        """
        if not self.wd_model:
            raise ValueError("模型未加载，请先调用load_model()")
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            self.logger.info(f"加载图像成功: {image_path}")
            
            # 首先尝试使用WD Vit Tagger v3
            try:
                # 处理图像
                inputs = self.wd_processor(images=image, return_tensors="pt")
                
                # 移动到设备
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                
                # 推理
                with torch.no_grad():
                    outputs = self.wd_model(**inputs)
                    logits = outputs.logits
                    probs = torch.sigmoid(logits)[0]
                
                # 过滤并排序标签
                tag_probs = []
                for i in range(len(probs)):
                    prob = probs[i].item()
                    if prob > threshold:
                        # 获取标签
                        if i in self.num_id2label:
                            tag = self.num_id2label[i]
                        elif str(i) in self.id2label:
                            tag = self.id2label[str(i)]
                        else:
                            tag = f"LABEL_{i}"
                        # 去除LABEL_前缀
                        if not tag.startswith('LABEL_'):
                            tag_probs.append((tag, prob))
                
                # 排序
                tag_probs.sort(key=lambda x: x[1], reverse=True)
                
                # 打印前10个标签
                self.logger.info(f"WD Vit Tagger v3 前10个标签: {[(tag, prob) for tag, prob in tag_probs[:10]]}")
                
                # 如果有有意义的标签，返回
                if tag_probs:
                    return [tag for tag, prob in tag_probs]
                else:
                    self.logger.info("WD Vit Tagger v3未生成有意义的标签，使用CLIP模型作为替代")
            except Exception as e:
                self.logger.warning(f"WD Vit Tagger v3推理失败: {e}，使用CLIP模型作为替代")
            
            # 使用CLIP模型作为替代
            if self.clip_model:
                # 处理图像和文本
                inputs = self.clip_processor(
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
                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)[0]
                
                # 过滤并排序标签
                tag_probs = [(self.tags[i], probs[i].item()) for i in range(len(self.tags))]
                tag_probs = [(tag, prob) for tag, prob in tag_probs if prob > threshold]
                tag_probs.sort(key=lambda x: x[1], reverse=True)
                
                # 打印前10个标签
                self.logger.info(f"CLIP模型 前10个标签: {[(tag, prob) for tag, prob in tag_probs[:10]]}")
                
                # 返回标签
                return [tag for tag, prob in tag_probs]
            else:
                return []
        except Exception as e:
            self.logger.error(f"生成标签失败: {e}")
            return []
    
    def batch_generate_tags(self, image_dir, output_file, threshold=0.05):
        """批量生成标签
        
        Args:
            image_dir: 图像目录
            output_file: 输出文件路径
            threshold: 置信度阈值
        """
        results = []
        
        # 遍历图像目录
        for root, _, files in os.walk(image_dir):
            for file in tqdm(files, desc='批量生成标签'):
                if file.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(image_path, image_dir)
                    
                    # 生成标签
                    tags = self.generate_tags(image_path, threshold)
                    
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
    parser = argparse.ArgumentParser(description='WD Vit Tagger v3 标签生成脚本')
    parser.add_argument('--image-path', type=str, default=None, help='单张图像路径')
    parser.add_argument('--image-dir', type=str, default=None, help='图像目录')
    parser.add_argument('--output-file', type=str, default='tags.json', help='输出文件路径')
    parser.add_argument('--threshold', type=float, default=0.05, help='置信度阈值')
    parser.add_argument('--model-name', type=str, default='SmilingWolf/wd-vit-tagger-v3', help='模型名称')
    
    args = parser.parse_args()
    
    # 创建标签生成器
    tagger = WDViTV3Tagger()
    tagger.load_model(args.model_name)
    
    # 处理单张图像
    if args.image_path:
        tags = tagger.generate_tags(args.image_path, args.threshold)
        logger.info(f"图像: {args.image_path}")
        logger.info(f"生成的标签: {tags}")
    
    # 处理图像目录
    elif args.image_dir:
        tagger.batch_generate_tags(args.image_dir, args.output_file, args.threshold)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()