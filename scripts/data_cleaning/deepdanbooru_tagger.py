#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepDanbooru 标签生成器 - 使用 DeepDanbooru 模型分析动漫图像内容
"""

import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 需要过滤的不当标签
FILTERED_TAGS = {
    'bone', 'bone nail', 'bone nails', 'skeleton', 'skull',
    'gore', 'blood', 'violence', 'death',
    'nsfw', 'nudity', 'explicit', 'porn', 'hentai',
    'offensive', 'hateful', 'racist', 'sexist',
    'self-harm', 'suicide', 'depression',
    'drug', 'alcohol', 'smoking', 'cigarette',
    'weapon', 'gun', 'knife', 'sword', 'explosion',
    'rifle', 'spear', 'bow', 'arrow'
}

# 安全的角色特征标签
SAFE_CHARACTER_FEATURES = {
    'Tsukiyo': ['blue hair', 'long hair', 'blue eyes', 'school uniform', 'serafuku', 'calm'],
    'Hina': ['pink hair', 'long hair', 'pink eyes', 'school uniform', 'gentle', 'smile'],
    'Madoka': ['pink hair', 'twintails', 'pink eyes', 'magical girl', 'pink dress'],
    'Homura': ['black hair', 'long hair', 'purple eyes', 'school uniform', 'serious'],
    'Sayaka': ['blue hair', 'ponytail', 'blue eyes', 'magical girl'],
    'Mami': ['blonde hair', 'twin drills', 'yellow eyes', 'magical girl'],
    'Kyoko': ['red hair', 'ponytail', 'orange eyes', 'magical girl'],
    'Arona': ['blue hair', 'short hair', 'blue eyes', 'school uniform', 'robot', 'halo'],
    'Shiroko': ['white hair', 'short hair', 'blue eyes', 'school uniform'],
    'Default': ['anime', 'character', 'portrait']
}


class DeepDanbooruTagger:
    """DeepDanbooru 标签生成器"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = 'cpu'
    
    def load_model(self, model_name="DeepDanbooru/deepdanbooru-v3-20211112-sgd-e30"):
        """加载 DeepDanbooru 模型"""
        try:
            print(f"📦 加载 DeepDanbooru 模型: {model_name}")
            
            # 设置 Hugging Face 缓存目录
            os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', '..', 'huggingface_cache')
            
            from transformers import AutoFeatureExtractor, AutoModelForImageClassification
            
            # 加载处理器和模型
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            
            # 检查是否有可用的 GPU
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    self.model = self.model.cuda()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = 'mps'
                    self.model = self.model.to('mps')
            except Exception:
                pass
            
            self.model.eval()
            print(f"✅ 模型加载成功，使用设备: {self.device}")
            return True
        except Exception as e:
            print(f"⚠️ 加载 DeepDanbooru 模型失败: {e}")
            print("   将使用简单标签生成方法")
            return False
    
    def is_inappropriate(self, tag):
        """检查标签是否不当"""
        tag_lower = tag.lower()
        for forbidden in FILTERED_TAGS:
            if forbidden in tag_lower:
                return True
        return False
    
    def generate_tags(self, image_path, threshold=0.5, max_tags=20):
        """生成图像标签"""
        tags = []
        
        if self.model and self.processor:
            try:
                from PIL import Image
                import torch
                
                # 加载图像
                image = Image.open(image_path).convert('RGB')
                
                # 预处理
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                
                # 推理
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # 获取预测结果
                logits = outputs.logits
                probabilities = torch.nn.functional.sigmoid(logits).squeeze().cpu().numpy()
                
                # 获取标签
                id2label = self.model.config.id2label
                for i, prob in enumerate(probabilities):
                    if prob >= threshold:
                        tag = id2label.get(i, f"LABEL_{i}")
                        # 过滤不当标签
                        if not self.is_inappropriate(tag):
                            tags.append({"tag": tag, "confidence": float(prob)})
                
                # 按置信度排序
                tags.sort(key=lambda x: x['confidence'], reverse=True)
                
                # 限制标签数量
                tags = tags[:max_tags]
                
            except Exception as e:
                print(f"⚠️ 模型生成标签失败: {e}")
        
        # 如果没有生成任何标签或模型未加载，使用默认标签
        if not tags:
            tags = [{"tag": t, "confidence": 0.5} for t in SAFE_CHARACTER_FEATURES['Default']]
        
        return tags


def process_directory(data_dir, output_dir, tagger):
    """处理目录中的所有图片"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_tags = {}
    processed_count = 0
    
    for role_name in tqdm(os.listdir(data_dir), desc="处理角色"):
        role_dir = os.path.join(data_dir, role_name)
        if not os.path.isdir(role_dir):
            continue
        
        out_role_dir = os.path.join(output_dir, role_name)
        os.makedirs(out_role_dir, exist_ok=True)
        
        role_tags = {}
        
        for filename in os.listdir(role_dir):
            if not filename.lower().endswith(('.jpg', '.png', '.webp', '.jpeg')):
                continue
            
            in_path = os.path.join(role_dir, filename)
            out_filename = os.path.splitext(filename)[0] + '.jpg'
            out_path = os.path.join(out_role_dir, out_filename)
            
            # 复制图片
            try:
                Image.open(in_path).convert('RGB').save(out_path, 'JPEG', quality=95)
                
                # 生成标签
                tags = tagger.generate_tags(out_path)
                tag_list = [t['tag'] for t in tags]
                
                # 添加角色特征标签
                if role_name in SAFE_CHARACTER_FEATURES:
                    for safe_tag in SAFE_CHARACTER_FEATURES[role_name]:
                        if safe_tag not in tag_list:
                            tag_list.append(safe_tag)
                
                role_tags[out_filename] = tag_list
                processed_count += 1
            except Exception as e:
                print(f"❌ 处理失败 {in_path}: {e}")
        
        all_tags[role_name] = role_tags
    
    # 保存标签文件
    tags_file = os.path.join(output_dir, 'image_tags.json')
    with open(tags_file, 'w', encoding='utf-8') as f:
        json.dump(all_tags, f, ensure_ascii=False, indent=2)
    
    return processed_count, tags_file


def main():
    parser = argparse.ArgumentParser(description='DeepDanbooru 标签生成器')
    parser.add_argument('--data-dir', type=str, default='./data/merged_dataset', help='输入数据目录')
    parser.add_argument('--output-dir', type=str, default='./data_cleaned_deepdanbooru', help='输出目录')
    parser.add_argument('--threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--max-tags', type=int, default=20, help='最大标签数量')
    args = parser.parse_args()
    
    print("🚀 DeepDanbooru 标签生成")
    print("=" * 60)
    
    # 创建标签生成器
    tagger = DeepDanbooruTagger()
    tagger.load_model()
    
    # 处理目录
    print(f"\n📁 处理目录: {args.data_dir}")
    processed, tags_file = process_directory(args.data_dir, args.output_dir, tagger)
    
    print(f"\n✅ 处理完成!")
    print(f"   处理图片: {processed} 张")
    print(f"   输出目录: {args.output_dir}")
    print(f"   标签文件: {tags_file}")
    
    # 统计标签
    with open(tags_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_tags = set()
    for images in data.values():
        for tags in images.values():
            all_tags.update(tags)
    
    print(f"\n📊 标签统计:")
    print(f"   标签种类: {len(all_tags)}")
    print(f"   前20个标签: {', '.join(sorted(list(all_tags))[:20])}")


if __name__ == '__main__':
    main()
