#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建角色索引脚本
"""

import os
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

def build_role_index(data_dir, index_dir="role_index"):
    """
    从数据集目录构建角色索引
    
    Args:
        data_dir: 数据集目录
        index_dir: 索引保存目录
    """
    print(f"开始构建角色索引...")
    print(f"数据集目录: {data_dir}")
    print(f"索引保存目录: {index_dir}")
    
    # 创建索引目录
    os.makedirs(index_dir, exist_ok=True)
    
    # 加载CLIP模型
    print("加载CLIP模型...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP模型加载完成")
    
    # 获取设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 遍历数据集目录
    role_count = 0
    for role_name in os.listdir(data_dir):
        role_dir = os.path.join(data_dir, role_name)
        if not os.path.isdir(role_dir):
            continue
        
        # 收集图像路径
        image_paths = []
        for filename in os.listdir(role_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_paths.append(os.path.join(role_dir, filename))
        
        if not image_paths:
            continue
        
        print(f"处理角色: {role_name} ({len(image_paths)} 张图像)")
        
        # 计算角色嵌入
        embeddings = []
        for image_path in image_paths[:50]:  # 最多使用50张图像
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().numpy()[0])
            except Exception as e:
                print(f"  跳过损坏图像: {image_path} - {e}")
                continue
        
        if embeddings:
            # 计算平均嵌入
            average_embedding = np.mean(embeddings, axis=0)
            average_embedding = average_embedding / np.linalg.norm(average_embedding)
            
            # 保存嵌入
            embedding_path = os.path.join(index_dir, f"{role_name}.npy")
            np.save(embedding_path, average_embedding)
            role_count += 1
            print(f"  ✓ 已保存角色嵌入")
        else:
            print(f"  ✗ 无法生成嵌入")
    
    print(f"\n索引构建完成！共处理 {role_count} 个角色")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="构建角色索引")
    parser.add_argument("--dataset", type=str, default="data/combined_dataset", help="数据集目录")
    parser.add_argument("--output", type=str, default="role_index", help="索引输出目录")
    
    args = parser.parse_args()
    
    build_role_index(args.dataset, args.output)
