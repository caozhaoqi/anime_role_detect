#!/usr/bin/env python3
"""
背景增强脚本
1. 抠图去除背景
2. 合成到随机背景上
3. 生成增强后的训练数据
"""
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import torch
import gc

# 尝试导入rembg
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("⚠️ rembg 未安装，将使用简单背景替换")

# 配置
TRAIN_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
OUTPUT_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/augmented_dataset")
BG_IMAGES_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/backgrounds")

# 生成随机背景
def generate_random_background(width, height):
    """生成随机渐变背景"""
    colors = [
        (255, 220, 180),  # 暖色
        (220, 240, 255),  # 冷色
        (240, 255, 220),  # 绿色调
        (255, 230, 255),  # 紫色调
        (200, 200, 220),  # 灰色调
        (255, 240, 230),  # 米色
    ]
    
    bg = Image.new('RGB', (width, height), random.choice(colors))
    draw = ImageDraw.Draw(bg)
    
    # 添加一些随机形状作为干扰
    for _ in range(random.randint(2, 5)):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 200)
        color = tuple(random.randint(180, 255) for _ in range(3))
        draw.ellipse([x1, y1, x2, y2], fill=color, outline=None)
    
    return bg


def remove_background_simple(img):
    """简单的背景移除（基于颜色检测）"""
    img_array = np.array(img.convert('RGB'))
    
    # 检测边缘
    gray = np.mean(img_array, axis=2)
    edge_threshold = 30
    
    # 创建mask
    mask = np.ones_like(gray)
    
    # 简化：假设主体在中心，边缘是背景
    h, w = gray.shape
    margin = int(min(h, w) * 0.05)
    
    # 中心区域高概率为主体
    center_mask = np.zeros_like(gray)
    center_mask[margin:-margin, margin:-margin] = 1
    
    # 使用简单的颜色聚类
    pixels = img_array.reshape(-1, 3)
    
    # 中心颜色作为主体颜色
    center_pixels = img_array[margin:-margin, margin:-margin].reshape(-1, 3)
    main_color = np.mean(center_pixels, axis=0)
    
    # 计算每个像素到主体颜色的距离
    distances = np.sqrt(np.sum((pixels - main_color) ** 2, axis=1))
    distances = distances.reshape(h, w)
    
    # 距离小于阈值的保留
    threshold = np.percentile(distances, 70)
    foreground_mask = distances < threshold
    
    # 结合中心mask
    final_mask = foreground_mask & (center_mask > 0)
    
    return final_mask.astype(np.uint8) * 255


def composite_with_background(foreground, mask, background):
    """将前景合成到背景上"""
    # 确保尺寸匹配
    if foreground.size != background.size:
        background = background.resize(foreground.size, Image.LANCZOS)
    
    # 转换mask
    mask_img = Image.fromarray(mask).convert('L')
    mask_img = mask_img.resize(foreground.size, Image.LANCZOS)
    
    # 合成
    result = Image.composite(foreground, background, mask_img)
    return result


def process_character(char_dir, output_dir, num_augmented=5):
    """处理单个角色目录"""
    char_name = char_dir.name
    char_output = output_dir / char_name
    char_output.mkdir(parents=True, exist_ok=True)
    
    # 复制原始图片
    original_count = 0
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        for img_path in char_dir.glob(ext):
            if original_count >= 2:  # 最多复制2张原图
                break
            try:
                img = Image.open(img_path)
                img.save(char_output / f"original_{img_path.name}")
                original_count += 1
            except:
                pass
        if original_count >= 2:
            break
    
    # 生成增强图片
    augmented_count = 0
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        for img_path in char_dir.glob(ext):
            if augmented_count >= num_augmented:
                break
            try:
                img = Image.open(img_path)
                
                # 去除背景
                if REMBG_AVAILABLE:
                    img_no_bg = remove(img, alpha_matting=True)
                else:
                    # 使用简单方法
                    mask = remove_background_simple(img)
                    img_no_bg = Image.fromarray(np.array(img).copy())
                
                # 生成到不同背景上
                for i in range(3):  # 每个原图生成3张
                    bg = generate_random_background(img.width, img.height)
                    result = composite_with_background(img_no_bg, mask if not REMBG_AVAILABLE else None, bg)
                    
                    # 缩放并保存
                    result = result.resize((224, 224), Image.LANCZOS)
                    result.save(char_output / f"aug_{char_name}_{augmented_count}_{i}.jpg", quality=95)
                    
                    del result
                    gc.collect()
                
                augmented_count += 1
                del img_no_bg
                gc.collect()
                
            except Exception as e:
                print(f"  ⚠️ {img_path}: {e}")
                continue
    
    return original_count + augmented_count * 3


def main():
    print("=" * 60)
    print("🎨 背景增强数据生成")
    print("=" * 60)
    
    if not REMBG_AVAILABLE:
        print("\n⚠️ rembg 未安装，使用简单背景移除")
        print("   安装rembg以获得更好的效果:")
        print("   pip install rembg")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 处理所有角色
    total_images = 0
    chars = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    
    print(f"\n处理 {len(chars)} 个角色...")
    
    for i, char_dir in enumerate(chars):
        count = process_character(char_dir, OUTPUT_DIR, num_augmented=5)
        total_images += count
        print(f"  [{i+1}/{len(chars)}] {char_dir.name}: +{count} 张")
        gc.collect()
    
    print(f"\n✅ 完成！共生成 {total_images} 张增强图片")
    print(f"📁 输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()