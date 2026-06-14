#!/usr/bin/env python3
"""
训练时实时背景增强 - 不需要预处理
直接在 DataLoader 中进行背景模糊/替换
"""
import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageDraw
import random
import numpy as np


class BackgroundAugmentation:
    """训练时的背景增强"""
    
    def __init__(self, blur_prob=0.3, replace_prob=0.2):
        self.blur_prob = blur_prob
        self.replace_prob = replace_prob
        
        # 预设背景颜色
        self.bg_colors = [
            (255, 255, 255),  # 白色
            (128, 128, 128),  # 灰色
            (0, 0, 0),        # 黑色
            (200, 200, 220),  # 浅蓝灰
            (220, 200, 180),  # 米色
        ]
    
    def generate_blur_bg(self, img, radius=15):
        """生成模糊背景"""
        bg = img.filter(ImageFilter.GaussianBlur(radius))
        return bg
    
    def generate_solid_bg(self, img_size, color):
        """生成纯色背景"""
        return Image.new('RGB', img_size, color)
    
    def generate_gradient_bg(self, img_size):
        """生成渐变背景"""
        w, h = img_size
        color1 = tuple(random.randint(150, 255) for _ in range(3))
        color2 = tuple(random.randint(150, 255) for _ in range(3))
        
        bg = Image.new('RGB', img_size)
        draw = ImageDraw.Draw(bg)
        
        for y in range(h):
            ratio = y / h
            color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2))
            draw.line([(0, y), (w, y)], fill=color)
        
        return bg
    
    def __call__(self, img):
        """应用随机背景增强"""
        rand = random.random()
        img_size = img.size
        
        if rand < self.blur_prob:
            # 模糊背景
            radius = random.randint(10, 25)
            return self.generate_blur_bg(img, radius)
        
        elif rand < self.blur_prob + self.replace_prob:
            # 纯色/渐变背景
            if random.random() < 0.5:
                return self.generate_solid_bg(img_size, random.choice(self.bg_colors))
            else:
                return self.generate_gradient_bg(img_size)
        
        return img


class CharacterCentering:
    """角色居中 + 边缘填充"""
    
    def __init__(self, pad_value=128):
        self.pad_value = pad_value
    
    def center_crop(self, img, margin_ratio=0.15):
        """裁剪掉边缘空白"""
        w, h = img.size
        
        # 转换为数组找主体位置
        img_array = np.array(img.convert('RGB'))
        gray = np.mean(img_array, axis=2)
        
        # 找非边缘区域
        threshold = np.percentile(gray, 10)
        mask = gray < threshold
        
        # 找主体边界
        rows = np.any(~mask, axis=1)
        cols = np.any(~mask, axis=0)
        
        if not rows.any() or not cols.any():
            return img
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # 添加边距
        margin_h = int((rmax - rmin) * margin_ratio)
        margin_w = int((cmax - cmin) * margin_ratio)
        
        rmin = max(0, rmin - margin_h)
        rmax = min(h, rmax + margin_h)
        cmin = max(0, cmin - margin_w)
        cmax = min(w, cmax + margin_w)
        
        return img.crop((cmin, rmin, cmax, rmax))


# 测试
if __name__ == "__main__":
    from pathlib import Path
    
    test_dir = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
    
    # 找一张测试图片
    for char_dir in test_dir.iterdir():
        if char_dir.is_dir():
            imgs = list(char_dir.glob('*.jpg'))[:1]
            if imgs:
                test_img = Image.open(imgs[0])
                print(f"测试图片: {imgs[0]}")
                print(f"原始尺寸: {test_img.size}")
                
                # 测试背景增强
                aug = BackgroundAugmentation(blur_prob=0.5, replace_prob=0.3)
                
                # 保存不同增强版本
                aug(test_img).save(f"/tmp/aug_blur.jpg")
                aug(test_img).save(f"/tmp/aug_solid.jpg")
                
                print("✅ 测试完成，生成 /tmp/aug_*.jpg")
                break