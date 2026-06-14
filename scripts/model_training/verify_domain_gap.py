#!/usr/bin/env python3
"""
验证 Domain Gap - 统计两个数据集的特征差异
"""
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

TRAIN_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
FINAL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")


def analyze_dataset(dataset_dir, max_samples=1000):
    """分析数据集特征"""
    stats = {
        'total_images': 0,
        'total_chars': 0,
        'resolutions': [],
        'aspect_ratios': [],
        'file_sizes': [],
        'modes': defaultdict(int),
    }
    
    chars = [d for d in dataset_dir.iterdir() if d.is_dir()]
    stats['total_chars'] = len(chars)
    
    sample_count = 0
    
    for char_dir in chars:
        if sample_count >= max_samples:
            break
            
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            for img_path in char_dir.glob(ext):
                if sample_count >= max_samples:
                    break
                    
                try:
                    # 文件大小
                    stats['file_sizes'].append(img_path.stat().st_size)
                    
                    # 图片信息
                    with Image.open(img_path) as img:
                        w, h = img.size
                        stats['resolutions'].append((w, h))
                        stats['aspect_ratios'].append(w / h)
                        stats['modes'][img.mode] += 1
                    
                    stats['total_images'] += 1
                    sample_count += 1
                    
                except Exception as e:
                    continue
            
            if sample_count >= max_samples:
                break
    
    return stats


def print_stats(name, stats):
    """打印统计结果"""
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print(f"{'='*60}")
    print(f"角色数: {stats['total_chars']}")
    print(f"图片数: {stats['total_images']}")
    
    if stats['resolutions']:
        resolutions = np.array(stats['resolutions'])
        avg_w, avg_h = resolutions.mean(axis=0)
        min_w, min_h = resolutions.min(axis=0)
        max_w, max_h = resolutions.max(axis=0)
        
        print(f"\n📐 分辨率:")
        print(f"  平均: {int(avg_w)} x {int(avg_h)}")
        print(f"  最小: {int(min_w)} x {int(min_h)}")
        print(f"  最大: {int(max_w)} x {int(max_h)}")
        
    if stats['aspect_ratios']:
        ratios = np.array(stats['aspect_ratios'])
        print(f"\n📏 宽高比:")
        print(f"  平均: {ratios.mean():.2f}")
        print(f"  最小: {ratios.min():.2f}")
        print(f"  最大: {ratios.max():.2f}")
        print(f"  中位数: {np.median(ratios):.2f}")
        
    if stats['file_sizes']:
        sizes = np.array(stats['file_sizes']) / 1024  # KB
        print(f"\n📦 文件大小 (KB):")
        print(f"  平均: {sizes.mean():.1f}")
        print(f"  最小: {sizes.min():.1f}")
        print(f"  最大: {sizes.max():.1f}")
        print(f"  中位数: {np.median(sizes):.1f}")
    
    if stats['modes']:
        print(f"\n🎨 图像模式:")
        for mode, count in stats['modes'].items():
            print(f"  {mode}: {count}")


def generate_sample_collage(train_dir, final_dir, output_path, samples_per_char=2):
    """生成对比拼图"""
    try:
        from PIL import ImageDraw, ImageFont
        
        # 获取共同角色
        train_chars = set(d.name for d in train_dir.iterdir() if d.is_dir())
        final_chars = set(d.name for d in final_dir.iterdir() if d.is_dir())
        common_chars = sorted(list(train_chars & final_chars))[:5]
        
        print(f"\n🎨 生成对比拼图 (5个共同角色)...")
        
        # 创建拼图
        collage_w, collage_h = 1200, 600
        collage = Image.new('RGB', (collage_w, collage_h), (255, 255, 255))
        draw = ImageDraw.Draw(collage)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 12)
        except:
            font = ImageFont.load_default()
        
        for i, char_name in enumerate(common_chars):
            # 获取训练集图片
            train_imgs = list((train_dir / char_name).glob('*.jpg'))[:samples_per_char]
            # 获取final集图片
            final_imgs = list((final_dir / char_name).glob('*.jpg'))[:samples_per_char]
            
            for j in range(samples_per_char):
                # 训练集位置
                x = i * 200 + 20
                y = j * 140 + 40
                
                if j < len(train_imgs):
                    img = Image.open(train_imgs[j]).resize((80, 100), Image.LANCZOS)
                    collage.paste(img, (x, y))
                    
                # final集位置
                x = i * 200 + 100
                y = j * 140 + 40
                
                if j < len(final_imgs):
                    img = Image.open(final_imgs[j]).resize((80, 100), Image.LANCZOS)
                    collage.paste(img, (x, y))
            
            # 角色名
            draw.text((i * 200 + 40, 20), char_name, font=font, fill=(0, 0, 0))
        
        # 标签
        draw.text((30, 320), "Training", font=font, fill=(0, 100, 0))
        draw.text((110, 320), "Final", font=font, fill=(100, 0, 0))
        
        collage.save(output_path)
        print(f"✅ 拼图已保存: {output_path}")
        
    except Exception as e:
        print(f"⚠️ 生成拼图失败: {e}")


def main():
    print("=" * 60)
    print("🔍 Domain Gap 验证")
    print("=" * 60)
    
    # 分析训练集
    print("\n正在分析 training_dataset...")
    train_stats = analyze_dataset(TRAIN_DIR, max_samples=500)
    
    # 分析final_dataset
    print("\n正在分析 final_dataset...")
    final_stats = analyze_dataset(FINAL_DIR, max_samples=500)
    
    # 打印结果
    print_stats("training_dataset", train_stats)
    print_stats("final_dataset", final_stats)
    
    # 对比分析
    print("\n" + "=" * 60)
    print("🔬 对比分析")
    print("=" * 60)
    
    train_avg_ratio = np.mean(train_stats['aspect_ratios']) if train_stats['aspect_ratios'] else 0
    final_avg_ratio = np.mean(final_stats['aspect_ratios']) if final_stats['aspect_ratios'] else 0
    
    train_avg_size = np.mean(train_stats['file_sizes']) / 1024 if train_stats['file_sizes'] else 0
    final_avg_size = np.mean(final_stats['file_sizes']) / 1024 if final_stats['file_sizes'] else 0
    
    print(f"\n宽高比差异: {abs(train_avg_ratio - final_avg_ratio):.2f}")
    print(f"文件大小差异: {abs(train_avg_size - final_avg_size):.1f} KB")
    
    # 生成拼图
    output_path = TRAIN_DIR.parent.parent / "logs" / "dataset_comparison.png"
    output_path.parent.mkdir(exist_ok=True)
    generate_sample_collage(TRAIN_DIR, FINAL_DIR, output_path)
    
    print("\n" + "=" * 60)
    print("✅ Domain Gap 验证完成")
    print("=" * 60)


if __name__ == "__main__":
    main()