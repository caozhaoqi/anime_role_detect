#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于模型的数据清洗脚本 - 使用WD Vit Tagger v3模型分析图像内容生成标签
"""

import os
import hashlib
import json
import argparse
from PIL import Image
from tqdm import tqdm

# 配置参数
MIN_WIDTH = 100
MIN_HEIGHT = 100
MIN_FILE_SIZE_KB = 5
TARGET_SIZE = (512, 512)

# 需要过滤的不当标签（包括骨钉等不适内容）
FILTERED_TAGS = {
    'bone', 'bone nail', 'bone nails', 'skeleton', 'skull',
    'gore', 'blood', 'violence', 'death',
    'nsfw', 'nudity', 'explicit', 'porn',
    'offensive', 'hateful', 'racist', 'sexist',
    'self-harm', 'suicide', 'depression',
    'drug', 'alcohol', 'smoking', 'cigarette',
    'weapon', 'gun', 'knife', 'sword', 'explosion'
}

# 角色基础标签（用于补充模型标签）
CHARACTER_FEATURES = {
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


def get_image_hash(img_path):
    """计算图片的MD5哈希值"""
    try:
        with open(img_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def is_low_quality(img_path):
    """快速检查图片是否为低质量"""
    try:
        file_size_kb = os.path.getsize(img_path) / 1024
        if file_size_kb < MIN_FILE_SIZE_KB:
            return True, f"文件过小 ({file_size_kb:.1f}KB)"
        
        with Image.open(img_path) as img:
            width, height = img.size
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return True, f"尺寸过小 ({width}x{height})"
            
            if img.format not in ['JPEG', 'PNG', 'WEBP']:
                return True, f"格式不支持 ({img.format})"
            
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 5:
                return True, f"宽高比异常"
        
        return False, ""
    except Exception as e:
        return True, f"无法读取: {str(e)}"


def resize_and_convert(img_path, output_path):
    """调整图片尺寸并转换格式"""
    try:
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            new_img = Image.new('RGB', TARGET_SIZE, (255, 255, 255))
            x = (TARGET_SIZE[0] - img.size[0]) // 2
            y = (TARGET_SIZE[1] - img.size[1]) // 2
            new_img.paste(img, (x, y))
            
            new_img.save(output_path, 'JPEG', quality=95)
        return True
    except Exception:
        return False


def load_tagger_model():
    """加载WD Vit Tagger模型"""
    try:
        from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
        tagger = WDViTV3Tagger()
        tagger.load_model()
        return tagger
    except Exception as e:
        print(f"⚠️ 加载标签模型失败，将使用简单标签生成: {e}")
        return None


def filter_inappropriate_tags(tags):
    """过滤不当标签"""
    filtered = []
    for tag_info in tags:
        tag = tag_info['tag'].lower()
        # 检查是否包含不当内容
        is_inappropriate = False
        for forbidden in FILTERED_TAGS:
            if forbidden in tag:
                is_inappropriate = True
                break
        
        if not is_inappropriate:
            filtered.append(tag_info)
    
    return filtered


def generate_tags_with_model(tagger, img_path, role_name):
    """使用模型生成标签"""
    tags = []
    
    if tagger:
        try:
            model_tags = tagger.generate_tags(img_path, threshold=0.2)
            model_tags = filter_inappropriate_tags(model_tags)
            tags.extend([t['tag'] for t in model_tags[:15]])
        except Exception as e:
            print(f"⚠️ 模型生成标签失败: {e}")
    
    # 添加角色特征标签（补充）
    if role_name in CHARACTER_FEATURES:
        tags.extend(CHARACTER_FEATURES[role_name])
    
    # 添加默认标签
    tags.extend(CHARACTER_FEATURES['Default'])
    
    # 去重并返回
    return list(set(tags))


def main():
    parser = argparse.ArgumentParser(description='基于模型的数据清洗')
    parser.add_argument('--data-dir', type=str, default='./data/merged_dataset')
    parser.add_argument('--output-dir', type=str, default='./data_cleaned')
    parser.add_argument('--report-file', type=str, default='cleaning_report.json')
    args = parser.parse_args()
    
    print("🚀 开始基于模型的数据清洗")
    
    # 加载标签模型
    print("📦 加载图像标签模型...")
    tagger = load_tagger_model()
    if tagger and tagger.wd_model:
        print("✅ 模型加载成功，将使用WD Vit Tagger进行图像分析")
    else:
        print("⚠️ 模型加载失败，将使用简单标签生成")
    
    report = {}
    
    # 步骤1: 删除重复图片
    print("\n📦 步骤1: 删除重复图片")
    hashes = {}
    duplicates = []
    for root, dirs, files in os.walk(args.data_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg')):
                path = os.path.join(root, f)
                h = get_image_hash(path)
                if h:
                    if h in hashes:
                        duplicates.append(path)
                    else:
                        hashes[h] = path
    
    for p in duplicates:
        os.remove(p)
    report['step1'] = {'duplicates_removed': len(duplicates), 'unique_images': len(hashes)}
    print(f"  ✅ 删除 {len(duplicates)} 张重复图片")
    
    # 步骤2: 过滤低质量图片
    print("\n🔍 步骤2: 过滤低质量图片")
    low_quality = []
    for root, dirs, files in os.walk(args.data_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg')):
                path = os.path.join(root, f)
                is_bad, _ = is_low_quality(path)
                if is_bad:
                    low_quality.append(path)
    
    for p in low_quality:
        os.remove(p)
    report['step2'] = {'low_quality_removed': len(low_quality)}
    print(f"  ✅ 删除 {len(low_quality)} 张低质量图片")
    
    # 步骤3: 标准化和使用模型生成标签
    print("\n📐 步骤3: 标准化图片和生成标签")
    os.makedirs(args.output_dir, exist_ok=True)
    all_tags = {}
    processed = 0
    
    for role_name in tqdm(os.listdir(args.data_dir), desc="处理角色"):
        role_dir = os.path.join(args.data_dir, role_name)
        if not os.path.isdir(role_dir):
            continue
        
        out_role_dir = os.path.join(args.output_dir, role_name)
        os.makedirs(out_role_dir, exist_ok=True)
        role_tags = {}
        
        for f in os.listdir(role_dir):
            if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg')):
                in_path = os.path.join(role_dir, f)
                out_path = os.path.join(out_role_dir, os.path.splitext(f)[0] + '.jpg')
                
                if resize_and_convert(in_path, out_path):
                    # 使用模型生成标签
                    tags = generate_tags_with_model(tagger, out_path, role_name)
                    role_tags[os.path.basename(out_path)] = tags
                    processed += 1
        
        all_tags[role_name] = role_tags
    
    # 保存标签文件
    tags_file = os.path.join(args.output_dir, 'image_tags.json')
    with open(tags_file, 'w', encoding='utf-8') as f:
        json.dump(all_tags, f, ensure_ascii=False, indent=2)
    
    report['step3'] = {
        'processed_images': processed, 
        'output_dir': args.output_dir,
        'tags_file': tags_file,
        'model_used': tagger is not None and tagger.wd_model is not None
    }
    print(f"  ✅ 处理 {processed} 张图片")
    print(f"  🏷️ 标签已保存到: {tags_file}")
    
    # 步骤4: 数据均衡性检查
    print("\n⚖️ 步骤4: 数据均衡性检查")
    role_counts = {}
    for role_name in os.listdir(args.output_dir):
        role_dir = os.path.join(args.output_dir, role_name)
        if os.path.isdir(role_dir):
            cnt = len([f for f in os.listdir(role_dir) if f.endswith('.jpg')])
            role_counts[role_name] = cnt
    
    report['step4'] = {
        'total_roles': len(role_counts),
        'min_images': min(role_counts.values()),
        'max_images': max(role_counts.values()),
        'avg_images': sum(role_counts.values()) // len(role_counts),
        'role_details': role_counts
    }
    print(f"  📊 {len(role_counts)} 个角色, 平均 {sum(role_counts.values()) // len(role_counts)} 张/角色")
    
    # 保存报告
    with open(args.report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 清洗完成! 报告: {args.report_file}, 输出: {args.output_dir}")
    
    # 打印标签统计
    all_tag_counts = {}
    for role_tags in all_tags.values():
        for tags in role_tags.values():
            for tag in tags:
                all_tag_counts[tag] = all_tag_counts.get(tag, 0) + 1
    
    print("\n📈 标签统计（前20个高频标签）:")
    for tag, count in sorted(all_tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {tag}: {count} 次")


if __name__ == '__main__':
    main()
