#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整数据清洗脚本 - 执行去重、过滤低质量、标准化、标签标注等全部步骤
"""

import os
import hashlib
import json
import argparse
from PIL import Image
from tqdm import tqdm
import shutil

# 配置参数
MIN_WIDTH = 100
MIN_HEIGHT = 100
MIN_FILE_SIZE_KB = 5
TARGET_SIZE = (512, 512)

# 预定义角色特征标签
CHARACTER_FEATURES = {
    'Tsukiyo': ['blue hair', 'long hair', 'blue eyes', 'school uniform', 'serafuku', 'calm'],
    'Hina': ['pink hair', 'long hair', 'pink eyes', 'school uniform', 'gentle', 'smile'],
    'Madoka': ['pink hair', 'twintails', 'pink eyes', 'magical girl', 'pink dress'],
    'Homura': ['black hair', 'long hair', 'purple eyes', 'school uniform', 'serious'],
    'Sayaka': ['blue hair', 'ponytail', 'blue eyes', 'magical girl', 'sword'],
    'Mami': ['blonde hair', 'twin drills', 'yellow eyes', 'magical girl', 'rifle'],
    'Kyoko': ['red hair', 'ponytail', 'orange eyes', 'magical girl', 'spear'],
    'Arona': ['blue hair', 'short hair', 'blue eyes', 'school uniform', 'robot', 'halo'],
    'Shiroko': ['white hair', 'short hair', 'blue eyes', 'school uniform', 'gun'],
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
    """检查图片是否为低质量"""
    try:
        # 检查文件大小
        file_size_kb = os.path.getsize(img_path) / 1024
        if file_size_kb < MIN_FILE_SIZE_KB:
            return True, f"文件过小 ({file_size_kb:.1f}KB)"
        
        with Image.open(img_path) as img:
            width, height = img.size
            
            # 检查尺寸
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return True, f"尺寸过小 ({width}x{height})"
            
            # 检查是否为损坏图片
            if img.format not in ['JPEG', 'PNG', 'WEBP']:
                return True, f"格式不支持 ({img.format})"
            
            # 检查是否为纯色/低多样性图片
            if img.mode == 'RGB':
                pixels = list(img.getdata())
                unique_colors = len(set(pixels))
                if unique_colors <= 10:
                    return True, f"纯色/低多样性图片 ({unique_colors}种颜色)"
            
            # 检查宽高比是否异常
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 5:
                return True, f"宽高比异常 ({width}x{height})"
        
        return False, ""
    except Exception as e:
        return True, f"无法读取: {str(e)}"


def resize_and_convert(img_path, output_path, target_size=TARGET_SIZE):
    """调整图片尺寸并转换格式为JPEG"""
    try:
        with Image.open(img_path) as img:
            # 转换为RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 按比例调整尺寸
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # 创建背景画布
            new_img = Image.new('RGB', target_size, (255, 255, 255))
            
            # 居中放置
            x = (target_size[0] - img.size[0]) // 2
            y = (target_size[1] - img.size[1]) // 2
            new_img.paste(img, (x, y))
            
            # 保存为JPEG
            new_img.save(output_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"❌ 处理图片失败 {img_path}: {e}")
        return False


def generate_tags(img_path, role_name):
    """生成图片的内容标签"""
    tags = []
    
    # 根据角色名称添加特征标签
    if role_name in CHARACTER_FEATURES:
        tags.extend(CHARACTER_FEATURES[role_name])
    else:
        tags.extend(CHARACTER_FEATURES['Default'])
    
    # 添加尺寸标签
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            if width >= 512 and height >= 512:
                tags.append('high resolution')
            elif width >= 256 and height >= 256:
                tags.append('medium resolution')
            else:
                tags.append('low resolution')
    except:
        pass
    
    return list(set(tags))


def step1_remove_duplicates(data_dir, report):
    """步骤1: 删除重复图片"""
    print("\n" + "=" * 60)
    print("📦 步骤1: 删除重复图片")
    print("=" * 60)
    
    hashes = {}
    duplicates = []
    
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if not filename.lower().endswith(('.jpg', '.png', '.webp', '.jpeg')):
                continue
            
            img_path = os.path.join(root, filename)
            img_hash = get_image_hash(img_path)
            
            if img_hash:
                if img_hash in hashes:
                    duplicates.append(img_path)
                else:
                    hashes[img_hash] = img_path
    
    # 删除重复图片
    for dup in tqdm(duplicates, desc="删除重复图片"):
        os.remove(dup)
    
    report['step1'] = {
        'duplicates_found': len(duplicates),
        'duplicates_removed': len(duplicates),
        'unique_images_after': len(hashes)
    }
    
    print(f"✅ 已删除 {len(duplicates)} 张重复图片")
    print(f"📊 剩余唯一图片: {len(hashes)} 张")


def step2_filter_low_quality(data_dir, report):
    """步骤2: 过滤低质量图片"""
    print("\n" + "=" * 60)
    print("🔍 步骤2: 过滤低质量图片")
    print("=" * 60)
    
    low_quality_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if not filename.lower().endswith(('.jpg', '.png', '.webp', '.jpeg')):
                continue
            
            img_path = os.path.join(root, filename)
            is_bad, reason = is_low_quality(img_path)
            
            if is_bad:
                low_quality_files.append((img_path, reason))
    
    # 删除低质量图片
    for img_path, reason in tqdm(low_quality_files, desc="删除低质量图片"):
        os.remove(img_path)
    
    report['step2'] = {
        'low_quality_found': len(low_quality_files),
        'low_quality_removed': len(low_quality_files),
        'details': [{'path': p, 'reason': r} for p, r in low_quality_files[:20]]
    }
    
    print(f"✅ 已删除 {len(low_quality_files)} 张低质量图片")


def step3_balance_data(data_dir, report):
    """步骤3: 检查数据均衡性"""
    print("\n" + "=" * 60)
    print("⚖️ 步骤3: 数据均衡性检查")
    print("=" * 60)
    
    role_stats = {}
    for role_name in os.listdir(data_dir):
        role_dir = os.path.join(data_dir, role_name)
        if not os.path.isdir(role_dir):
            continue
        
        img_count = len([f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg'))])
        role_stats[role_name] = img_count
    
    if role_stats:
        min_count = min(role_stats.values())
        max_count = max(role_stats.values())
        avg_count = sum(role_stats.values()) // len(role_stats)
        
        # 找出数据不足的角色（低于平均值的50%）
        roles_to_supplement = [r for r, cnt in role_stats.items() if cnt < avg_count * 0.5]
        
        report['step3'] = {
            'total_roles': len(role_stats),
            'min_images': min_count,
            'max_images': max_count,
            'avg_images': avg_count,
            'roles_to_supplement': roles_to_supplement,
            'role_details': role_stats
        }
        
        print(f"📊 角色总数: {len(role_stats)}")
        print(f"📈 最多图片: {max_count} 张")
        print(f"📉 最少图片: {min_count} 张")
        print(f"⚖️ 平均图片: {avg_count} 张")
        
        if roles_to_supplement:
            print(f"⚠️ 需要补充数据的角色 ({len(roles_to_supplement)}个):")
            for r in roles_to_supplement:
                print(f"  - {r}: {role_stats[r]} 张")
        else:
            print("✅ 数据分布均衡")


def step4_standardize(data_dir, output_dir, report):
    """步骤4: 标准化图片尺寸和格式"""
    print("\n" + "=" * 60)
    print("📐 步骤4: 标准化图片尺寸和格式")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    failed_count = 0
    
    for role_name in tqdm(os.listdir(data_dir), desc="处理角色"):
        role_dir = os.path.join(data_dir, role_name)
        if not os.path.isdir(role_dir):
            continue
        
        # 创建角色输出目录
        output_role_dir = os.path.join(output_dir, role_name)
        os.makedirs(output_role_dir, exist_ok=True)
        
        for filename in os.listdir(role_dir):
            if not filename.lower().endswith(('.jpg', '.png', '.webp', '.jpeg')):
                continue
            
            img_path = os.path.join(role_dir, filename)
            # 转换为JPEG格式
            output_filename = os.path.splitext(filename)[0] + '.jpg'
            output_path = os.path.join(output_role_dir, output_filename)
            
            if resize_and_convert(img_path, output_path):
                processed_count += 1
            else:
                failed_count += 1
    
    report['step4'] = {
        'output_dir': output_dir,
        'processed_images': processed_count,
        'failed_images': failed_count,
        'target_size': TARGET_SIZE,
        'target_format': 'JPEG'
    }
    
    print(f"✅ 已处理 {processed_count} 张图片")
    print(f"❌ 失败 {failed_count} 张")
    print(f"📁 输出目录: {output_dir}")


def step5_tagging(data_dir, report):
    """步骤5: 标签标注"""
    print("\n" + "=" * 60)
    print("🏷️ 步骤5: 标签标注")
    print("=" * 60)
    
    all_tags = {}
    
    for role_name in tqdm(os.listdir(data_dir), desc="标注角色"):
        role_dir = os.path.join(data_dir, role_name)
        if not os.path.isdir(role_dir):
            continue
        
        role_tags = {}
        for filename in os.listdir(role_dir):
            if not filename.lower().endswith(('.jpg', '.png', '.webp', '.jpeg')):
                continue
            
            img_path = os.path.join(role_dir, filename)
            tags = generate_tags(img_path, role_name)
            role_tags[filename] = tags
        
        all_tags[role_name] = role_tags
    
    # 保存标签文件
    tags_file = os.path.join(data_dir, 'image_tags.json')
    with open(tags_file, 'w', encoding='utf-8') as f:
        json.dump(all_tags, f, ensure_ascii=False, indent=2)
    
    # 统计标签
    tag_stats = {}
    for role_tags in all_tags.values():
        for tags in role_tags.values():
            for tag in tags:
                tag_stats[tag] = tag_stats.get(tag, 0) + 1
    
    report['step5'] = {
        'tags_file': tags_file,
        'total_images_tagged': sum(len(t) for t in all_tags.values()),
        'unique_tags': len(tag_stats),
        'top_tags': sorted(tag_stats.items(), key=lambda x: x[1], reverse=True)[:10]
    }
    
    print(f"✅ 已标注 {sum(len(t) for t in all_tags.values())} 张图片")
    print(f"🏷️ 共 {len(tag_stats)} 种标签")
    print(f"📄 标签文件: {tags_file}")


def main():
    parser = argparse.ArgumentParser(description='完整数据清洗流程')
    parser.add_argument('--data-dir', type=str, default='./data', help='输入数据集目录')
    parser.add_argument('--output-dir', type=str, default='./data_cleaned', help='清洗后输出目录')
    parser.add_argument('--report-file', type=str, default='cleaning_report.json', help='清洗报告路径')
    args = parser.parse_args()
    
    print("🚀 开始完整数据清洗流程")
    print("=" * 60)
    
    report = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'steps': []
    }
    
    # 步骤1: 删除重复图片
    step1_remove_duplicates(args.data_dir, report)
    
    # 步骤2: 过滤低质量图片
    step2_filter_low_quality(args.data_dir, report)
    
    # 步骤3: 检查数据均衡性
    step3_balance_data(args.data_dir, report)
    
    # 步骤4: 标准化图片尺寸和格式
    step4_standardize(args.data_dir, args.output_dir, report)
    
    # 步骤5: 标签标注
    step5_tagging(args.output_dir, report)
    
    # 保存完整报告
    with open(args.report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("🎉 数据清洗完成!")
    print("=" * 60)
    print(f"📄 完整报告已保存至: {args.report_file}")
    print(f"📁 清洗后数据目录: {args.output_dir}")


if __name__ == '__main__':
    main()
