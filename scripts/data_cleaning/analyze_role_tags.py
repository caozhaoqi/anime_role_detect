#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗分析脚本 - 提取动漫图片内容标签，分析角色公共标签
用于帮助进行角色图片分类的数据清洗工作
"""

import os
import json
import hashlib
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import argparse

# 配置参数
MIN_WIDTH = 100
MIN_HEIGHT = 100
MIN_FILE_SIZE_KB = 5

# 预定义标签分类体系
TAG_CATEGORIES = {
    'character_count': ['1girl', '2girls', '3girls', '4+girls', '1boy', '2boys', '3boys', '4+boys', 'solo', 'group'],
    'hair_length': ['long hair', 'medium hair', 'short hair', 'very short hair'],
    'hair_style': ['twintails', 'ponytail', 'bun', 'braids', 'single braid', 'ahoge'],
    'hair_color': ['blue hair', 'blonde hair', 'black hair', 'red hair', 'green hair', 'purple hair', 
                   'pink hair', 'brown hair', 'grey hair', 'yellow hair', 'white hair', 'silver hair',
                   'aqua hair', 'orange hair', 'multicolored hair', 'gradient hair', 'streaked hair'],
    'eye_color': ['blue eyes', 'red eyes', 'green eyes', 'purple eyes', 'brown eyes', 'yellow eyes',
                  'pink eyes', 'grey eyes', 'black eyes', 'white eyes', 'aqua eyes', 'orange eyes',
                  'multicolored eyes', 'heterochromia'],
    'expression': ['smile', 'laugh', 'sad', 'angry', 'surprised', 'confused', 'happy', 'calm', 
                   'excited', 'tired', 'blush', 'sweat', 'tears', 'open mouth', 'tongue', 'grin',
                   'frown', 'pout', 'looking at viewer', 'looking away'],
    'accessories': ['cat ears', 'animal ears', 'horns', 'wings', 'tail', 'hat', 'cap', 'headband',
                    'bandana', 'helmet', 'glasses', 'sunglasses', 'mask', 'headphones', 'earphones',
                    'ribbon', 'bow', 'flower'],
    'clothing': ['dress', 'skirt', 'pants', 'shorts', 'jacket', 'sweater', 'hoodie', 't-shirt',
                 'blouse', 'coat', 'swimsuit', 'uniform', 'costume', 'maid outfit', 'nurse outfit',
                 'school uniform', 'gym uniform', 'sailor uniform', 'military uniform'],
    'pose': ['standing', 'sitting', 'lying', 'kneeling', 'walking', 'running', 'jumping', 'dancing',
             'fighting', 'side view', 'front view', 'back view', 'close-up', 'medium shot',
             'full body', 'upper body'],
    'scene': ['outdoors', 'indoors', 'school', 'room', 'street', 'park', 'beach', 'mountain',
              'forest', 'city', 'night', 'day', 'sunset', 'sunrise', 'raining', 'snowing'],
    'quality': ['high quality', 'masterpiece', 'best quality', 'detailed', 'beautiful', 'cute',
                'sexy', 'cool', 'adorable', 'stylish']
}

# 角色特征关键词（根据常见动漫角色设定）
CHARACTER_FEATURES = {
    'Tsukiyo': ['blue hair', 'long hair', 'blue eyes', 'school uniform', 'serafuku', 'blue eyes', 'calm'],
    'Hina': ['pink hair', 'long hair', 'pink eyes', 'school uniform', 'gentle', 'smile'],
    'Madoka': ['pink hair', 'twintails', 'pink eyes', 'magical girl', 'pink dress'],
    'Homura': ['black hair', 'long hair', 'purple eyes', 'school uniform', 'serious', 'time manipulation'],
    'Sayaka': ['blue hair', 'ponytail', 'blue eyes', 'magical girl', 'sword'],
    'Mami': ['blonde hair', 'twin drills', 'yellow eyes', 'magical girl', 'rifle'],
    'Kyoko': ['red hair', 'ponytail', 'orange eyes', 'magical girl', 'spear'],
    'Arona': ['blue hair', 'short hair', 'blue eyes', 'school uniform', 'robot', 'halo'],
    'Shiroko': ['white hair', 'short hair', 'blue eyes', 'school uniform', 'gun']
}


def get_image_hash(img_path):
    """计算图片的MD5哈希值用于去重"""
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


def extract_image_features(img_path):
    """提取图片的基本特征"""
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            mode = img.mode
            format = img.format
            file_size_kb = os.path.getsize(img_path) / 1024
            
            return {
                'width': width,
                'height': height,
                'mode': mode,
                'format': format,
                'file_size_kb': round(file_size_kb, 2),
                'aspect_ratio': round(max(width, height) / min(width, height), 2)
            }
    except Exception as e:
        return {'error': str(e)}


def generate_content_tags(img_path, role_name):
    """
    生成图片的内容标签
    结合图像分析和角色特征知识
    """
    tags = []
    features = extract_image_features(img_path)
    
    if 'error' in features:
        return tags
    
    # 添加尺寸相关标签
    width, height = features['width'], features['height']
    if width >= 512 and height >= 512:
        tags.append('high resolution')
    elif width >= 256 and height >= 256:
        tags.append('medium resolution')
    else:
        tags.append('low resolution')
    
    # 添加宽高比标签
    ar = features['aspect_ratio']
    if ar < 1.2:
        tags.append('square')
    elif ar < 1.6:
        tags.append('portrait')
    elif ar < 2.0:
        tags.append('wide')
    else:
        tags.append('panoramic')
    
    # 根据角色名称添加特征标签
    if role_name in CHARACTER_FEATURES:
        tags.extend(CHARACTER_FEATURES[role_name])
    
    # 添加质量标签
    if features['file_size_kb'] >= 50:
        tags.append('high quality')
    elif features['file_size_kb'] >= 20:
        tags.append('medium quality')
    else:
        tags.append('low quality')
    
    # 去重并返回
    return list(set(tags))


def analyze_role_dataset(data_dir):
    """分析角色数据集，提取标签信息"""
    role_data = {}
    duplicate_hashes = {}
    low_quality_images = []
    all_tags = defaultdict(int)
    role_tags = defaultdict(lambda: defaultdict(int))
    
    print("=" * 80)
    print("🔍 开始分析角色数据集")
    print(f"📁 数据集路径: {data_dir}")
    print("=" * 80)
    
    for role_name in tqdm(os.listdir(data_dir), desc="处理角色"):
        role_dir = os.path.join(data_dir, role_name)
        if not os.path.isdir(role_dir):
            continue
        
        role_info = {
            'total_images': 0,
            'valid_images': 0,
            'duplicate_images': 0,
            'low_quality_images': 0,
            'images': [],
            'top_tags': [],
            'feature_summary': {
                'avg_width': 0,
                'avg_height': 0,
                'avg_file_size_kb': 0,
                'formats': defaultdict(int)
            }
        }
        
        total_width = 0
        total_height = 0
        total_size = 0
        
        for img_name in os.listdir(role_dir):
            if not img_name.lower().endswith(('.jpg', '.png', '.webp', '.jpeg')):
                continue
            
            img_path = os.path.join(role_dir, img_name)
            role_info['total_images'] += 1
            
            # 检查低质量
            is_bad, reason = is_low_quality(img_path)
            if is_bad:
                role_info['low_quality_images'] += 1
                low_quality_images.append({
                    'role': role_name,
                    'path': img_path,
                    'reason': reason
                })
                continue
            
            # 检查重复
            img_hash = get_image_hash(img_path)
            if img_hash:
                if img_hash in duplicate_hashes:
                    role_info['duplicate_images'] += 1
                    continue
                duplicate_hashes[img_hash] = img_path
            
            # 提取特征和标签
            features = extract_image_features(img_path)
            tags = generate_content_tags(img_path, role_name)
            
            # 更新统计
            role_info['valid_images'] += 1
            total_width += features['width']
            total_height += features['height']
            total_size += features['file_size_kb']
            role_info['feature_summary']['formats'][features['format']] += 1
            
            # 统计标签
            for tag in tags:
                all_tags[tag] += 1
                role_tags[role_name][tag] += 1
            
            # 保存图片信息
            role_info['images'].append({
                'filename': img_name,
                'features': features,
                'tags': tags
            })
        
        # 计算平均值
        if role_info['valid_images'] > 0:
            role_info['feature_summary']['avg_width'] = round(total_width / role_info['valid_images'])
            role_info['feature_summary']['avg_height'] = round(total_height / role_info['valid_images'])
            role_info['feature_summary']['avg_file_size_kb'] = round(total_size / role_info['valid_images'], 2)
        
        # 获取Top标签
        role_info['top_tags'] = sorted(
            role_tags[role_name].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        role_data[role_name] = role_info
    
    return {
        'role_data': role_data,
        'all_tags': dict(sorted(all_tags.items(), key=lambda x: x[1], reverse=True)),
        'role_tags': {k: dict(v) for k, v in role_tags.items()},
        'duplicate_count': len(duplicate_hashes),
        'low_quality_images': low_quality_images,
        'total_roles': len(role_data),
        'total_images': sum(r['total_images'] for r in role_data.values()),
        'valid_images': sum(r['valid_images'] for r in role_data.values())
    }


def find_common_tags(role_tags, min_roles=2, min_count=5):
    """
    查找多个角色之间的公共标签
    :param role_tags: 各角色的标签统计
    :param min_roles: 至少在多少个角色中出现
    :param min_count: 每个角色中至少出现多少次
    """
    common_tags = defaultdict(lambda: {'roles': [], 'total_count': 0, 'role_counts': {}})
    
    for role_name, tags in role_tags.items():
        for tag, count in tags.items():
            if count >= min_count:
                common_tags[tag]['roles'].append(role_name)
                common_tags[tag]['total_count'] += count
                common_tags[tag]['role_counts'][role_name] = count
    
    # 过滤出满足条件的公共标签
    result = {}
    for tag, info in common_tags.items():
        if len(info['roles']) >= min_roles:
            result[tag] = info
    
    # 按角色数量和总计数排序
    return dict(sorted(result.items(), key=lambda x: (len(x[1]['roles']), x[1]['total_count']), reverse=True))


def find_role_unique_tags(role_tags, min_count=3):
    """
    查找每个角色的独特标签（其他角色中很少出现的标签）
    """
    # 计算每个标签在多少个角色中出现
    tag_role_count = defaultdict(int)
    for tags in role_tags.values():
        for tag in tags.keys():
            tag_role_count[tag] += 1
    
    unique_tags = {}
    for role_name, tags in role_tags.items():
        # 找出在其他角色中出现较少的标签
        role_unique = []
        for tag, count in tags.items():
            if count >= min_count and tag_role_count[tag] <= 2:
                role_unique.append((tag, count))
        
        # 按出现次数排序
        unique_tags[role_name] = sorted(role_unique, key=lambda x: x[1], reverse=True)[:5]
    
    return unique_tags


def generate_cleaning_report(analysis_result):
    """生成数据清洗报告"""
    report = {
        'summary': {
            'total_roles': analysis_result['total_roles'],
            'total_images': analysis_result['total_images'],
            'valid_images': analysis_result['valid_images'],
            'duplicate_count': analysis_result['duplicate_count'],
            'low_quality_count': len(analysis_result['low_quality_images']),
            'valid_rate': round(analysis_result['valid_images'] / analysis_result['total_images'] * 100, 2) if analysis_result['total_images'] > 0 else 0
        },
        'role_details': {},
        'common_tags': {},
        'unique_tags': {},
        'top_tags': list(analysis_result['all_tags'].keys())[:20],
        'cleaning_suggestions': []
    }
    
    # 添加角色详情
    for role_name, info in analysis_result['role_data'].items():
        report['role_details'][role_name] = {
            'total_images': info['total_images'],
            'valid_images': info['valid_images'],
            'duplicate_images': info['duplicate_images'],
            'low_quality_images': info['low_quality_images'],
            'avg_width': info['feature_summary']['avg_width'],
            'avg_height': info['feature_summary']['avg_height'],
            'avg_file_size_kb': info['feature_summary']['avg_file_size_kb'],
            'formats': dict(info['feature_summary']['formats']),
            'top_tags': [tag for tag, _ in info['top_tags']]
        }
    
    # 查找公共标签
    report['common_tags'] = find_common_tags(analysis_result['role_tags'])
    
    # 查找独特标签
    report['unique_tags'] = find_role_unique_tags(analysis_result['role_tags'])
    
    # 生成清洗建议
    report['cleaning_suggestions'] = generate_cleaning_suggestions(analysis_result)
    
    return report


def generate_cleaning_suggestions(analysis_result):
    """根据分析结果生成清洗建议"""
    suggestions = []
    
    # 检查重复图片
    if analysis_result['duplicate_count'] > 0:
        suggestions.append({
            'priority': 'high',
            'action': '删除重复图片',
            'description': f"发现 {analysis_result['duplicate_count']} 张重复图片，建议删除以减少冗余",
            'affected_roles': list(analysis_result['role_data'].keys())
        })
    
    # 检查低质量图片
    low_quality_count = len(analysis_result['low_quality_images'])
    if low_quality_count > 0:
        suggestions.append({
            'priority': 'high',
            'action': '删除低质量图片',
            'description': f"发现 {low_quality_count} 张低质量图片（尺寸过小、文件损坏、纯色等）",
            'details': analysis_result['low_quality_images'][:10]  # 只显示前10个示例
        })
    
    # 检查角色数据不均衡
    valid_counts = [(r, info['valid_images']) for r, info in analysis_result['role_data'].items()]
    valid_counts.sort(key=lambda x: x[1])
    
    min_count = valid_counts[0][1] if valid_counts else 0
    max_count = valid_counts[-1][1] if valid_counts else 0
    
    if max_count > 0 and min_count < max_count * 0.3:
        suggestions.append({
            'priority': 'medium',
            'action': '补充数据不足的角色',
            'description': f"数据分布不均衡，最少的角色只有 {min_count} 张有效图片，最多的有 {max_count} 张",
            'roles_to_supplement': [r for r, cnt in valid_counts if cnt < max_count * 0.3]
        })
    
    # 检查分辨率
    for role_name, info in analysis_result['role_data'].items():
        if info['valid_images'] > 0:
            avg_size = info['feature_summary']['avg_width'] * info['feature_summary']['avg_height']
            if avg_size < 128 * 128:
                suggestions.append({
                    'priority': 'medium',
                    'action': '补充高分辨率图片',
                    'description': f"角色 {role_name} 的平均分辨率较低，建议补充更高分辨率的图片"
                })
    
    return suggestions


def main():
    parser = argparse.ArgumentParser(description='分析动漫角色图片数据集，提取内容标签')
    parser.add_argument('--data-dir', type=str, default='./data', help='数据集目录')
    parser.add_argument('--output', type=str, default='./data_cleaning_report.json', help='输出报告路径')
    args = parser.parse_args()
    
    # 分析数据集
    analysis_result = analyze_role_dataset(args.data_dir)
    
    # 生成报告
    report = generate_cleaning_report(analysis_result)
    
    # 保存报告
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("📊 数据清洗报告摘要")
    print("=" * 80)
    print(f"🎭 角色总数: {report['summary']['total_roles']}")
    print(f"🖼️ 图片总数: {report['summary']['total_images']}")
    print(f"✅ 有效图片: {report['summary']['valid_images']}")
    print(f"🔄 重复图片: {report['summary']['duplicate_count']}")
    print(f"❌ 低质量图片: {report['summary']['low_quality_count']}")
    print(f"📈 有效率: {report['summary']['valid_rate']}%")
    print("\n🏷️ 前20个高频标签:")
    for i, tag in enumerate(report['top_tags'], 1):
        print(f"  {i}. {tag}")
    
    print("\n🔗 公共标签（跨角色共享）:")
    for tag, info in list(report['common_tags'].items())[:10]:
        print(f"  {tag}: 在 {len(info['roles'])} 个角色中出现")
    
    print("\n🎯 角色独特标签:")
    for role, tags in report['unique_tags'].items():
        if tags:
            print(f"  {role}: {[t[0] for t in tags]}")
    
    print("\n💡 清洗建议:")
    for i, suggestion in enumerate(report['cleaning_suggestions'], 1):
        priority = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[suggestion['priority']]
        print(f"  {priority} {suggestion['action']}: {suggestion['description']}")
    
    print("\n📄 完整报告已保存至: " + args.output)


if __name__ == '__main__':
    main()
