#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标签清洗脚本 - 过滤不当标签，确保标签内容安全
"""

import json
import argparse

# 需要过滤的不当标签
FILTERED_TAGS = {
    'bone', 'bone nail', 'bone nails', 'skeleton', 'skull',
    'gore', 'blood', 'violence', 'death',
    'nsfw', 'nudity', 'explicit', 'porn',
    'offensive', 'hateful', 'racist', 'sexist',
    'self-harm', 'suicide', 'depression',
    'drug', 'alcohol', 'smoking', 'cigarette',
    'weapon', 'gun', 'knife', 'sword', 'explosion',
    'rifle', 'spear'
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


def is_inappropriate(tag):
    """检查标签是否不当"""
    tag_lower = tag.lower()
    for forbidden in FILTERED_TAGS:
        if forbidden in tag_lower:
            return True
    return False


def clean_tags(input_file, output_file):
    """清洗标签文件"""
    print(f"📦 加载标签文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_data = {}
    removed_tags = []
    
    for role_name, images in data.items():
        cleaned_images = {}
        for img_name, tags in images.items():
            # 过滤不当标签
            cleaned_tags = []
            for tag in tags:
                if is_inappropriate(tag):
                    removed_tags.append({'role': role_name, 'image': img_name, 'tag': tag})
                else:
                    cleaned_tags.append(tag)
            
            # 补充安全的角色特征标签
            if role_name in SAFE_CHARACTER_FEATURES:
                for safe_tag in SAFE_CHARACTER_FEATURES[role_name]:
                    if safe_tag not in cleaned_tags:
                        cleaned_tags.append(safe_tag)
            
            cleaned_images[img_name] = cleaned_tags
        
        cleaned_data[role_name] = cleaned_images
    
    # 保存清洗后的标签
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 标签清洗完成!")
    print(f"   输出文件: {output_file}")
    print(f"   移除不当标签: {len(removed_tags)} 个")
    
    if removed_tags:
        print("\n🔍 移除的不当标签:")
        for item in removed_tags[:10]:
            print(f"   - {item['role']}/{item['image']}: {item['tag']}")
        if len(removed_tags) > 10:
            print(f"   ... 还有 {len(removed_tags) - 10} 个")
    
    # 统计最终标签
    all_tags = set()
    for images in cleaned_data.values():
        for tags in images.values():
            all_tags.update(tags)
    
    print(f"\n📊 最终标签统计:")
    print(f"   标签种类: {len(all_tags)}")
    print(f"   所有标签: {', '.join(sorted(all_tags))}")


def main():
    parser = argparse.ArgumentParser(description='清洗标签文件，移除不当内容')
    parser.add_argument('--input', type=str, default='data_cleaned/image_tags.json', help='输入标签文件')
    parser.add_argument('--output', type=str, default='data_cleaned/image_tags_cleaned.json', help='输出标签文件')
    args = parser.parse_args()
    
    clean_tags(args.input, args.output)


if __name__ == '__main__':
    main()
