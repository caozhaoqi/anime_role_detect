#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
剔除冷门角色脚本
"""

import os
import shutil

# 需要剔除的冷门角色
COLD_ROLES = [
    'luo4ke4ke4',   # 洛可可
    'ke4luo2luo2',  # 克萝萝
    'xiao3shan3',   # 小闪
    'ai4li4er3',    # 爱丽儿
    'fu2li4xi1ya4', # 芙丽希娅
    'qi2ta3'        # 奇塔
]

DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
ROLE_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'

def remove_cold_roles(data_dir, cold_roles):
    """删除冷门角色文件夹"""
    removed_count = 0
    removed_roles = []
    
    print("=" * 70)
    print("🗑️ 开始剔除冷门角色")
    print("=" * 70)
    
    for role in cold_roles:
        role_path = os.path.join(data_dir, role)
        
        if os.path.exists(role_path) and os.path.isdir(role_path):
            # 统计该角色的图片数量
            img_count = len([f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])
            removed_roles.append((role, img_count))
            
            shutil.rmtree(role_path)
            removed_count += 1
            print(f"已删除: {role} ({img_count} 张图片)")
        else:
            print(f"跳过: {role} (目录不存在)")
    
    print("\n" + "=" * 70)
    print(f"已成功删除 {removed_count} 个冷门角色")
    print("=" * 70)
    
    return removed_roles

def update_role_file(role_file, cold_roles):
    """更新角色名单文件，移除冷门角色"""
    print("\n" + "=" * 70)
    print("📝 更新角色名单文件")
    print("=" * 70)
    
    # 创建拼音到中文名的映射
    pinyin_to_name = {
        'luo4ke4ke4': '洛可可',
        'ke4luo2luo2': '克萝萝',
        'xiao3shan3': '小闪',
        'ai4li4er3': '爱丽儿',
        'fu2li4xi1ya4': '芙丽希娅',
        'qi2ta3': '奇塔'
    }
    
    # 读取原始文件
    with open(role_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 筛选出非冷门角色的行
    new_lines = []
    removed_names = []
    
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 1:
                name = parts[0]
                # 检查是否为冷门角色
                is_cold = False
                for pinyin, chinese_name in pinyin_to_name.items():
                    if name == chinese_name:
                        is_cold = True
                        removed_names.append(name)
                        break
                if not is_cold:
                    new_lines.append(line)
    
    # 写入更新后的文件
    with open(role_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines) + '\n')
    
    print(f"已从名单中移除 {len(removed_names)} 个角色: {', '.join(removed_names)}")
    print(f"更新后名单共 {len(new_lines)} 个角色")
    print("=" * 70)

def final_statistics(data_dir):
    """输出清理后的数据集统计"""
    print("\n" + "=" * 70)
    print("📊 清理后的数据集统计")
    print("=" * 70)
    
    roles = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')])
    
    total_images = 0
    under_50_count = 0
    under_50_roles = []
    
    for role in roles:
        role_path = os.path.join(data_dir, role)
        img_count = len([f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])
        total_images += img_count
        
        if img_count < 50:
            under_50_count += 1
            under_50_roles.append((role, img_count))
    
    print(f"角色总数: {len(roles)}")
    print(f"图片总数: {total_images}")
    print(f"平均每角色: {total_images / len(roles):.2f} 张")
    
    if under_50_count > 0:
        print(f"\n⚠️ 图片不足50张的角色 ({under_50_count}个):")
        for role, count in under_50_roles:
            print(f"  - {role}: {count} 张")
    else:
        print("\n🎉 所有角色图片数均≥50张！")
    
    print("=" * 70)

def main():
    print("🚀 开始剔除冷门角色...")
    
    # 删除冷门角色文件夹
    removed = remove_cold_roles(DATA_DIR, COLD_ROLES)
    
    # 更新角色名单文件
    update_role_file(ROLE_FILE, COLD_ROLES)
    
    # 输出统计结果
    final_statistics(DATA_DIR)
    
    print("\n✅ 冷门角色剔除完成！")

if __name__ == '__main__':
    main()
