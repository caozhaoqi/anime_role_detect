#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回滚检测结果到 combined_dataset
从 yunet_detection_results 和 multi_face_detected_anime 读取需要回滚的文件列表
"""

import os
import shutil

# 配置路径
BASE_DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data'
COMBINED_DATASET = os.path.join(BASE_DATA_DIR, 'combined_dataset')
YUNET_RESULTS = os.path.join(BASE_DATA_DIR, 'yunet_detection_results')
MULTI_FACE_DIR = os.path.join(BASE_DATA_DIR, 'multi_face_detected_anime')

# 需要搜索的可能位置
SEARCH_DIRS = [
    YUNET_RESULTS,
    MULTI_FACE_DIR,
    os.path.join(BASE_DATA_DIR, 'training_dataset'),
    BASE_DATA_DIR  # 根目录
]


def find_file(filename, search_dirs):
    """在多个目录中查找文件"""
    for search_dir in search_dirs:
        # 尝试直接路径
        full_path = os.path.join(search_dir, filename)
        if os.path.exists(full_path):
            return full_path
        
        # 尝试角色子目录
        parts = filename.split('/')
        if len(parts) == 2:
            role_dir, img_name = parts
            full_path = os.path.join(search_dir, role_dir, img_name)
            if os.path.exists(full_path):
                return full_path
        
        # 尝试下划线格式
        if '_' in filename:
            full_path = os.path.join(search_dir, filename)
            if os.path.exists(full_path):
                return full_path
    
    return None


def rollback_from_multi_role_list():
    """从 multi_role_list.txt 回滚"""
    multi_role_file = os.path.join(YUNET_RESULTS, 'multi_role_list.txt')
    if not os.path.exists(multi_role_file):
        print(f"❌ 未找到: {multi_role_file}")
        return 0
    
    count = 0
    with open(multi_role_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 解析格式: "2人 - Hina/Hina_0118.jpg"
            if ' - ' in line:
                _, filepath = line.split(' - ', 1)
                filepath = filepath.strip()
                
                # 获取角色名和图片名
                if '/' in filepath:
                    role_name, img_name = filepath.split('/')
                else:
                    # 处理没有斜杠的情况
                    role_name = filepath.split('_')[0]
                    img_name = filepath
                
                # 查找文件
                source_path = find_file(filepath, SEARCH_DIRS)
                if not source_path:
                    # 尝试其他格式
                    source_path = find_file(img_name, SEARCH_DIRS)
                
                if source_path:
                    # 创建目标目录
                    target_dir = os.path.join(COMBINED_DATASET, role_name)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    target_path = os.path.join(target_dir, img_name)
                    
                    # 如果目标已存在，跳过
                    if os.path.exists(target_path):
                        print(f"⚠️ 已存在，跳过: {target_path}")
                        continue
                    
                    # 移动文件
                    shutil.move(source_path, target_path)
                    print(f"✅ 回滚: {source_path} -> {target_path}")
                    count += 1
                else:
                    print(f"❌ 未找到文件: {filepath}")
    
    return count


def rollback_from_multi_face_list():
    """从 multi_face_images.txt 回滚"""
    multi_face_file = os.path.join(MULTI_FACE_DIR, 'multi_face_images.txt')
    if not os.path.exists(multi_face_file):
        print(f"❌ 未找到: {multi_face_file}")
        return 0
    
    count = 0
    with open(multi_face_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 解析格式: "15\t[彩色]\t/path/to/image.jpg"
            parts = line.split('\t')
            if len(parts) >= 3:
                full_path = parts[-1].strip()
                
                # 提取角色名和图片名
                filename = os.path.basename(full_path)
                if '_' in filename:
                    # 格式: RoleName_RoleName_0001.jpg
                    parts = filename.split('_')
                    role_name = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                    img_name = filename
                else:
                    role_name = filename.split('.')[0]
                    img_name = filename
                
                # 查找文件
                source_path = find_file(filename, SEARCH_DIRS)
                if not source_path:
                    source_path = find_file(img_name, SEARCH_DIRS)
                
                if source_path:
                    # 创建目标目录
                    target_dir = os.path.join(COMBINED_DATASET, role_name)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    target_path = os.path.join(target_dir, img_name)
                    
                    # 如果目标已存在，跳过
                    if os.path.exists(target_path):
                        print(f"⚠️ 已存在，跳过: {target_path}")
                        continue
                    
                    # 移动文件
                    shutil.move(source_path, target_path)
                    print(f"✅ 回滚: {source_path} -> {target_path}")
                    count += 1
                else:
                    print(f"❌ 未找到文件: {filename}")
    
    return count


def main():
    """主函数"""
    print("=" * 60)
    print("📦 开始回滚检测结果到 combined_dataset")
    print("=" * 60)
    
    # 创建目标目录
    os.makedirs(COMBINED_DATASET, exist_ok=True)
    
    # 回滚 multi_role_list.txt
    print("\n--- 处理 multi_role_list.txt ---")
    count1 = rollback_from_multi_role_list()
    
    # 回滚 multi_face_images.txt
    print("\n--- 处理 multi_face_images.txt ---")
    count2 = rollback_from_multi_face_list()
    
    print("\n" + "=" * 60)
    print(f"📊 回滚完成")
    print(f"  - 从 multi_role_list.txt 回滚: {count1} 个文件")
    print(f"  - 从 multi_face_images.txt 回滚: {count2} 个文件")
    print(f"  - 总计: {count1 + count2} 个文件")
    print("=" * 60)


if __name__ == '__main__':
    main()