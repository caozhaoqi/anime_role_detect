#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清洗organized_images目录中的已有数据"""

import os
import sys
import hashlib
from pathlib import Path

ORGANIZED_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images')

# 已知的拼音映射
PINYIN_MAP = {
    '阿洛娜': 'a1luo4na4', '普拉娜': 'pu3la1na4', '纳西妲': 'na4xi1da2', '可莉': 'ke3li4',
    '迪奥娜': 'di2ao4na4', '瑶瑶': 'yao2yao2', '黑塔': 'hei1ta3', '符玄': 'fu2xuan2',
    '七七': 'qi1qi1', '早柚': 'zao3you4', '多莉': 'duo1li4', '卡齐娜': 'ka3qi2na4',
    '三月七': 'san1yue4qi1', '花火': 'hua1huo3', '银狼': 'yin2lang2', '雷姆': 'lei2mu3',
    '拉姆': 'la1mu3', '缇宝': 'ti2bao3', '维里奈': 'wei2li3nai4', '安可': 'an1ke3',
    '釉壶': 'you4hu2', '洛可可': 'luo4ke3ke3', '鹿目圆': 'lu4mu4yuan2', '晓美焰': 'xiao3mei3yan4',
    '血小板': 'xue4xiao3ban3', '康娜': 'kang1na4', '四糸乃': 'si4mi4nai3', '凯露': 'kai3lu4',
    '克萝萝': 'ke4luo2luo2', '小闪': 'xiao3shan3', '伊莉雅': 'yi1li4ya3', '忍野忍': 'ren3ye3ren3',
    '智乃': 'zhi4nai3', '小埋': 'xiao3mai2', '纱雾': 'sha1wu4', '猫宫又奈': 'mao1gong1you4nai4',
    '德丽莎': 'de2li4sha1', '布洛妮娅': 'bu4luo4ni2ya4', '可琳': 'ke3lin2', '爱丽儿': 'ai4li4er3',
    '神乐': 'shen2le4', '白上吹雪': 'bai2shang4chui1xue3', '月千夜': 'yue4qian1ye4',
    '芙丽希娅': 'fu2li4xi1ya4', '莉塔拉': 'li4ta3la1', '维普蕾': 'wei2pu3lei3', '夏克里': 'xia4ke4li3',
    '纳甘': 'na4gan1', '科谢尼娅': 'ke1xie4ni2ya4', '奇塔': 'qi2ta3', '寇尔芙': 'kou4er3fu2',
    '克罗丽科': 'ke4luo2li4ke1', '佩里缇亚': 'pei4li3ti2ya4', '阿尼亚': 'a1ni4ya4', '洛茜': 'luo4qian4',
    '祢豆子': 'mi3dou4zi5', '希儿': 'xi1er3', '杏': 'xing4', '伊瑟琳': 'yi1se4lin2',
    '芙兰': 'fu2lan2', '菲米莉丝': 'fei1mi3li4si1', '天童爱丽丝': 'tian1tong2ai4li4si1',
    '希格雯': 'xi1ge2wen2', '蕾贝': 'lei3bei4', '早雾': 'zao3wu4', 'ユウホ': 'you4hu2',
    'ありす': 'tian1tong2ai4li4si1',
}

def is_valid_image(filepath):
    """检查图片文件是否有效"""
    if not filepath.is_file():
        return False
    
    ext = filepath.suffix.lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
        return False
    
    # 检查文件大小（至少1KB）
    if filepath.stat().st_size < 1024:
        return False
    
    return True

def clean_empty_dirs(base_dir):
    """清理空目录"""
    empty_dirs = []
    for dir_path in sorted(base_dir.iterdir(), reverse=True):
        if dir_path.is_dir():
            files = list(dir_path.glob('*'))
            if len(files) == 0:
                print(f"删除空目录: {dir_path}")
                dir_path.rmdir()
                empty_dirs.append(dir_path.name)
    return empty_dirs

def clean_invalid_images(base_dir):
    """清理无效图片文件"""
    invalid_files = []
    for dir_path in base_dir.iterdir():
        if dir_path.is_dir():
            for img_file in dir_path.glob('*'):
                if not is_valid_image(img_file):
                    print(f"删除无效图片: {img_file}")
                    img_file.unlink()
                    invalid_files.append(str(img_file))
    return invalid_files

def deduplicate_images(base_dir):
    """删除重复图片（基于MD5哈希）"""
    hashes = {}
    duplicates = []
    
    for dir_path in base_dir.iterdir():
        if dir_path.is_dir():
            for img_file in dir_path.glob('*'):
                if is_valid_image(img_file):
                    try:
                        with open(img_file, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        
                        if file_hash in hashes:
                            print(f"删除重复图片: {img_file} (重复于 {hashes[file_hash]})")
                            img_file.unlink()
                            duplicates.append(str(img_file))
                        else:
                            hashes[file_hash] = str(img_file)
                    except Exception as e:
                        print(f"处理文件失败: {img_file} - {e}")
    return duplicates

def rename_chinese_dirs(base_dir):
    """将中文目录名重命名为拼音格式"""
    renamed = []
    for dir_path in base_dir.iterdir():
        if dir_path.is_dir():
            dir_name = dir_path.name
            
            # 如果已经是拼音格式，跳过
            if dir_name in PINYIN_MAP.values():
                continue
            
            # 检查是否在映射表中
            if dir_name in PINYIN_MAP:
                new_name = PINYIN_MAP[dir_name]
                new_path = base_dir / new_name
                
                if new_path.exists():
                    # 合并到已有目录
                    print(f"合并目录: {dir_path} -> {new_path}")
                    for img_file in dir_path.glob('*'):
                        if img_file.is_file():
                            new_file = new_path / img_file.name
                            if not new_file.exists():
                                img_file.rename(new_file)
                    # 删除原目录
                    dir_path.rmdir()
                else:
                    print(f"重命名目录: {dir_name} -> {new_name}")
                    dir_path.rename(new_path)
                renamed.append((dir_name, new_name))
            elif ' ' in dir_name:
                # 处理包含空格的目录名（如"纱雾 埃罗芒阿老师 Sagiri さぎり"）
                parts = dir_name.split()
                if parts[0] in PINYIN_MAP:
                    new_name = PINYIN_MAP[parts[0]]
                    new_path = base_dir / new_name
                    print(f"重命名复合目录: {dir_name} -> {new_name}")
                    if new_path.exists():
                        # 合并
                        for img_file in dir_path.glob('*'):
                            if img_file.is_file():
                                new_file = new_path / img_file.name
                                if not new_file.exists():
                                    img_file.rename(new_file)
                        dir_path.rmdir()
                    else:
                        dir_path.rename(new_path)
                    renamed.append((dir_name, new_name))
    return renamed

def main():
    print("=" * 60)
    print("🧹 开始清洗organized_images目录")
    print("=" * 60)
    
    # 1. 清理无效图片
    print("\n🔍 步骤1: 清理无效图片文件...")
    invalid_files = clean_invalid_images(ORGANIZED_DIR)
    print(f"   删除了 {len(invalid_files)} 个无效图片")
    
    # 2. 删除重复图片
    print("\n🔍 步骤2: 删除重复图片...")
    duplicates = deduplicate_images(ORGANIZED_DIR)
    print(f"   删除了 {len(duplicates)} 个重复图片")
    
    # 3. 重命名中文目录
    print("\n🔍 步骤3: 重命名中文目录为拼音格式...")
    renamed = rename_chinese_dirs(ORGANIZED_DIR)
    print(f"   重命名了 {len(renamed)} 个目录")
    
    # 4. 清理空目录
    print("\n🔍 步骤4: 清理空目录...")
    empty_dirs = clean_empty_dirs(ORGANIZED_DIR)
    print(f"   删除了 {len(empty_dirs)} 个空目录")
    
    # 统计结果
    print("\n" + "=" * 60)
    print("📊 清洗完成")
    print("=" * 60)
    
    total_images = 0
    total_dirs = 0
    for dir_path in ORGANIZED_DIR.iterdir():
        if dir_path.is_dir():
            total_dirs += 1
            for img_file in dir_path.glob('*'):
                if is_valid_image(img_file):
                    total_images += 1
    
    print(f"\n📁 角色目录数: {total_dirs}")
    print(f"🖼️  图片总数: {total_images}")
    print(f"\n✅ 清洗完成!")

if __name__ == '__main__':
    main()
