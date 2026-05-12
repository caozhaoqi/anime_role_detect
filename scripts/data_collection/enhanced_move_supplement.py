#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版移动补充脚本 - 确保正确移动图片
"""

import os
import shutil

DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
OTHER_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
TARGET_COUNT = 100

PINYIN_MAPPING = {
    'a1luo4na4': '阿洛娜',
    'a1ni4ya4': '阿妮娅',
    'an1ke3': '安可',
    'bai2shang4chui1xue3': '白上吹雪',
    'bu4luo4ni2ya4': '布洛妮娅',
    'de2li4sha1': '德丽莎',
    'di2ao4na4': '迪奥娜',
    'duo1li4': '多莉',
    'fei1mi3li4si1': '菲米莉丝',
    'fu2lan2': '芙兰',
    'fu2xuan2': '符玄',
    'hei1ta3': '黑塔',
    'hua1huo3': '火花',
    'ka3qi2na4': '卡琪娜',
    'kai3lu4': '开萝',
    'kang1na4': '康娜',
    'ke1xie4ni2ya4': '克谢尼娅',
    'ke3li4': '刻晴',
    'ke3lin2': '克林',
    'ke4la1la1': '克拉拉',
    'ke4luo2li4ke1': '克罗丽克',
    'kou4er3fu2': '蔻尔芙',
    'la1mu3': '拉姆',
    'lei2mu3': '雷姆',
    'lei3bei4': '蕾贝',
    'li4ta3la1': '莉塔菈',
    'lu4mu4yuan2': '鹿目圆',
    'luo4qian4': '洛茜',
    'mao1gong1you4nai4': '猫宫又奈',
    'mi2dou4zi': '蜜豆子',
    'na4gan1': '娜甘',
    'na4xi1da2': '纳西妲',
    'pei4li3ti2ya4': '佩丽缇娅',
    'pu3la1na4': '普拉娜',
    'qi1qi1': '七七',
    'ren3ye3ren3': '人外人',
    'san1yue4qi1': '三月七',
    'sha1wu4': '砂狼',
    'shen1yue4': '神乐',
    'si4mi4nai3': '四宫奈',
    'ti2bao3': '提宝',
    'tian1tong2ai4li4si1': '天音爱莉丝',
    'wei2li3nai4': '薇莉奈',
    'wei2pu3lei3': '薇普蕾',
    'xi1er3': '希尔',
    'xi1ge2wen2': '希格雯',
    'xia4ke4li3': '夏可莉',
    'xiao3mai2': '小麦',
    'xiao3mei3yan4': '小美焰',
    'xing4': '星',
    'xue4xiao3ban3': '学校版',
    'yao2yao2': '瑶瑶',
    'yi1li4ya3': '伊莉雅',
    'yi1se4lin2': '伊瑟琳',
    'yin2lang2': '银狼',
    'you4hu2': '釉瑚',
    'yue4qian1ye4': '月千夜',
    'zao3wu4': '早雾',
    'zao3you4': '早柚',
    'zhi4nai3': '智乃'
}

def get_image_count(dir_path):
    """获取目录中的图片数量"""
    if not os.path.exists(dir_path):
        return 0
    return len([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])

def get_image_names(dir_path):
    """获取目录中的图片文件名集合"""
    if not os.path.exists(dir_path):
        return set()
    return set([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])

def move_images(source_dir, target_dir, need_count):
    """移动指定数量的图片"""
    if not os.path.exists(source_dir):
        return 0
    
    source_imgs = get_image_names(source_dir)
    target_imgs = get_image_names(target_dir)
    
    # 找出源目录中目标目录没有的图片
    available_imgs = list(source_imgs - target_imgs)
    actual_move = min(len(available_imgs), need_count)
    
    moved_count = 0
    for img_name in available_imgs[:actual_move]:
        src_path = os.path.join(source_dir, img_name)
        dst_path = os.path.join(target_dir, img_name)
        shutil.move(src_path, dst_path)
        moved_count += 1
    
    return moved_count

def main():
    print("=" * 70)
    print("🚀 增强版移动补充图片")
    print("规则: 仅使用移动，不使用复制")
    print("=" * 70)
    
    total_moved = 0
    remaining_roles = []
    
    # 首先统计organized_images中有多少可用图片
    print("\n📋 organized_images可用图片统计:")
    for pinyin in PINYIN_MAPPING.keys():
        other_dir = os.path.join(OTHER_DIR, pinyin)
        count = get_image_count(other_dir)
        if count > 0:
            print(f"  {pinyin}: {count} 张")
    
    print("\n" + "=" * 70)
    
    for pinyin, name in PINYIN_MAPPING.items():
        target_dir = os.path.join(DATA_DIR, pinyin)
        other_dir = os.path.join(OTHER_DIR, pinyin)
        
        current_count = get_image_count(target_dir)
        
        if current_count >= TARGET_COUNT:
            print(f"  ✅ {name}: {current_count} 张 (已达标)")
            continue
        
        need_count = TARGET_COUNT - current_count
        print(f"\n📦 {name} ({pinyin}): 当前 {current_count} 张, 需要补充 {need_count} 张")
        
        # 从organized_images移动
        if os.path.exists(other_dir):
            other_count = get_image_count(other_dir)
            if other_count > 0:
                moved = move_images(other_dir, target_dir, need_count)
                if moved > 0:
                    print(f"  🔄 从organized_images移动: {moved} 张")
                    total_moved += moved
                    need_count -= moved
        
        current_count = get_image_count(target_dir)
        
        if current_count >= TARGET_COUNT:
            print(f"  ✅ 当前: {current_count} 张 (已达标)")
        else:
            print(f"  ⚠️ 当前: {current_count} 张 (仍需 {need_count} 张)")
            remaining_roles.append((name, current_count, need_count))
    
    print("\n" + "=" * 70)
    print(f"总共移动: {total_moved} 张")
    print("=" * 70)
    
    # 最终统计
    total_images = sum(get_image_count(os.path.join(DATA_DIR, p)) for p in PINYIN_MAPPING.keys())
    
    print(f"\n📊 最终统计:")
    print(f"角色总数: {len(PINYIN_MAPPING)}")
    print(f"图片总数: {total_images}")
    print(f"平均每角色: {total_images / len(PINYIN_MAPPING):.1f} 张")
    
    if remaining_roles:
        print(f"\n⚠️ 未达标的角色 ({len(remaining_roles)}个):")
        for name, current, need in remaining_roles:
            print(f"  {name}: {current}/{TARGET_COUNT}")
        
        print("\n💡 未达标的角色需要通过URL下载补充")
        print("请启动爬虫API服务后运行URL采集脚本")
    else:
        print("\n🎉 所有角色均已达标！")

if __name__ == '__main__':
    main()
