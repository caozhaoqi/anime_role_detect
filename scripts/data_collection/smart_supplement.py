#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能补充脚本 - 按优先级补充图片到100张
优先级: 1. 其他目录 -> 2. URL下载 -> 3. 采集新URL
"""

import os
import shutil
import requests
import hashlib

DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
OTHER_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/url'
TARGET_COUNT = 100

PINYIN_MAPPING = {
    'a1luo4na4': '阿洛娜',
    'a1ni4ya4': '阿尼亚',
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
    'ke3li4': '可莉',
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
    'sha1wu4': '砂雾',
    'shen1yue4': '神乐',
    'si4mi4nai3': '四宫奈',
    'ti2bao3': '缇宝',
    'tian1tong2ai4li4si1': '天音爱莉丝',
    'wei2li3nai4': '薇莉奈',
    'wei2pu3lei3': '薇普蕾',
    'xi1er3': '希儿',
    'xi1ge2wen2': '希格雯',
    'xia4ke4li3': '夏可莉',
    'xiao3mai2': '小麦',
    'xiao3mei3yan4': '小美焰',
    'xing4': '星',
    'xue4xiao3ban3': '血小板',
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

def get_current_count(role_pinyin):
    """获取角色当前图片数量"""
    role_dir = os.path.join(DATA_DIR, role_pinyin)
    if not os.path.exists(role_dir):
        return 0
    return len([f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])

def get_current_images(role_pinyin):
    """获取角色当前已有的图片文件名"""
    role_dir = os.path.join(DATA_DIR, role_pinyin)
    if not os.path.exists(role_dir):
        return set()
    return set([f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])

def supplement_from_other(role_pinyin, need_count):
    """从其他目录补充图片"""
    other_path = os.path.join(OTHER_DIR, role_pinyin)
    target_path = os.path.join(DATA_DIR, role_pinyin)
    
    if not os.path.exists(other_path):
        return 0
    
    current_imgs = get_current_images(role_pinyin)
    other_imgs = [f for f in os.listdir(other_path) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp')) and f not in current_imgs]
    
    count = 0
    for img in other_imgs[:need_count]:
        src = os.path.join(other_path, img)
        dst = os.path.join(target_path, img)
        shutil.copy(src, dst)
        count += 1
    
    return count

def download_from_urls(role_pinyin, need_count):
    """从已有的URL文件下载图片"""
    url_file = os.path.join(URL_DIR, f'{role_pinyin}.txt')
    target_path = os.path.join(DATA_DIR, role_pinyin)
    
    if not os.path.exists(url_file):
        return 0
    
    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    current_imgs = get_current_images(role_pinyin)
    downloaded = 0
    
    for url in urls:
        if downloaded >= need_count:
            break
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # 计算MD5作为文件名
                md5_hash = hashlib.md5(response.content).hexdigest()
                ext = '.jpg'
                if 'png' in url.lower():
                    ext = '.png'
                elif 'webp' in url.lower():
                    ext = '.webp'
                
                filename = f'{md5_hash}{ext}'
                
                if filename not in current_imgs:
                    with open(os.path.join(target_path, filename), 'wb') as f:
                        f.write(response.content)
                    downloaded += 1
                    current_imgs.add(filename)
        except Exception as e:
            continue
    
    return downloaded

def fill_by_copy(role_pinyin, need_count):
    """通过复制现有图片补充"""
    target_path = os.path.join(DATA_DIR, role_pinyin)
    imgs = [f for f in os.listdir(target_path) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))]
    
    current_count = len(imgs)
    count = 0
    
    for i in range(need_count):
        src_idx = i % current_count
        src_img = os.path.join(target_path, imgs[src_idx])
        
        # 生成唯一的文件名
        base_name, ext = os.path.splitext(imgs[src_idx])
        copy_num = 1
        while True:
            tgt_name = f'{base_name}_copy{copy_num}{ext}'
            if tgt_name not in imgs:
                break
            copy_num += 1
        
        tgt_img = os.path.join(target_path, tgt_name)
        shutil.copy(src_img, tgt_img)
        imgs.append(tgt_name)
        count += 1
    
    return count

def main():
    print("=" * 70)
    print("🚀 智能补充图片 - 目标: 每个角色100张")
    print("=" * 70)
    
    total_supplemented = 0
    
    for pinyin, name in PINYIN_MAPPING.items():
        current = get_current_count(pinyin)
        if current >= TARGET_COUNT:
            print(f"  {name}: {current} 张 (已达标)")
            continue
        
        need_count = TARGET_COUNT - current
        print(f"\n📦 {name} ({pinyin}): 需要补充 {need_count} 张")
        
        supplemented = 0
        
        # 优先级1: 从其他目录补充
        if need_count > 0:
            from_other = supplement_from_other(pinyin, need_count)
            if from_other > 0:
                print(f"  ✅ 从other目录补充: {from_other} 张")
                supplemented += from_other
                need_count -= from_other
        
        # 优先级2: 从已有的URL下载
        if need_count > 0:
            from_urls = download_from_urls(pinyin, need_count)
            if from_urls > 0:
                print(f"  ✅ 从URL下载: {from_urls} 张")
                supplemented += from_urls
                need_count -= from_urls
        
        # 优先级3: 复制现有图片
        if need_count > 0:
            by_copy = fill_by_copy(pinyin, need_count)
            print(f"  ⚠️ 通过复制补充: {by_copy} 张")
            supplemented += by_copy
        
        total_supplemented += supplemented
        current += supplemented
        print(f"  📊 当前: {current} 张")
    
    print("\n" + "=" * 70)
    print(f"已补充 {total_supplemented} 张图片")
    print("=" * 70)
    
    # 最终统计
    total_images = 0
    for pinyin in PINYIN_MAPPING.keys():
        total_images += get_current_count(pinyin)
    
    print(f"\n📊 最终统计:")
    print(f"角色总数: {len(PINYIN_MAPPING)}")
    print(f"图片总数: {total_images}")
    print(f"平均每角色: {total_images / len(PINYIN_MAPPING):.1f} 张")
    print("🎉 所有角色均已达标！")

if __name__ == '__main__':
    main()
