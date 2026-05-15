#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并拼音目录到英文目录，统一数据目录结构
"""
import os
import shutil
from pypinyin import pinyin, Style

# 配置
DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
ROLE_LIST_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'

# 手动映射表（拼音到英文）
PINYIN_TO_ENGLISH = {
    'qi1qi1': 'Qiqi',
    'zao3you4': 'Sayu',
    'ya4oyao4': 'Yaoyao',
    'fu4xuan2': 'Fu Xuan',
    'a1luo2na4': 'Arona',
    'pu3la1na4': 'Plana',
    'sha1lang2bai2zi3': 'Shiroko',
    'na4xi1da2': 'Nahida',
    'ti2bao3': 'Princess',
    'ke3li4': 'Klee',
    'di4ao4na4': 'Diona',
    'xi1ge2wen2': 'Sigewinne',
    'lei3bei4': 'Rebe',
    'hei1ta3': 'Herta',
    'duo1li4': 'Dori',
    'ka3qi2na4': 'Kachina',
    'san1yue4qi1': 'March 7th',
    'hua1huo3': 'Sparkle',
    'yin2lang2': 'Silver Wolf',
    'tian1tong2ai4li4si1': 'Aris',
    'zao3wu4': 'Hayiri',
    'wei2li3nai4': 'Verina',
    'an1ke4': 'Encore',
    'you4hu2': 'Youhu',
    'lu4mu4yuan2': 'Madoka Kaname',
    'xiao3mei3yan4': 'Homura Akemi',
    'xue3xiao4ban3': 'Platelet',
    'lei2mu3': 'Rem',
    'la1mu3': 'Ram',
    'kang1na4': 'Kanna',
    'si4mi4nai3': 'Yoshino',
    'kai3lu4': 'Kyaru',
    'yi1li4ya3': 'Illya',
    'ren3ye3ren3': 'Oshino Shinobu',
    'xiang1feng1zhi4nai3': 'Chino',
    'xiao3mai2': 'Umaru',
    'sha1wu4': 'Sagiri',
    'mao1gong1you4nai4': 'Yanagi',
    'de2li4sha1': 'Theresa',
    'bu4luo4ni4ya4': 'Bronya',
    'ke3lin2': 'Kira',
    'shen2le4': 'Kagura',
    'bai2shang4chui1xue3': 'Shirogane Noel',
    'yue4qian1ye4': 'Tsukiyo',
    'li4ta1la1': 'Lita',
    'wei2pu3lei2': 'Viprey',
    'xia4ke1li3': 'Shakri',
    'na4gan1': 'Nagan',
    'ke1she4ni4ya4': 'Koshenia',
    'kou4er3fu2': 'Korvu',
    'ke1luo2li3ke4': 'Krokri',
    'pei4li2ti4ya4': 'Peritia',
    'a1ni4ya4': 'Anya Forger',
    'luo4qian1': 'Rosci',
    'dou4men2ni2dou4zi5': 'Nezuko Kamado',
    'xi1er3': 'Seele Vollerei',
    'xing4': 'An Marhall',
    'yi1se4lin2': 'Iselin LeviSius',
    'fu2lan2': 'Fran',
    'fei1mi4li2si1': 'Fimilis',
    'ke4la1la1': 'Clara',
    'ling2lan2': 'Suzuran',
    'bai2xiao4hua1': 'Shirosaki Hana',
    'xing1ye3ri4xiang4': 'Hoshino Hinata',
    'ji1ban3nai4ai4': 'Himesaka Noa',
    'zhong3cun1xiao3yi1': 'Tanemura Koharu',
    'xiao3zhi1sen1xia4yin1': 'Konomori Kanon',
    'chu2he4ai4': 'Hinaatsu Ai',
    'ye4cha1shen2tian1yi1': 'Yashajin Ti衣',
    'kong1yin2zi3': 'Kuonji Gin',
    'zao3lai4you1xiang1': 'Yuka Hayase',
    'yi1zhi1lai4ming2ri4nai4': 'Ichinose Asuna',
    'kong1qi1ri4nai4': 'Hina Sorasaki',
    'sheng4yuan2wei4hua1': 'Mika Misono',
    'xiao3niao3you2xing1ye3': 'Hoshino Tendou',
    'bo2he2': 'Hayiri',
    'a1ni4ya4fu2ge2er3': 'Anya Forger',
    'a1ni4': 'Anya',
    'a1ni4ya4': 'Anya',
    'a1ni4ya4fu2ge2er3': 'Anya Forger',
    # 额外的拼音变体
    'ri4nai4': 'Rem',
    'shen1yue4': 'Nahida',
    'ren3': 'Rem',
    'hua1yin1': 'Homura',
    'qi1qi1': 'Qiqi',
    'cong2yu3': 'Sayu',
    'dong1you1zi': 'Princess',
    'jia1dai4zi': 'Kyaru',
    'li3shi4': 'Theresa',
    'luo4ke3ke3': 'Krokri',
    'mu4yue4': 'March 7th',
    'ni2dou4zi5': 'Nezuko Kamado',
    'qian1xia4': 'Platelet',
    'wu4yu3mo2li3sha1': 'Iselin LeviSius',
    'ai4li4er3': 'Illya',
    'an1ka3xi1ya3': 'Aris',
}

def load_role_list():
    """加载角色列表"""
    roles = {}
    with open(ROLE_LIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                cn_name = parts[0]
                en_name = parts[2]
                roles[cn_name] = en_name
    return roles

def generate_pinyin_variants(cn_name):
    """生成拼音变体"""
    variants = []
    # 带声调的拼音
    pinyin_with_tone = ''.join([p[0] for p in pinyin(cn_name, style=Style.TONE3)])
    variants.append(pinyin_with_tone)
    # 无声调的拼音
    pinyin_without_tone = ''.join([p[0] for p in pinyin(cn_name, style=Style.NORMAL)])
    variants.append(pinyin_without_tone)
    return variants

def find_pinyin_directories():
    """找到所有拼音目录（包含数字的目录）"""
    pinyin_dirs = []
    for dirname in os.listdir(DATASET_PATH):
        dir_path = os.path.join(DATASET_PATH, dirname)
        if os.path.isdir(dir_path) and any(c.isdigit() for c in dirname):
            pinyin_dirs.append(dirname)
    return pinyin_dirs

def merge_pinyin_to_english():
    """合并拼音目录到英文目录"""
    roles = load_role_list()
    pinyin_dirs = find_pinyin_directories()
    
    merged_count = 0
    moved_files = 0
    deleted_dirs = 0
    
    print("=" * 80)
    print("🔄 开始合并拼音目录到英文目录")
    print("=" * 80)
    print(f"找到 {len(pinyin_dirs)} 个拼音目录")
    
    for pinyin_name in sorted(pinyin_dirs):
        pinyin_dir = os.path.join(DATASET_PATH, pinyin_name)
        
        # 检查是否为空目录
        if not os.listdir(pinyin_dir):
            print(f"🗑️ 删除空目录: {pinyin_name}")
            os.rmdir(pinyin_dir)
            deleted_dirs += 1
            continue
        
        # 查找对应的英文目录
        en_name = None
        
        # 1. 先查手动映射表
        if pinyin_name in PINYIN_TO_ENGLISH:
            en_name = PINYIN_TO_ENGLISH[pinyin_name]
        else:
            # 2. 尝试匹配角色列表中的拼音
            for cn_name, target_en_name in roles.items():
                variants = generate_pinyin_variants(cn_name)
                if pinyin_name in variants or pinyin_name.startswith(variants[0][:4]):
                    en_name = target_en_name
                    break
        
        if en_name:
            en_dir = os.path.join(DATASET_PATH, en_name)
            
            # 创建英文目录（如果不存在）
            if not os.path.isdir(en_dir):
                os.makedirs(en_dir)
                print(f"📁 创建目录: {en_name}")
            
            # 移动文件
            files = [f for f in os.listdir(pinyin_dir) if f.lower().endswith('.jpg')]
            if files:
                print(f"🔄 合并: {pinyin_name} -> {en_name}")
                print(f"  文件数: {len(files)}")
                
                for filename in files:
                    src_path = os.path.join(pinyin_dir, filename)
                    dst_path = os.path.join(en_dir, filename)
                    
                    # 处理重复文件名
                    counter = 1
                    while os.path.exists(dst_path):
                        name, ext = os.path.splitext(filename)
                        dst_path = os.path.join(en_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.move(src_path, dst_path)
                    moved_files += 1
                
                # 删除空目录
                if not os.listdir(pinyin_dir):
                    os.rmdir(pinyin_dir)
                    deleted_dirs += 1
                    print(f"  ✅ 已删除空目录: {pinyin_name}")
                
                merged_count += 1
        else:
            print(f"❓ 未找到 {pinyin_name} 对应的英文目录，跳过")
    
    print("\n" + "=" * 80)
    print("✅ 合并完成")
    print("=" * 80)
    print(f"合并目录数: {merged_count}")
    print(f"移动文件数: {moved_files}")
    print(f"删除空目录数: {deleted_dirs}")
    
    # 检查剩余的拼音目录
    remaining_pinyin = find_pinyin_directories()
    if remaining_pinyin:
        print(f"\n⚠️ 仍有 {len(remaining_pinyin)} 个拼音目录未合并:")
        for p in remaining_pinyin:
            print(f"  - {p}")
    
    return merged_count, moved_files, deleted_dirs

if __name__ == '__main__':
    merge_pinyin_to_english()