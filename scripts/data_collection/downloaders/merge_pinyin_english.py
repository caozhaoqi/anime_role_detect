#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil

# 拼音到英文名的映射
PINYIN_TO_ENGLISH = {
    'duo1li4': 'Dori',           # 多莉
    'qi1qi1': 'Qiqi',             # 七七
    'ke4li4': 'Klee',             # 可莉
    'nai4xi1da2': 'Nahida',       # 纳西妲
    'ti2bao3': 'Princess',        # 缇宝
    'yao2yao2': 'Yaoyao',         # 瑶瑶
    'xi1ge2wen2': 'Sigewinne',    # 希格雯
    'fu2xuan2': 'Fu',             # 符玄
    'san1yue4qi1': 'March',       # 三月七
    'hua1huo3': 'Sparkle',        # 花火
    'yin2lang2': 'Silver',        # 银狼
    'tian1tong2ai4li4si1': 'Aris', # 天童爱丽丝
    'an1ke3': 'Encore',           # 安可
    'lu4mu4yuan2': 'Madoka',      # 鹿目圆
    'xiao3mei3yan4': 'Homura',    # 晓美焰
    'xue3xiao3ban3': 'Platelet',  # 血小板
    'lei2mu3': 'Rem',             # 雷姆
    'la1mu3': 'Ram',              # 拉姆
    'kang1na4': 'Kanna',          # 康娜
    'si4si1nai3': 'Yoshino',      # 四糸乃
    'kai3lu4': 'Kyaru',           # 凯露
    'yi4li4ya4': 'Illya',         # 伊莉雅
    'ren3ye3ren3': 'Oshino',      # 忍野忍
    'xiang1feng1zhi4nai3': 'Chino', # 香风智乃
    'zhi4nai3': 'Chino',          # 智乃
    'xiao3mai2': 'Umaru',         # 小埋
    'sha1wu4': 'Sagiri',          # 纱雾
    'mao1gong1you4nai4': 'Yanagi', # 猫宫又奈
    'de2li4sha1': 'Theresa',      # 德丽莎
    'bu4luo4ni2ya4': 'Bronya',    # 布洛妮娅
    'ke4lin2': 'Kira',            # 可琳
    'shen2le4': 'Kagura',         # 神乐
    'bai2shang4chui1xue3': 'Shirogane', # 白上吹雪
    'yue4qian1ye4': 'Tsukiyo',    # 月千夜
    'li4ta3la1': 'Lita',          # 莉塔拉
    'wei2pu3lei3': 'Viprey',      # 维普蕾
    'xia4ke4li3': 'Shakri',       # 夏克里
    'na4gan1': 'Nagan',           # 纳甘
    'ke1xie4ni2ya4': 'Koshenia',  # 科谢尼娅
    'kou4er3fu2': 'Korvu',        # 寇尔芙
    'ke4luo2li4ke1': 'Krokri',    # 克罗丽科
    'pei4li2ti2ya4': 'Peritia',   # 佩里缇亚
    'a1ni2ya4': 'Anya',           # 阿尼亚
    'luo4qian4': 'Rosci',         # 洛茜
    'zao4men2mi2dou4zi': 'Nezuko', # 灶门祢豆子
    'xi1er3': 'Seele',            # 希儿
    'xing4': 'An',                # 杏
    'yi1se4lin2': 'Iselin',       # 伊瑟琳
    'fu2lan2': 'Fran',            # 芙兰
    'fei1mi3li4si1': 'Fimilis',   # 菲米莉丝
    'sha1lang2bai2zi': 'Shiroko', # 砂狼白子
    'pai4meng2': 'Paimon',        # 派蒙
    'ke4la1la1': 'Clara',         # 克拉拉
    'ling2lan2': 'Suzuran',       # 铃兰
    'bai2xiao4hua1': 'Shirosaki', # 白咲花
    'xing1ye3ri4xiang4': 'Hoshino', # 星野日向
    'xing1ye3': 'Hoshino',        # 星野
    'ji1ban3nai4ai4': 'Himesaka', # 姬坂乃爱
    'zhong3cun1xiao3yi1': 'Tanemura', # 种村小依
    'xiao3zhi1sen1xia4yin1': 'Konomori', # 小之森夏音
    'chu2he4ai4': 'Hinaatsu',     # 雏鹤爱
    'ye4cha1shen2tian1yi1': 'Yashajin', # 夜叉神天衣
    'kong1yin2zi': 'Kuonji',      # 空银子
    'zao3lai4you1xiang1': 'Dream!', # 早濑优香
    'yi1zhi1lai4ming2ri4nai4': 'Ichinose', # 一之濑明日奈
    'sheng4yuan2wei4hua1': 'Mika', # 圣园未花
    'lei3bei4': 'Rebe',           # 蕾贝
    'luo4ke3ke3': 'luo4ke3ke3',   # 洛可可（无英文对应）
    'a1ni4ya4': 'Anya',           # 阿尼亚（备用）
    'jia1dai4zi': 'jia1dai4zi',   # 珈黛子（无英文对应）
    'li3shi4': 'li3shi4',         # 历史（无英文对应）
    'an1ka3xi1ya3': 'an1ka3xi1ya3', # 安卡西娅（无英文对应）
    'hua1yin1': 'hua1yin1',       # 花音（无英文对应）
    'shen1yue4': 'shen1yue4',     # 神乐（备用）
    'you4hu2': 'Youhu',           # 釉瑚
    'cong2yu3': 'cong2yu3',       # 从雨（无英文对应）
}

def merge_pinyin_english_directories(dataset_path):
    """合并拼音目录到英文目录"""
    merged_count = 0
    moved_files = 0
    
    for pinyin_name, english_name in PINYIN_TO_ENGLISH.items():
        pinyin_dir = os.path.join(dataset_path, pinyin_name)
        english_dir = os.path.join(dataset_path, english_name)
        
        # 跳过自己映射到自己的情况
        if pinyin_name == english_name:
            continue
            
        # 检查拼音目录是否存在
        if os.path.isdir(pinyin_dir):
            # 确保英文目录存在
            if not os.path.isdir(english_dir):
                os.makedirs(english_dir)
                print(f"创建目录: {english_dir}")
            
            # 获取拼音目录中的所有图片文件
            files = [f for f in os.listdir(pinyin_dir) if f.lower().endswith('.jpg')]
            
            if files:
                print(f"\n正在合并: {pinyin_name} -> {english_name}")
                print(f"  源目录文件数: {len(files)}")
                
                for filename in files:
                    src_path = os.path.join(pinyin_dir, filename)
                    dst_path = os.path.join(english_dir, filename)
                    
                    # 如果目标文件已存在，添加后缀
                    counter = 1
                    while os.path.exists(dst_path):
                        name, ext = os.path.splitext(filename)
                        dst_path = os.path.join(english_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.move(src_path, dst_path)
                    moved_files += 1
                
                # 删除空目录
                if not os.listdir(pinyin_dir):
                    os.rmdir(pinyin_dir)
                    print(f"  删除空目录: {pinyin_dir}")
                
                merged_count += 1
                print(f"  成功移动 {len(files)} 个文件")
    
    print(f"\n=== 合并完成 ===")
    print(f"合并目录数: {merged_count}")
    print(f"移动文件数: {moved_files}")

if __name__ == '__main__':
    dataset_path = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
    merge_pinyin_english_directories(dataset_path)