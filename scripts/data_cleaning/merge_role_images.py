#!/usr/bin/env python3
"""
根据角色合并两个目录的图片文件
将 downloaded_images 中的图片合并到 merged_english_dataset 对应的角色目录
"""

import os
import shutil
from pathlib import Path

# 定义目录路径
BASE_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect")
SOURCE_DIR = BASE_DIR / "spider_image_system/src/run/data/downloaded_images"
TARGET_DIR = BASE_DIR / "data/merged_english_dataset"

# 角色名称映射表 (拼音 -> 英文)
ROLE_MAPPING = {
    "a1luo4na4": "Arona",           # 阿罗娜
    "a1ni4ya4": "Anya",             # 阿尼亚
    "ai4li4er3": "Elysia",          # 爱莉希雅
    "an1ka3xi1ya3": "Ankaxiya",     # 安卡西亚
    "an1ke3": "Anke",               # 安可
    "bai2shang4chui1xue3": "Bai Shang Chuixue",  # 白上吹雪
    "bu4luo4ni2ya4": "Bronya",      # 布洛妮娅
    "cong2yu3": "Congyu",           # 葱郁
    "de2li4sha1": "Theresa",        # 德丽莎
    "di2ao4na4": "Di Ao Na",        # 迪奥娜
    "duo1li4": "Dori",              # 多莉
    "fei1mi3li4si1": "Fimilis",     # 菲米莉丝
    "fei1xie4er3": "Feixieer",      # 菲谢尔
    "fu2lan2": "Fulan",             # 芙兰
    "fu2li4xi1ya4": "Furixiya",     # 芙莉西亚
    "fu2xuan2": "Fu Xuan",          # 符玄
    "gu3ming2di4lian4": "Gu Ming Di Lian",  # 古明地恋
    "hei1ta3": "Herta",             # 黑塔
    "hua1huo3": "Sparkle",          # 花火
    "ka3qi2na4": "Kachina",         # 卡奇娜
    "kai3lu4": "Kaelu",             # 凯露
    "kang1na4": "Kanna",            # 康娜
    "ke1xie4ni2ya4": "Kexieni Ya",  # 克谢尼娅
    "ke3li2": "Klee",               # 可莉
    "ke3li4": "Klee",               # 可莉
    "ke3lin2": "Collei",            # 柯莱
    "ke4luo2luo2": "Columbina",     # 哥伦比娜
    "kou4er3fu2": "Kou Erfu",       # 库尔夫
    "la1mu3": "Lam",                # 拉姆
    "li3sa4": "Lisa",               # 丽莎
    "lu4ke3ke3": "Rukia",           # 露琪亚
    "lu4mu4yuan2": "Madoka",        # 鹿目圆
    "luo4qian4": "Luo Qian",        # 罗茜
    "na4xi1da2": "Nahida",          # 纳西妲
    "ni2dou4zi5": "Nezuko",         # 祢豆子
    "qi1qi1": "Qiqi",               # 七七
    "qing1que4": "Qing Que",        # 青雀
    "ren3ye3ren3": "Ren Ye Ren",    # 人夜人
    "san1yue4qi1": "March 7th",     # 三月七
    "si4mi4nai3": "Sigewinne",      # 希格雯
    "ti2bao3": "Tibao",             # 提宝
    "wei2li3nai4": "Viprey",        # 薇尔莉娜
    "wei2pu3lei3": "Vepley",        # 薇普蕾
    "wu4la1": "Wula",               # 乌拉
    "xi1er3": "Seele",              # 希儿
    "xia4ke4li3": "Xia Keli",       # 夏克莉
    "xiao3mei3yan4": "Xiao Mei Yan", # 小媚眼
    "xing4": "Xing",                # 星
    "yin2lang2": "Yin Lang",        # 银狼
    "yao2yao2 yuan2shen2": "Yaoyao", # 瑶瑶(原神)
    "yue4qian1ye4": "Yue Qian Ye",  # 月下夜
    "zao3wu4": "Zao Wu",            # 早雾
    "zao3you4": "Zao You",          # 早柚
    "ありす": "Alice"                # 爱丽丝(日文)
}

def merge_images():
    """合并两个目录的图片"""
    merged_count = 0
    skipped_count = 0
    error_count = 0
    
    # 确保目标目录存在
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== 开始合并角色图片 ===")
    print(f"源目录: {SOURCE_DIR}")
    print(f"目标目录: {TARGET_DIR}")
    print()
    
    # 遍历源目录中的所有角色文件夹
    for source_role in os.listdir(SOURCE_DIR):
        source_path = SOURCE_DIR / source_role
        
        # 跳过非目录项
        if not source_path.is_dir():
            continue
            
        # 获取图片列表
        images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")) + list(source_path.glob("*.webp"))
        
        if not images:
            print(f"跳过空目录: {source_role}")
            skipped_count += 1
            continue
            
        # 确定目标角色名称
        if source_role in ROLE_MAPPING:
            target_role = ROLE_MAPPING[source_role]
        else:
            # 直接使用原名称（可能是英文）
            target_role = source_role
        
        # 创建目标目录
        target_path = TARGET_DIR / target_role
        target_path.mkdir(parents=True, exist_ok=True)
        
        # 复制图片
        print(f"合并角色: {source_role} -> {target_role} ({len(images)} 张图片)")
        
        for img in images:
            try:
                target_file = target_path / img.name
                
                # 如果文件已存在，跳过
                if target_file.exists():
                    skipped_count += 1
                    continue
                    
                shutil.copy2(img, target_file)
                merged_count += 1
            except Exception as e:
                print(f"  复制失败 {img.name}: {e}")
                error_count += 1
    
    print()
    print("=== 合并完成 ===")
    print(f"成功合并: {merged_count} 张")
    print(f"跳过(已存在): {skipped_count} 张")
    print(f"错误: {error_count} 张")
    
    return merged_count, skipped_count, error_count

if __name__ == "__main__":
    merge_images()
