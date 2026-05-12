#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采集剩余角色 - 使用中文名采集（避免英文名问题）
"""

import os
import sys
import requests
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import *
from scripts.utils import *

# 剩余未采集成功的角色
remaining_roles = [
    ('维里奈', 'Verina'),
    ('安可', 'Encore'),
    ('釉瑚', 'Youhu'),
    ('鹿目圆', 'Madoka Kaname'),
    ('晓美焰', 'Homura Akemi'),
    ('血小板', 'Platelet'),
    ('雷姆', 'Rem'),
    ('拉姆', 'Ram'),
    ('康娜', 'Kanna'),
    ('四糸乃', 'Yoshino'),
    ('凯露', 'Kyaru'),
    ('伊莉雅', 'Illya'),
    ('忍野忍', 'Oshino Shinobu'),
    ('香风智乃', 'Chino'),
    ('小埋', 'Umaru'),
    ('纱雾', 'Sagiri'),
    ('猫宫又奈', 'Yanagi'),
    ('德丽莎', 'Theresa'),
    ('布洛妮娅', 'Bronya'),
    ('可琳', 'Kira'),
    ('神乐', 'Kagura'),
    ('白上吹雪', 'Shirogane Noel'),
    ('月千夜', 'Tsukiyo'),
    ('莉塔拉', 'Lita'),
    ('维普蕾', 'Viprey'),
    ('夏克里', 'Shakri'),
    ('纳甘', 'Nagan'),
    ('科谢尼娅', 'Koshenia'),
    ('寇尔芙', 'Korvu'),
    ('克罗丽科', 'Krokri'),
    ('佩里缇亚', 'Peritia'),
    ('阿尼亚', 'Anya Forger'),
    ('洛茜', 'Rosci'),
    ('灶门祢豆子', 'Nezuko Kamado'),
    ('希儿', 'Seele Vollerei'),
    ('杏', 'An Marhall'),
    ('伊瑟琳', 'Iselin LeviSius'),
    ('芙兰', 'Fran'),
    ('菲米莉丝', 'Fimilis'),
    ('克拉拉', 'Clara'),
    ('铃兰', 'Suzuran'),
    ('白咲花', 'Shirosaki Hana'),
    ('星野日向', 'Hoshino Hinata'),
    ('姬坂乃爱', 'Himesaka Noa'),
    ('种村小依', 'Tanemura Koharu'),
    ('小之森夏音', 'Konomori Kanon'),
    ('雏鹤爱', 'Hinaatsu Ai'),
    ('夜叉神天衣', 'Yashajin Ti'),
    ('空银子', 'Kuonji Gin'),
    ('早濑优香', 'Yuka Hayase'),
    ('一之濑明日奈', 'Ichinose Asuna'),
    ('空崎日奈', 'Hina Sorasaki'),
    ('圣园未花', 'Mika Misono'),
    ('小鸟游星野', 'Hoshino Tendou')
]

def main():
    print("=" * 70)
    print("🚀 采集剩余角色（使用中文名）")
    print(f"剩余角色数: {len(remaining_roles)}")
    print("=" * 70)
    
    # 检查爬虫服务
    try:
        response = requests.get(f"{API_BASE}/spider/status", timeout=5)
        print("✅ 爬虫服务连接成功")
    except Exception as e:
        print(f"❌ 爬虫服务未运行: {str(e)}")
        return
    
    success_count = 0
    fail_count = 0
    
    for i, (name, en_name) in enumerate(remaining_roles, 1):
        print(f"\n[{i}/{len(remaining_roles)}] {name}")
        
        if spider_single_role(name):
            print(f"  ✅ 采集成功")
            success_count += 1
        else:
            print(f"  ❌ 采集失败")
            fail_count += 1
        
        time.sleep(3)
    
    print("\n" + "=" * 70)
    print(f"采集完成: 成功 {success_count} 个, 失败 {fail_count} 个")
    print("=" * 70)

if __name__ == '__main__':
    main()
