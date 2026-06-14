#!/usr/bin/env python3
"""
通过网络搜索补充角色的英文名和日文名信息
输出格式: 中文名 作品名 英文名 日文名
"""

import os
import re
import json
import time
import random
from pathlib import Path
from collections import defaultdict

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("请先安装依赖: pip install requests beautifulsoup4")
    exit(1)

INPUT_FILE = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/all_characters_formatted.txt")
OUTPUT_FILE = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/all_characters_complete.txt")
CACHE_FILE = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/character_info_cache.json")

# 已知的角色信息缓存
KNOWN_CHARACTERS = {
    # 原神
    "钟离": {"english": "Zhongli", "japanese": "鍾離"},
    "雷电将军": {"english": "Raiden Shogun", "japanese": "雷電将軍"},
    "八重神子": {"english": "Yae Miko", "japanese": "八重神子"},
    "胡桃": {"english": "Hu Tao", "japanese": "胡桃"},
    "甘雨": {"english": "Ganyu", "japanese": "甘雨"},
    "刻晴": {"english": "Keqing", "japanese": "刻晴"},
    "七七": {"english": "Qiqi", "japanese": "チチ"},
    "可莉": {"english": "Klee", "japanese": "クレー"},
    "纳西妲": {"english": "Nahida", "japanese": "ナヒダ"},
    "迪奥娜": {"english": "Diona", "japanese": "ディオナ"},
    "早柚": {"english": "Sayu", "japanese": "サユ"},
    "派蒙": {"english": "Paimon", "japanese": "パイモン"},
    "芙宁娜": {"english": "Furina", "japanese": "フリーナ"},
    "纳西妲": {"english": "Nahida", "japanese": "ナヒダ"},
    "柯莱": {"english": "Collei", "japanese": "コレイ"},
    "多莉": {"english": "Dori", "japanese": "ドリ"},
    "绮良良": {"english": "Kirara", "japanese": "綺良々"},
    "希格雯": {"english": "Sigewinne", "japanese": "ジーグウィン"},
    "瑶瑶": {"english": "Yaoyao", "japanese": "ヤオヤオ"},
    "卡齐娜": {"english": "Kachina", "japanese": "カチナ"},
    
    # 崩坏星穹铁道
    "三月七": {"english": "March 7th", "japanese": "三月なのか"},
    "银狼": {"english": "Silver Wolf", "japanese": "シルバーウルフ"},
    "符玄": {"english": "Fu Xuan", "japanese": "符玄"},
    "黑塔": {"english": "Herta", "japanese": "ヘルタ"},
    "克拉拉": {"english": "Clara", "japanese": "クララ"},
    "虎克": {"english": "Hook", "japanese": "フック"},
    "白露": {"english": "Bailu", "japanese": "白露"},
    "流萤": {"english": "Firefly", "japanese": "ホタル"},
    "花火": {"english": "Sparkle", "japanese": "スパークル"},
    "云璃": {"english": "Yunli", "japanese": "雲璃"},
    "缇宝": {"english": "Tribbie", "japanese": "トリビー"},
    "阮梅": {"english": "Ruan Mei", "japanese": "ルアン・メイ"},
    "黄泉": {"english": "Yuan", "japanese": "黄泉"},
    "黑天鹅": {"english": "Black Swan", "japanese": "ブラックスワン"},
    "藿藿": {"english": "Huohuo", "japanese": "フォフォ"},
    "青雀": {"english": "Qingque", "japanese": "青雀"},
    "驭空": {"english": "Yukong", "japanese": "御空"},
    
    # 蔚蓝档案
    "阿洛娜": {"english": "Arona", "japanese": "アロナ"},
    "砂狼白子": {"english": "Shiroko", "japanese": "シロコ"},
    "伊吹": {"english": "Ibuki", "japanese": "イブキ"},
    "白洲梓": {"english": "Azusa", "japanese": "白洲アズサ"},
    "久田泉奈": {"english": "Izuna", "japanese": "イズナ"},
    "阿慈谷日富美": {"english": "Hifumi", "japanese": "阿慈谷ヒフミ"},
    "天见和香": {"english": "Nodoka", "japanese": "天見ノドカ"},
    "空崎日奈": {"english": "Hina", "japanese": "空崎ヒナ"},
    "圣园未花": {"english": "Mika", "japanese": "聖園ミカ"},
    "小鸟游星野": {"english": "Hoshino", "japanese": "小鳥遊ホシノ"},
    "天童爱丽丝": {"english": "Aris", "japanese": "アリス"},
    "黑见芹香": {"english": "Serika", "japanese": "黒見セリカ"},
    "十六夜野乃美": {"english": "Nonomi", "japanese": "十六夜ノノミ"},
    "天雨亚子": {"english": "Ako", "japanese": "天雨アコ"},
    "陆八魔爱露": {"english": "Aru", "japanese": "陸八魔アル"},
    "浅黄睦月": {"english": "Mutsuki", "japanese": "浅黄ムツキ"},
    "鬼方佳代子": {"english": "Kayoko", "japanese": "鬼方カヨコ"},
    "下江小春": {"english": "Koharu", "japanese": "下江コハル"},
    "狐坂若藻": {"english": "Wakamo", "japanese": "狐坂ワカモ"},
    
    # 崩坏3
    "德丽莎": {"english": "Theresa", "japanese": "テレサ"},
    "布洛妮娅": {"english": "Bronya", "japanese": "ブロニア"},
    "格蕾修": {"english": "Griseo", "japanese": "グレーシュ"},
    "符华": {"english": "Fu Hua", "japanese": "フファ"},
    "八重樱": {"english": "Yae Sakura", "japanese": "八重桜"},
    "雷电芽衣": {"english": "Raiden Mei", "japanese": "雷電芽衣"},
    "爱莉希雅": {"english": "Elysia", "japanese": "エリシア"},
    "丽塔": {"english": "Rita", "japanese": "リタ"},
    
    # 绝区零
    "可琳·威克斯": {"english": "Corin Wickes", "japanese": "コリン・ウィックス"},
    "珂蕾妲": {"english": "Koleda", "japanese": "コレーダ"},
    "柳": {"english": "Yanagi", "japanese": "柳"},
    
    # 其他
    "阿尼亚": {"english": "Anya", "japanese": "アーニャ"},
    "蕾姆": {"english": "Rem", "japanese": "レム"},
    "拉姆": {"english": "Ram", "japanese": "ラム"},
    "康娜": {"english": "Kanna", "japanese": "カンナ"},
}

def load_cache():
    """加载缓存"""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """保存缓存"""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def search_character_info(char_name, work_name):
    """搜索角色信息"""
    # 先检查已知信息
    if char_name in KNOWN_CHARACTERS:
        return KNOWN_CHARACTERS[char_name]
    
    # 检查是否是英文名或拼音
    if re.match(r'^[A-Za-z\s\.\-]+$', char_name):
        return {"english": char_name.strip(), "japanese": ""}
    
    print(f"搜索: {char_name} ({work_name})")
    
    try:
        # 使用 DuckDuckGo 搜索
        query = f"{char_name} {work_name} 英文名 日文名"
        url = f"https://duckduckgo.com/html/?q={requests.utils.quote(query)}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        english = ""
        japanese = ""
        
        # 尝试从搜索结果中提取信息
        for result in soup.find_all('div', class_='result'):
            text = result.get_text()
            
            # 查找英文名
            eng_match = re.search(r'(英文名|English name|English):?\s*([A-Za-z\s\.\-]+)', text)
            if eng_match and not english:
                english = eng_match.group(2).strip()
            
            # 查找日文名
            jp_match = re.search(r'(日文名|Japanese name|Japanese|假名):?\s*([\u3040-\u30FF\u4E00-\u9FFF]+)', text)
            if jp_match and not japanese:
                japanese = jp_match.group(2).strip()
            
            if english and japanese:
                break
        
        # 简单的启发式：如果角色名看起来像拼音，直接作为英文名
        if not english and re.match(r'^[A-Za-z\s]+$', char_name):
            english = char_name
        
        time.sleep(random.uniform(1.0, 3.0))
        
        return {"english": english, "japanese": japanese}
    
    except Exception as e:
        print(f"搜索失败: {e}")
        return {"english": "", "japanese": ""}

def main():
    print("=== 开始补充角色信息 ===")
    
    cache = load_cache()
    output_lines = []
    current_work = ""
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 解析作品名
            if line.startswith('#'):
                current_work = line.replace('#', '').split('(')[0].strip()
                output_lines.append(line)
                continue
            
            if not line:
                output_lines.append("")
                continue
            
            char_name = line
            
            # 检查缓存
            cache_key = f"{char_name}_{current_work}"
            if cache_key in cache:
                info = cache[cache_key]
            else:
                info = search_character_info(char_name, current_work)
                cache[cache_key] = info
                save_cache(cache)
            
            english = info.get("english", "")
            japanese = info.get("japanese", "")
            
            output_lines.append(f"{char_name} {current_work} {english} {japanese}")
    
    # 写入输出文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n=== 完成 ===")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"缓存文件: {CACHE_FILE}")

if __name__ == "__main__":
    main()
