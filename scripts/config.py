#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置文件
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REORGANIZED_DATASET = os.path.join(DATA_DIR, "reorganized_dataset")
SPIDER_DATA_DIR = os.path.join(PROJECT_ROOT, "spider_image_system", "data")
URL_DIR = os.path.join(SPIDER_DATA_DIR, "img_url")

# 角色名单文件
ROLE_FILE = os.path.join(PROJECT_ROOT, "auto_spider_img", "loli-role.txt")

# 爬虫API配置
API_BASE = "http://localhost:33333/api/v1.2.5.260305/sis"

# 目标数量
TARGET_COUNT = 100

# 日志配置
LOG_LEVEL = "INFO"
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# 角色映射表（中文名 -> 英文名）
ROLE_MAPPING = {
    "阿洛娜": {"en": "Arona", "jp": "アロナ", "pinyin": "a1luo4na4"},
    "普拉娜": {"en": "Plana", "jp": "プラナ", "pinyin": "pu1la1na4"},
    "砂狼白子": {"en": "Shiroko", "jp": "シロコ", "pinyin": "sha1lang2bai2zi3"},
    "纳西妲": {"en": "Nahida", "jp": "ナヒダ", "pinyin": "na4xi1da2"},
    "缇宝": {"en": "Princess", "jp": "プリンセス", "pinyin": "ti2bao3"},
    "可莉": {"en": "Klee", "jp": "クレー", "pinyin": "ke3li4"},
    "迪奥娜": {"en": "Diona", "jp": "ディオナ", "pinyin": "di2ao4na4"},
    "瑶瑶": {"en": "Yaoyao", "jp": "ヤオヤオ", "pinyin": "yao2yao2"},
    "希格雯": {"en": "Sigewinne", "jp": "ジーグウィン", "pinyin": "xi1ge2wen2"},
    "蕾贝": {"en": "Rebe", "jp": "レベ", "pinyin": "lei3bei4"},
    "黑塔": {"en": "Herta", "jp": "ヘルタ", "pinyin": "hei1ta3"},
    "符玄": {"en": "Fu Xuan", "jp": "フウゲン", "pinyin": "fu2xuan2"},
    "七七": {"en": "Qiqi", "jp": "チチ", "pinyin": "qi1qi1"},
    "早柚": {"en": "Sayu", "jp": "サユ", "pinyin": "zao3you4"},
    "多莉": {"en": "Dori", "jp": "ドリ", "pinyin": "duo1li4"},
    "派蒙": {"en": "Paimon", "jp": "パイモン", "pinyin": "pai4meng2"},
    "卡齐娜": {"en": "Kachina", "jp": "カチナ", "pinyin": "ka3qi2na4"},
    "三月七": {"en": "March 7th", "jp": "マーチセブンス", "pinyin": "san1yue4qi1"},
    "花火": {"en": "Sparkle", "jp": "スパークル", "pinyin": "hua1huo3"},
    "火花": {"en": "Spark", "jp": "スパーク", "pinyin": "hua1huo3_h"},
    "银狼": {"en": "Silver Wolf", "jp": "シルバーウルフ", "pinyin": "yin2lang2"},
    "天童爱丽丝": {"en": "Aris", "jp": "アリス", "pinyin": "tian1tong2ai4li4si1"},
    "早雾": {"en": "Hayiri", "jp": "ハヤヒリ", "pinyin": "zao3wu4"},
    "维里奈": {"en": "Verina", "jp": "ヴェリーナ", "pinyin": "wei2li3nai4"},
    "安可": {"en": "Encore", "jp": "アンコール", "pinyin": "an1ke3"},
    "釉瑚": {"en": "Youhu", "jp": "ユウホ", "pinyin": "you4hu2"},
    "鹿目圆": {"en": "Madoka Kaname", "jp": "鹿目まどか", "pinyin": "lu4mu4yuan2"},
    "晓美焰": {"en": "Homura Akemi", "jp": "暁美ほむら", "pinyin": "xiao3mei3yan4"},
    "血小板": {"en": "Platelet", "jp": "血小板", "pinyin": "xue3xiao3ban3"},
    "雷姆": {"en": "Rem", "jp": "レム", "pinyin": "lei2mu3"},
    "拉姆": {"en": "Ram", "jp": "ラム", "pinyin": "la1mu3"},
    "康娜": {"en": "Kanna", "jp": "カンナ", "pinyin": "kang1na4"},
    "四糸乃": {"en": "Yoshino", "jp": "四糸乃", "pinyin": "si4mi4nai3"},
    "凯露": {"en": "Kyaru", "jp": "キャル", "pinyin": "kai3lu4"},
    "伊莉雅": {"en": "Illya", "jp": "イリヤ", "pinyin": "yi1li4ya3"},
    "忍野忍": {"en": "Oshino Shinobu", "jp": "忍野忍", "pinyin": "ren3ye3ren3"},
    "香风智乃": {"en": "Chino", "jp": "チノ", "pinyin": "xiang1feng1zhi4nai3"},
    "小埋": {"en": "Umaru", "jp": "うまる", "pinyin": "xiao3mai2"},
    "纱雾": {"en": "Sagiri", "jp": "さぎり", "pinyin": "sha1wu4"},
    "猫宫又奈": {"en": "Yanagi", "jp": "ヤナギ", "pinyin": "mao1gong1you4nai4"},
    "德丽莎": {"en": "Theresa", "jp": "テレサ", "pinyin": "de2li4sha1"},
    "布洛妮娅": {"en": "Bronya", "jp": "ブロニア", "pinyin": "bu4luo4ni2ya4"},
    "可琳": {"en": "Kira", "jp": "キラ", "pinyin": "ke3lin2"},
    "神乐": {"en": "Kagura", "jp": "カグラ", "pinyin": "shen2le4"},
    "白上吹雪": {"en": "Shirogane Noel", "jp": "白上ふぶき", "pinyin": "bai2shang4chui1xue3"},
    "月千夜": {"en": "Tsukiyo", "jp": "月千夜", "pinyin": "yue4qian1ye4"},
    "莉塔拉": {"en": "Lita", "jp": "リタ", "pinyin": "li4ta3la1"},
    "维普蕾": {"en": "Viprey", "jp": "ヴィプレイ", "pinyin": "wei2pu3lei3"},
    "夏克里": {"en": "Shakri", "jp": "シャクリ", "pinyin": "xia4ke4li3"},
    "纳甘": {"en": "Nagan", "jp": "ナガン", "pinyin": "na4gan1"},
    "科谢尼娅": {"en": "Koshenia", "jp": "コシェニア", "pinyin": "ke1xie4ni2ya4"},
    "寇尔芙": {"en": "Korvu", "jp": "コルヴ", "pinyin": "kou4er3fu2"},
    "克罗丽科": {"en": "Krokri", "jp": "クロクリ", "pinyin": "ke4luo2li4ke1"},
    "佩里缇亚": {"en": "Peritia", "jp": "ペリティア", "pinyin": "pei4li3ti2ya4"},
    "阿尼亚": {"en": "Anya Forger", "jp": "アーニャ・フォージャー", "pinyin": "a1ni4ya4"},
    "洛茜": {"en": "Rosci", "jp": "ロシ", "pinyin": "luo4qian4"},
    "灶门祢豆子": {"en": "Nezuko Kamado", "jp": "竈門禰豆子", "pinyin": "zao4men2mi2dou4zi"},
    "希儿": {"en": "Seele Vollerei", "jp": "シーレ・ヴォルレライ", "pinyin": "xi1er2"},
    "杏": {"en": "An Marhall", "jp": "アン・マーホール", "pinyin": "xing4"},
    "伊瑟琳": {"en": "Iselin LeviSius", "jp": "イセリン・リーヴィシウス", "pinyin": "yi1se4lin2"},
    "芙兰": {"en": "Fran", "jp": "フラン", "pinyin": "fu2lan2"},
    "菲米莉丝": {"en": "Fimilis", "jp": "フィミリス", "pinyin": "fei1mi3li4si1"},
    "克拉拉": {"en": "Clara", "jp": "クララ", "pinyin": "ke4la1la1"},
    "铃兰": {"en": "Suzuran", "jp": "スズラン", "pinyin": "ling2lan2"},
    "白咲花": {"en": "Shirosaki Hana", "jp": "白咲花", "pinyin": "bai2xiao4hua1"},
    "星野日向": {"en": "Hoshino Hinata", "jp": "星野ひなた", "pinyin": "xing1ye4ri4xiang4"},
    "姬坂乃爱": {"en": "Himesaka Noa", "jp": "姫坂乃愛", "pinyin": "ji1ban3nai4ai4"},
    "种村小依": {"en": "Tanemura Koharu", "jp": "種村小依", "pinyin": "zhong3cun1xiao3yi1"},
    "小之森夏音": {"en": "Konomori Kanon", "jp": "小ノ森夏音", "pinyin": "xiao3zhi1sen1xia4yin1"},
    "雏鹤爱": {"en": "Hinaatsu Ai", "jp": "ひなつあい", "pinyin": "chu2he4ai4"},
    "夜叉神天衣": {"en": "Yashajin Ti", "jp": "夜叉神天衣", "pinyin": "ye4cha1shen2tian1yi1"},
    "空银子": {"en": "Kuonji Gin", "jp": "空銀子", "pinyin": "kong1yin2zi3"},
    "早濑优香": {"en": "Yuka Hayase", "jp": "早瀬優香", "pinyin": "zao3lai4you1xiang1"},
    "一之濑明日奈": {
        "en": "Ichinose Asuna",
        "jp": "一之瀬アスナ",
        "pinyin": "yi1zhi1lai4ming2ri4nai4",
    },
    "空崎日奈": {"en": "Hina Sorasaki", "jp": "空崎ヒナ", "pinyin": "kong1qi2ri4nai4"},
    "圣园未花": {"en": "Mika Misono", "jp": "聖園ミカ", "pinyin": "sheng4yuan2wei4hua1"},
    "小鸟游星野": {
        "en": "Hoshino Tendou",
        "jp": "小鳥遊ホシノ",
        "pinyin": "xiao3niao3you2xing1ye4",
    },
}


def get_role_info(name):
    """获取角色信息"""
    return ROLE_MAPPING.get(name, {"en": name, "jp": name, "pinyin": name})


def get_role_by_pinyin(pinyin):
    """通过拼音获取角色名"""
    for name, info in ROLE_MAPPING.items():
        if info["pinyin"] == pinyin:
            return name
    return None


def get_english_name(name):
    """获取英文名"""
    info = ROLE_MAPPING.get(name)
    return info["en"] if info else name
