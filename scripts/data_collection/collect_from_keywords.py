#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合采集脚本：
1. 读取所有 keyword 文件 → 中文角色名列表
2. 中文名 → 英文 Danbooru 标签（内置完整映射 + tag_resolver 回退）
3. 使用 DanbooruMirrorSpider 下载 → final_dataset

Usage:
    python3 scripts/data_collection/collect_from_keywords.py
    python3 scripts/data_collection/collect_from_keywords.py --site safebooru --workers 4
    python3 scripts/data_collection/collect_from_keywords.py --skip-existing  # 跳过已有≥100的角色
"""

import os
import sys
import time
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import OrderedDict

# ── 项目路径 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DANBOORU_DIR = PROJECT_ROOT / "archived" / "spider_image_system" / "src" / "danbooru"

from db_utils import DB
sys.path.insert(0, str(DANBOORU_DIR))
sys.path.insert(0, str(DANBOORU_DIR.parent))

FINAL_DATASET_DIR = PROJECT_ROOT / "data" / "final_dataset"
KEYWORD_DIR = PROJECT_ROOT / "archived" / "auto_spider_img" / "keywords"

# ── 作品名 → Danbooru work tag ────────────────────────────
GAME_TAG_MAP = {
    "genshin":        "genshin_impact",
    "star_rail":      "honkai:_star_rail",
    "honkai3":        "honkai_impact_3rd",
    "blda":           "blue_archive",
    "mc":             "wuthering_waves",
    "zzz":            "zenless_zone_zero",
    "cbjq":           "snowbreak",
    "ht":             "tower_of_fantasy",
}

# keyword 文件名 → game key
KEYWORD_GAME_MAP = {
    "1_genshin_chinese_spider_img_keyword.txt":      "genshin",
    "3_star_rail_chinese_spider_img_keyword.txt":    "star_rail",
    "6_honkai3_chinese_spider_img_keyword.txt":      "honkai3",
    "blda_spider_img_keyword.txt":                   "blda",
    "mc_spider_img_keyword.txt":                     "mc",
    "zzz_spider_img_keyword.txt":                    "zzz",
    "cbjq_spider_img_keyword.txt":                   "cbjq",
    "ht_spider_img_keyword.txt":                     "ht",
    "qlwh_spider_img_keyword.txt":                   None,   # 千恋万花
    "ll_spider_img_keyword.txt":                     None,
    "lsxy_spider_img_keyword.txt":                   None,
}

# ── 中文名 → 英文 Danbooru tag（完整映射） ─────────────────
# 格式: "中文名": "danbooru_tag"
CHARACTER_MAP: Dict[str, str] = {
    # ======== 原神 (genshin_impact) ========
    "琴":           "jean_(genshin_impact)",
    "安柏":         "amber_(genshin_impact)",
    "丽莎":         "lisa_(genshin_impact)",
    "芭芭拉":       "barbara_(genshin_impact)",
    "可莉":         "klee_(genshin_impact)",
    "诺艾尔":       "noelle_(genshin_impact)",
    "菲谢尔":       "fischl_(genshin_impact)",
    "砂糖":         "sucrose_(genshin_impact)",
    "莫娜":         "mona_(genshin_impact)",
    "迪奥娜":       "diona_(genshin_impact)",
    "罗莎莉亚":     "rosaria_(genshin_impact)",
    "优菈":         "eula_(genshin_impact)",
    "闲云":         "xianyun_(genshin_impact)",
    "瑶瑶":         "yaoyao_(genshin_impact)",
    "夜兰":         "yelan_(genshin_impact)",
    "申鹤":         "shenhe_(genshin_impact)",
    "云堇":         "yunjin_(genshin_impact)",
    "北斗":         "beidou_(genshin_impact)",
    "凝光":         "ningguang_(genshin_impact)",
    "香菱":         "xiangling_(genshin_impact)",
    "刻晴":         "keqing_(genshin_impact)",
    "七七":         "qiqi_(genshin_impact)",
    "辛焱":         "xinyan_(genshin_impact)",
    "甘雨":         "ganyu_(genshin_impact)",
    "胡桃":         "hu_tao_(genshin_impact)",
    "烟绯":         "yanfei_(genshin_impact)",
    "神里绫华":     "ayaka_(genshin_impact)",
    "宵宫":         "yoimiya_(genshin_impact)",
    "早柚":         "sayu_(genshin_impact)",
    "雷电将军":     "raiden_shogun_(genshin_impact)",
    "八重神子":     "yae_miko_(genshin_impact)",
    "九条裟罗":     "kujou_sara_(genshin_impact)",
    "珊瑚宫心海":   "kokomi_(genshin_impact)",
    "娜维娅":       "navia_(genshin_impact)",
    "芙宁娜":       "furina_(genshin_impact)",
    "千织":         "chiori_(genshin_impact)",
    "久岐忍":       "kuki_shinobu_(genshin_impact)",
    "珐露珊":       "faruzan_(genshin_impact)",
    "莱依拉":       "layla_(genshin_impact)",
    "妮露":         "nilou_(genshin_impact)",
    "坎蒂丝":       "candace_(genshin_impact)",
    "多莉":         "dori_(genshin_impact)",
    "柯莱":         "collei_(genshin_impact)",
    "绮良良":       "kirara_(genshin_impact)",
    "纳西妲":       "nahida_(genshin_impact)",
    "迪希雅":       "dehya_(genshin_impact)",
    "迪娜泽黛":     "dunyarzad_(genshin_impact)",
    "派蒙":         "paimon_(genshin_impact)",
    "夏沃蕾":       "chevreuse_(genshin_impact)",
    "夏洛蒂":       "charlotte_(genshin_impact)",
    "琳妮特":       "lynette_(genshin_impact)",
    "希格雯":       "sigewinne_(genshin_impact)",
    "克洛琳德":     "clorinde_(genshin_impact)",
    "荧":           "lumine_(genshin_impact)",
    "莉奈娅":       "lyney_(genshin_impact)",      # 林尼
    "玛拉妮":       "mualani_(genshin_impact)",
    "卡齐娜":       "kachina_(genshin_impact)",
    "希诺宁":       "xilonen_(genshin_impact)",
    "玛薇卡":       "mavuika_(genshin_impact)",
    "伊涅芙":       "inefuu_(genshin_impact)",
    "菈乌玛":       "la_uma_(genshin_impact)",     # 推测
    "爱诺":         "aino_(genshin_impact)",        # 推测
    "奈芙尔":       "nefer_(genshin_impact)",       # 推测
    "雅珂达":       "yakoda_(genshin_impact)",      # 推测
    "哥伦比娅":     "columbina_(genshin_impact)",
    "桑多涅":       "sandrone_(genshin_impact)",
    "尼可":         "nico_(genshin_impact)",
    "归终":         "guizhong_(genshin_impact)",
    "留云借风真君": "xianyun_(genshin_impact)",
    "歌尘浪市真君": "ping_(genshin_impact)",
    "埃洛伊":       "aloy_(horizon_zero_dawn)",
    "派蒙":         "paimon_(genshin_impact)",

    # ======== 崩坏：星穹铁道 (honkai:_star_rail) ========
    "艾丝妲":       "asta_(honkai:_star_rail)",
    "三月七":       "march_7th_(honkai:_star_rail)",
    "希露瓦":       "serval_(honkai:_star_rail)",
    "黑塔":         "herta_(honkai:_star_rail)",
    "大丽花":       "dahli_(honkai:_star_rail)",
    "银狼":         "silver_wolf_(honkai:_star_rail)",
    "希儿":         "seele_(honkai:_star_rail)",
    "卡芙卡":       "kafka_(honkai:_star_rail)",
    "素裳":         "sushang_(honkai:_star_rail)",
    "姬子":         "himeko_(honkai:_star_rail)",
    "布洛妮娅":     "bronya_(honkai:_star_rail)",
    "克拉拉":       "clara_(honkai:_star_rail)",
    "佩拉":         "pela_(honkai:_star_rail)",
    "虎克":         "hook_(honkai:_star_rail)",
    "黑天鹅":       "black_swan_(honkai:_star_rail)",
    "花火":         "sparkle_(honkai:_star_rail)",
    "阮梅":         "ruan_mei_(honkai:_star_rail)",
    "娜塔莎":       "natasha_(honkai:_star_rail)",
    "寒鸦":         "hanya_(honkai:_star_rail)",
    "镜流":         "jingliu_(honkai:_star_rail)",
    "雪衣":         "xueyi_(honkai:_star_rail)",
    "黄泉":         "acheron_(honkai:_star_rail)",
    "符玄":         "fu_xuan_(honkai:_star_rail)",
    "白露":         "bailu_(honkai:_star_rail)",
    "霍霍":         "huohuo_(honkai:_star_rail)",
    "玲妮":         "lynx_(honkai:_star_rail)",
    "青雀":         "qingque_(honkai:_star_rail)",
    "停云":         "tingyun_(honkai:_star_rail)",
    "托帕":         "topaz_(honkai:_star_rail)",
    "驭空":         "yukong_(honkai:_star_rail)",
    "流萤":         "firefly_(honkai:_star_rail)",
    "知更鸟":       "robin_(honkai:_star_rail)",
    "缇宝":         "tribbie_(honkai:_star_rail)",
    "真珠":         "pearl_(honkai:_star_rail)",       # 推测
    "火花":         "sparkle_(honkai:_star_rail)",
    "云璃":         "yunli_(honkai:_star_rail)",
    "阿格莱雅":     "aglaea_(honkai:_star_rail)",
    "遐蝶":         "castorice_(honkai:_star_rail)",
    "刻律德菈":     "keller_(honkai:_star_rail)",       # 推测
    "赛飞儿":       "sapphire_(honkai:_star_rail)",     # 推测
    "昔涟":         "cyrene_(honkai:_star_rail)",
    "风堇":         "husband_(honkai:_star_rail)",      # 推测
    "海瑟音":       "hester_(honkai:_star_rail)",       # 推测
    "绯英":         "scarlet_(honkai:_star_rail)",      # 推测
    "虚照":         "void_(honkai:_star_rail)",         # 推测
    "翡翠":         "jade_(honkai:_star_rail)",

    # ======== 崩坏3 (honkai_impact_3rd) ========
    "布洛妮娅·扎伊切克": "bronya_(honkai_impact_3rd)",
    "无量塔姬子":   "himeko_(honkai_impact_3rd)",
    "八重樱":       "yae_sakura_(honkai_impact_3rd)",
    "德丽莎·阿波卡利斯": "theresa_(honkai_impact_3rd)",
    "卡莲·卡斯兰娜": "kallen_(honkai_impact_3rd)",
    "丽塔·洛丝薇瑟": "rita_(honkai_impact_3rd)",
    "希儿·芙乐艾":  "seele_(honkai_impact_3rd)",
    "萝莎莉娅·阿琳": "rozaliya_(honkai_impact_3rd)",
    "莉莉娅·阿琳":  "liliya_(honkai_impact_3rd)",
    "时雨绮罗":     "shigure_kira_(honkai_impact_3rd)",
    "普罗米修斯":   "prometheus_(honkai_impact_3rd)",
    "米丝忒琳·沙尼亚特": "misteln_(honkai_impact_3rd)",
    "苏莎娜":       "susannah_(honkai_impact_3rd)",
    "爱衣·休伯利安Λ": "ai_(honkai_impact_3rd)",
    "李素裳":       "li_sushang_(honkai_impact_3rd)",
    "维尔薇":       "vill-v_(honkai_impact_3rd)",
    "梅比乌斯":     "mobius_(honkai_impact_3rd)",
    "帕朵菲莉丝":   "pardofelis_(honkai_impact_3rd)",
    "阿波尼亚":     "aponia_(honkai_impact_3rd)",
    "伊甸":         "eden_(honkai_impact_3rd)",
    "比安卡·幽兰黛尔·阿塔吉娜": "biana_youlandeier_altjina_(honkai_impact_3rd)",
    "式波·明日香·兰格雷": "asuka_langley_(honkai_impact_3rd)",
    "娜塔莎·希奥拉": "natasha_ciora_(honkai_impact_3rd)",
    "卡萝尔·佩珀":  "carol_pepper_(honkai_impact_3rd)",
    "布洛妮娅":     "bronya_(honkai_impact_3rd)",
    "符华":         "fu_hua_(honkai_impact_3rd)",
    "希儿":         "seele_(honkai_impact_3rd)",
    "格蕾修":       "griseo_(honkai_impact_3rd)",
    "丽塔":         "rita_(honkai_impact_3rd)",
    "爱莉希雅":     "elysia_(honkai_impact_3rd)",
    "琪亚娜":       "kiana_(honkai_impact_3rd)",
    "雷电芽衣":     "raiden_mei_(honkai_impact_3rd)",
    "识之律者":     "herrscher_of_sentience_(honkai_impact_3rd)",
    "雷之律者":     "herrscher_of_thunder_(honkai_impact_3rd)",
    "空之律者":     "herrscher_of_the_void_(honkai_impact_3rd)",
    "死生之律者":   "herrscher_of_death_(honkai_impact_3rd)",        # 推测
    "薪炎之律者":   "herrscher_of_flamescion_(honkai_impact_3rd)",
    "始源之律者":   "herrscher_of_origin_(honkai_impact_3rd)",
    "人之律者":     "herrscher_of_human_(honkai_impact_3rd)",
    "幽兰黛尔":     "durandal_(honkai_impact_3rd)",

    # ======== Blue Archive (blue_archive) ========
    "阿洛娜":       "arona_(blue_archive)",
    "普拉娜":       "plana_(blue_archive)",
    "日奈":         "hina_(blue_archive)",
    "亚子":         "ako_(blue_archive)",
    "伊织":         "iori_(blue_archive)",
    "千夏":         "chinatsu_(blue_archive)",
    "伊吕波":       "iroha_(blue_archive)",
    "阿露":         "aru_(blue_archive)",
    "睦月":         "mutsuki_(blue_archive)",
    "佳代子":       "kayoko_(blue_archive)",
    "遥香":         "haruka_(blue_archive)",
    "晴奈":         "haruna_(blue_archive)",
    "淳子":         "junko_(blue_archive)",
    "明里":         "akari_(blue_archive)",
    "泉":           "izumi_(blue_archive)",
    "枫香":         "fuuka_(blue_archive)",
    "朱莉":         "juri_(blue_archive)",
    "濑名":         "sena_(blue_archive)",
    "惠":           "megumi_(blue_archive)",
    "霞":           "kasumi_(blue_archive)",
    "优香":         "yuuka_(blue_archive)",
    "诺亚":         "noa_(blue_archive)",
    "小雪":         "koyuki_(blue_archive)",
    "尼禄":         "neru_(blue_archive)",
    "明日奈":       "asuna_(blue_archive)",
    "花凛":         "karin_(blue_archive)",
    "朱音":         "akane_(blue_archive)",
    "时":           "toki_(blue_archive)",
    "歌原":         "utaha_(blue_archive)",
    "响":           "hibiki_(blue_archive)",
    "柯托莉":       "kotori_(blue_archive)",
    "柚子":         "yuzu_(blue_archive)",
    "桃井":         "momoi_(blue_archive)",
    "绿":           "midori_(blue_archive)",
    "爱丽丝":       "aris_(blue_archive)",
    "千寻":         "chihiro_(blue_archive)",
    "真纪":         "maki_(blue_archive)",
    "晴":           "haru_(blue_archive)",
    "小玉":         "kotama_(blue_archive)",
    "日鞠":         "hifumi_(blue_archive)",
    "艾米":         "emi_(blue_archive)",
    "菫":           "sumire_(blue_archive)",
    "未花":         "mika_(blue_archive)",
    "渚":           "nagisa_(blue_archive)",
    "鹤城":         "hoshino_(blue_archive)",     # 其实鹤城是 hoshino... 不对
    "小鸟游星野":   "hoshino_(blue_archive)",
    "空崎日奈":     "hina_(blue_archive)",
    "鬼方佳代子":   "kayoko_(blue_archive)",
    "早濑优香":     "yuuka_(blue_archive)",
    "天雨亚子":     "ako_(blue_archive)",
    "陆八魔爱露":   "aru_(blue_archive)",
    "圣园未花":     "mika_(blue_archive)",
    "天童爱丽丝":   "aris_(blue_archive)",
    "杏山和纱":     "kazusa_(blue_archive)",
    "砂狼白子":     "shiroko_(blue_archive)",
    "一之濑明日奈": "asuna_(blue_archive)",

    # ======== 鸣潮 (wuthering_waves) ========
    "秧秧":         "yangyang_(wuthering_waves)",
    "安可":         "encore_(wuthering_waves)",
    "炽霞":         "chixia_(wuthering_waves)",
    "丹瑾":         "danjin_(wuthering_waves)",
    "白芷":         "baizhi_(wuthering_waves)",
    "散华":         "sanhua_(wuthering_waves)",
    "维里奈":       "verina_(wuthering_waves)",
    "今汐":         "jinhsi_(wuthering_waves)",
    "长离":         "changli_(wuthering_waves)",
    "吟霖":         "yinlin_(wuthering_waves)",
    "漂泊者":       "rover_(wuthering_waves)",
    "桃祈":         "taoqi_(wuthering_waves)",
    "鉴心":         "jianxin_(wuthering_waves)",
    "卡提希娅":     "carthya_(wuthering_waves)",
    "爱弥斯":       "aix_(wuthering_waves)",          # 推测
    "绯雪":         "feixue_(wuthering_waves)",       # 推测
    "洛可可":       "roccoco_(wuthering_waves)",
    "菲比":         "fifi_(wuthering_waves)",          # 推测
    "珂莱塔":       "carlotta_(wuthering_waves)",
    "釉瑚":         "youhu_(wuthering_waves)",
    "莫宁":         "moning_(wuthering_waves)",       # 推测
    "洛瑟拉":       "roscella_(wuthering_waves)",     # 推测
    "琳奈":         "linne_(wuthering_waves)",         # 推测
    "守岸人":       "shorekeeper_(wuthering_waves)",
    "达妮娅":       "dania_(wuthering_waves)",         # 推测
    "西格莉卡":     "siglika_(wuthering_waves)",       # 推测
    "丽贝卡":       "rebecca_(wuthering_waves)",       # 推测
    "露西":         "lucy_(wuthering_waves)",
    "弗洛洛":       "phrololo_(wuthering_waves)",
    "嘉贝莉娜":     "gabriel_(wuthering_waves)",       # 推测
    "奥古斯塔":     "augusta_(wuthering_waves)",       # 推测
    "尤诺":         "yuno_(wuthering_waves)",          # 推测
    "赞妮":         "zani_(wuthering_waves)",
    "坎特蕾拉":     "cantarella_(wuthering_waves)",
    "露帕":         "lupa_(wuthering_waves)",
    "夏空":         "xiakong_(wuthering_waves)",       # 推测
    "椿":           "tsubaki_(wuthering_waves)",       # 推测
    "灯灯":         "dengdeng_(wuthering_waves)",

    # ======== 绝区零 (zenless_zone_zero) ========
    "铃":           "ring_(zenless_zone_zero)",
    "零号":         "zero_(zenless_zone_zero)",
    "猫宫又奈":     "nekomiya_(zenless_zone_zero)",
    "猫又":         "nekomata_(zenless_zone_zero)",
    "妮可·德玛拉":  "nicole_(zenless_zone_zero)",
    "格莉丝·霍华德": "grace_(zenless_zone_zero)",
    "珂蕾妲·贝洛伯格": "koleda_(zenless_zone_zero)",
    "柏妮思·怀特":  "burnice_(zenless_zone_zero)",
    "凯撒·金":     "caesar_(zenless_zone_zero)",
    "莱特":         "lycaon_(zenless_zone_zero)",
    "派派·韦尔":    "piper_(zenless_zone_zero)",
    "波可娜·费雷尼": "pomona_(zenless_zone_zero)",     # 推测
    "雨果·维拉德":  "hugo_(zenless_zone_zero)",
    "薇薇安·班希":  "vivian_(zenless_zone_zero)",
    "爱丽丝·泰姆菲尔德": "alice_(zenless_zone_zero)",
    "狛野真斗":     "...(zenless_zone_zero)",
    "卢西娅·艾洛温": "lucia_(zenless_zone_zero)",
    "浮波柚叶":     "...(zenless_zone_zero)",
    "伊德海莉·墨菲": "yideli_(zenless_zone_zero)",    # 推测
    "爱芮":         "erii_(zenless_zone_zero)",        # 推测
    "南宫羽":       "mikuni_(zenless_zone_zero)",      # 推测
    "千夏":         "chinatsu_(zenless_zone_zero)",
    "亚历山德丽娜·莎芭丝缇安": "alexandrina_(zenless_zone_zero)",
    "可琳·威克斯":  "corin_(zenless_zone_zero)",
    "艾莲·乔":     "ellen_(zenless_zone_zero)",
    "冯·莱卡恩":    "von_lycaon_(zenless_zone_zero)",
    "简·杜":       "jane_(zenless_zone_zero)",
    "青衣":         "qingyi_(zenless_zone_zero)",
    "赛斯·洛威尔":  "seth_(zenless_zone_zero)",
    "朱鸢":         "zhu_yuan_(zenless_zone_zero)",
    "星见雅":       "yanagi_(zenless_zone_zero)",
    "苍角":         "sokaku_(zenless_zone_zero)",
    "月城柳":       "yanagi_(zenless_zone_zero)",
    "奥菲丝":       "anby_(zenless_zone_zero)",        # 推测
    "11号":         "soldier_11_(zenless_zone_zero)",
    "扳机":         "trigger_(zenless_zone_zero)",     # 推测
    "橘福福":       "...(zenless_zone_zero)",
    "潘引壶":       "...(zenless_zone_zero)",
    "叶瞬光":       "...(zenless_zone_zero)",
    "仪玄":         "...(zenless_zone_zero)",
    "般岳":         "...(zenless_zone_zero)",
    "琉音":         "...(zenless_zone_zero)",
    "希希芙":       "...(zenless_zone_zero)",

    # ======== 尘白禁区 (snowbreak) ========
    "妮塔":             "nita_(snowbreak)",
    "肴":               "yao_(snowbreak)",
    "瑟瑞斯":           "seres_(snowbreak)",
    "苔丝·科特金":      "tess_(snowbreak)",
    "芙提雅-缄默":      "fritia_(snowbreak)",
    "肴-冬至":          "yao_(snowbreak)",
    "鸣濑晴":           "haru_(snowbreak)",
    "晴-藏锋":          "haru_(snowbreak)",
    "辰星-云篆":        "chenxing_(snowbreak)",
    "凯茜娅·克莱因":    "kasia_(snowbreak)",
    "茉莉安·安德烈奥蒂": "mariya_(snowbreak)",
    "芬妮·戈尔登":      "fenni_(snowbreak)",
    "琴诺":             "qin_(snowbreak)",             # 推测
    "安卡希雅":         "ankaxiya_(snowbreak)",        # 推测
    "里芙":             "lifu_(snowbreak)",            # 推测
    "伊切尔·豹豹":      "yichier_(snowbreak)",        # 推测
    "凯茜娅·蓝闪":      "kasia_(snowbreak)",
    "辰星":             "chenxing_(snowbreak)",
    "恩雅":             "enya_(snowbreak)",
    "芬妮·咎冠":        "fenni_(snowbreak)",
    "猫汐尔·溯影":      "maoxier_(snowbreak)",        # 推测
    "猫汐尔":           "maoxier_(snowbreak)",

    # ======== 幻塔 (tower_of_fantasy) ========
    "夏佐":         "xiazao_(tower_of_fantasy)",       # 推测
    "莎莉":         "shali_(tower_of_fantasy)",        # 推测
    "梅丽尔":       "meriel_(tower_of_fantasy)",       # 推测
    "赛弥尔":       "samir_(tower_of_fantasy)",
    "艾莉丝":       "alice_(tower_of_fantasy)",
    "奈美西斯":     "nemesis_(tower_of_fantasy)",
    "蕾贝":         "rube_(tower_of_fantasy)",         # 推测
    "菲欧娜":       "fiona_(tower_of_fantasy)",
    "卡洛琳":       "caroline_(tower_of_fantasy)",     # 推测
    "伊卡洛斯":     "icarus_(tower_of_fantasy)",       # 推测
    "可可丽特":     "cocoritter_(tower_of_fantasy)",
    "凛夜":         "linye_(tower_of_fantasy)",        # 推测
    "白伶":         "bailing_(tower_of_fantasy)",      # 推测
    "克劳迪娅":     "claudia_(tower_of_fantasy)",
    "海拉":         "haila_(tower_of_fantasy)",        # 推测
    "伊蕾娜":       "yilena_(tower_of_fantasy)",       # 推测
    "薇拉":         "vera_(tower_of_fantasy)",         # 推测
    "弗丽嘉":       "frigg_(tower_of_fantasy)",
    "赛琳娜":       "serena_(tower_of_fantasy)",       # 推测
    "修玛":         "huma_(tower_of_fantasy)",
    "芬璃尔":       "fenrir_(tower_of_fantasy)",

    # ======== 通用/混合 ========
    "初音未来":     "hatsune_miku",
    "芙莉莲":       "frieren_(sousou_no_frieren)",
    "菲伦":         "fern_(sousou_no_frieren)",
    "尤贝尔":       "uvel_(sousou_no_frieren)",       # 推测
    "博丽灵梦":     "hakurei_reimu",
    "雾雨魔理沙":   "marisa_kirisame",
    "芙兰朵露":     "frandre_scarlet",
    "古明地恋":     "komeiji_koishi",
    "后藤一里":     "gouto_hitori_(bocchi_the_rock)",
    "星街彗星":     "hoshimachi_suisei",
    "宝钟玛琳":     "houshou_marine",
    "小埋":         "umaru_(himouto!_umaru-chan)",
    "干物妹":       "umaru_(himouto!_umaru-chan)",
    "工作细胞血小板": "platelet_(hataraku_saibou)",
    "蕾姆":         "rem_(re:zero)",
    "拉姆":         "ram_(re:zero)",
    "犬饲小麦":     "inu_kogal",                      # 推测

    # LoveLive
    "佩佩":         "pepe_(love_live)",               # 推测
    "诺诺":         "nono_(love_live)",               # 推测
    "忒拉拉":       "telara_(love_live)",             # 推测
    "寒悠悠":       "han_youyou_(love_live)",         # 推测
    "夏儿":         "xiar_(love_live)",               # 推测
    "洛卿":         "luoqing_(love_live)",            # 推测

    # 千恋万花
    "朝武芳乃":     "yoshino_(k_o_u_s_h_i_k_i)",
    "常陆茉子":     "maiko_(k_o_u_s_h_i_k_i)",
    "丛雨":         "murasame_(k_o_u_s_h_i_k_i)",
    "蕾娜·列支敦瑙尔": "lena_(k_o_u_s_h_i_k_i)",
    "伊莉雅":       "illya_(fate)",
    "莉格露":       "wriggle_(touhou)",
    "柚子":         "yuzu_(blue_archive)",
    "神崎堇":       "...",

    # 鸣潮 Lite
    "帕姆":         "pam_(honkai:_star_rail)",

    # 崩坏：星穹铁道 - extra
    "希儿":         "seele_(honkai:_star_rail)",       # 已在上面定义, 保留
    "翡翠":         "jade_(honkai:_star_rail)",

    # 阿蕾奇诺
    "阿蕾奇诺":     "arlecchino_(genshin_impact)",
    "仆人":         "arlecchino_(genshin_impact)",
    "杜布拉":       "dubra_(genshin_impact)",          # 推测
    "卡翠娜":       "katrina_(genshin_impact)",        # 推测
    "康士坦丝":     "constance_(genshin_impact)",      # 推测

    # ======== 其他角色 ========
    "埃洛伊":       "aloy_(horizon_zero_dawn)",
    "派蒙":         "paimon_(genshin_impact)",

    # LL补充
    "小春":         "koharu_(blue_archive)",

    # 剩余收集
    "归终":         "guizhong_(genshin_impact)",
    "留云借风真君": "xianyun_(genshin_impact)",
    "歌尘浪市真君": "ping_(genshin_impact)",

    # 猫又(绝区零)
    "猫宫又奈":     "nekomiya_(zenless_zone_zero)",
    "猫又":         "nekomata_(zenless_zone_zero)",

    # ── 英文名变体（spider_img_keyword.txt 中的 English name entries） ──
    "Jean":         "jean_(genshin_impact)",
    "Amber":        "amber_(genshin_impact)",
    "Lisa":         "lisa_(genshin_impact)",
    "Barbara":      "barbara_(genshin_impact)",
    "Klee":         "klee_(genshin_impact)",
    "Noelle":       "noelle_(genshin_impact)",
    "Fischier":     "fischl_(genshin_impact)",
    "Sucrose":      "sucrose_(genshin_impact)",
    "Mona":         "mona_(genshin_impact)",
    "Diona":        "diona_(genshin_impact)",
    "Rosaria":      "rosaria_(genshin_impact)",
    "Eura":         "eula_(genshin_impact)",
    "Aloy":         "aloy_(horizon_zero_dawn)",
    "Xianyun":      "xianyun_(genshin_impact)",
    "Yaoyao":       "yaoyao_(genshin_impact)",
    "Yelan":        "yelan_(genshin_impact)",
    "Shenhe":       "shenhe_(genshin_impact)",
    "Yunjin":       "yunjin_(genshin_impact)",
    "Beidou":       "beidou_(genshin_impact)",
    "Ningguang":    "ningguang_(genshin_impact)",
    "Xiangling":    "xiangling_(genshin_impact)",
    "Keqing":       "keqing_(genshin_impact)",
    "Qi Qi":        "qiqi_(genshin_impact)",
    "Xin Yan":      "xinyan_(genshin_impact)",
    "Gan Yu":       "ganyu_(genshin_impact)",
    "Hu Tao":       "hu_tao_(genshin_impact)",
    "Yanfei":       "yanfei_(genshin_impact)",
    "Ayaka":        "ayaka_(genshin_impact)",
    "Yoimiya":      "yoimiya_(genshin_impact)",
    "Sayu":         "sayu_(genshin_impact)",
    "Raiden Shogun":  "raiden_shogun_(genshin_impact)",
    "Yae Miko":     "yae_miko_(genshin_impact)",
    "Kujou Sara":   "kujou_sara_(genshin_impact)",
    "Kokomi":       "kokomi_(genshin_impact)",
    "Navia":        "navia_(genshin_impact)",
    "Furina":       "furina_(genshin_impact)",
    "Chiori":       "chiori_(genshin_impact)",
    "Kuki Shinobu": "kuki_shinobu_(genshin_impact)",
    "Faruzan":      "faruzan_(genshin_impact)",
    "Layla":        "layla_(genshin_impact)",
    "Nilou":        "nilou_(genshin_impact)",
    "Candace":      "candace_(genshin_impact)",
    "Dori":         "dori_(genshin_impact)",
    "Collei":       "collei_(genshin_impact)",
    "kirara":       "kirara_(genshin_impact)",
    "Nahida":       "nahida_(genshin_impact)",
    "Dehya":        "dehya_(genshin_impact)",
    "Dunyarzad Homayani": "dunyarzad_(genshin_impact)",
    "Paimon":       "paimon_(genshin_impact)",
    "Chevreuse":    "chevreuse_(genshin_impact)",
    "Charlotte":    "charlotte_(genshin_impact)",
    "Lynette":      "lynette_(genshin_impact)",
    "Sigewinne":    "sigewinne_(genshin_impact)",
    "Guizhong":     "guizhong_(genshin_impact)",
    "Pinglaolao":   "ping_(genshin_impact)",
    "Liuyun Jiefeng Zhenjun": "xianyun_(genshin_impact)",
    "Gechen Langshi Zhenjun": "ping_(genshin_impact)",
    "Lumine":       "lumine_(genshin_impact)",
    # 星穹铁道
    "Asta":         "asta_(honkai:_star_rail)",
    "March 7th":    "march_7th_(honkai:_star_rail)",
    "Serval":       "serval_(honkai:_star_rail)",
    "Herta":        "herta_(honkai:_star_rail)",
    "Silver Wolf":  "silver_wolf_(honkai:_star_rail)",
    "Seele":        "seele_(honkai:_star_rail)",
    "Kafka":        "kafka_(honkai:_star_rail)",
    "Su Shang":     "sushang_(honkai:_star_rail)",
    "Himeko":       "himeko_(honkai:_star_rail)",
    "Bronya":       "bronya_(honkai:_star_rail)",
    "Clara":        "clara_(honkai:_star_rail)",
    "Hilva":        "serval_(honkai:_star_rail)",
    "Peira":        "pela_(honkai:_star_rail)",
    "Hook":         "hook_(honkai:_star_rail)",
    "Black Swan":   "black_swan_(honkai:_star_rail)",
    "Sparkle":      "sparkle_(honkai:_star_rail)",
    "Ruan Mei":     "ruan_mei_(honkai:_star_rail)",
    "Natasha":      "natasha_(honkai:_star_rail)",
    "han ya":       "hanya_(honkai:_star_rail)",
    "Jing Liu":     "jingliu_(honkai:_star_rail)",
    "Xue Yi":       "xueyi_(honkai:_star_rail)",
    "Acheron":      "acheron_(honkai:_star_rail)",
    "Fu Xuan":      "fu_xuan_(honkai:_star_rail)",
    "bailu":        "bailu_(honkai:_star_rail)",
    "huohuo":       "huohuo_(honkai:_star_rail)",
    "Lynx":         "lynx_(honkai:_star_rail)",
    "qingque":      "qingque_(honkai:_star_rail)",
    "tingyun":      "tingyun_(honkai:_star_rail)",
    "Topaz":        "topaz_(honkai:_star_rail)",
    "yukong":       "yukong_(honkai:_star_rail)",
    # 崩坏3 Romanized
    "Fuhua":          "fu_hua_(honkai_impact_3rd)",
    "Hire":           "seele_(honkai_impact_3rd)",
    "Geleiu":         "griseo_(honkai_impact_3rd)",
    "Lita":           "rita_(honkai_impact_3rd)",
    "Aili Xiya":      "elysia_(honkai_impact_3rd)",
    "Qiyana":         "kiana_(honkai_impact_3rd)",
    "Lei Dian Meiyi": "raiden_mei_(honkai_impact_3rd)",
    "Shizhi Lǜzhe":   "herrscher_of_sentience_(honkai_impact_3rd)",
    "Lei Zhī Lǜzhě":  "herrscher_of_thunder_(honkai_impact_3rd)",
    "Kōng Zhī Lǜzhě": "herrscher_of_the_void_(honkai_impact_3rd)",
    "Sishēng Zhī Lǜzhě": "herrscher_of_death_(honkai_impact_3rd)",
    "Xīn Yán Zhī Lǜzhě": "herrscher_of_flamescion_(honkai_impact_3rd)",
    "Shǐyuán Zhī Lǜzhě": "herrscher_of_origin_(honkai_impact_3rd)",
    "Rén Zhī Lǜzhě":  "herrscher_of_human_(honkai_impact_3rd)",
    "Bronya Zayiqieke":  "bronya_(honkai_impact_3rd)",
    "Wuliang Ta Jizi":   "himeko_(honkai_impact_3rd)",
    "Bayong Ying":       "yae_sakura_(honkai_impact_3rd)",
    "Delisha Abokalis":  "theresa_(honkai_impact_3rd)",
    "Kalian Kaslanna":   "kallen_(honkai_impact_3rd)",
    "Lita Losu Weiser":  "rita_(honkai_impact_3rd)",
    "Hire Fulei":        "seele_(honkai_impact_3rd)",
    "Luoshaliaya Ailin": "rozaliya_(honkai_impact_3rd)",
    "Lilibia Ailin":     "liliya_(honkai_impact_3rd)",
    "Shiyu Qiula":       "shigure_kira_(honkai_impact_3rd)",
    "Prometheus":        "prometheus_(honkai_impact_3rd)",
    "Misti Lin Shaniate":  "misteln_(honkai_impact_3rd)",
    "Susana":            "susannah_(honkai_impact_3rd)",
    "Ayi XiubolianΛ":    "ai_(honkai_impact_3rd)",
    "Lisu Shang":        "li_sushang_(honkai_impact_3rd)",
    "Weierwei":          "vill-v_(honkai_impact_3rd)",
    "Meibiws":           "mobius_(honkai_impact_3rd)",
    "Paduofelis":        "pardofelis_(honkai_impact_3rd)",
    "Abonia":            "aponia_(honkai_impact_3rd)",
    "Eden":              "eden_(honkai_impact_3rd)",
    "Karol Peipei":      "carol_pepper_(honkai_impact_3rd)",
    "Biana Youlandeier Altjina": "durandal_(honkai_impact_3rd)",
    "Shibo Mingriang Ranglei":   "asuka_langley_(honkai_impact_3rd)",
    "Aixilia":           "seele_(honkai_impact_3rd)",
    "Natasha Xiora":     "natasha_ciora_(honkai_impact_3rd)",
    # 额外
    "巴尔泽布":     "raiden_shogun_(genshin_impact)",
    "艾西莉亚":     "elysia_(honkai_impact_3rd)",
    # ZZZ (bullet·point variants)
    "妮可•德玛拉":  "nicole_(zenless_zone_zero)",
    "格莉丝•霍华德": "grace_(zenless_zone_zero)",
    "珂蕾妲•贝洛伯格": "koleda_(zenless_zone_zero)",
    "柏妮思•怀特":  "burnice_(zenless_zone_zero)",
    "凯撒•金":     "caesar_(zenless_zone_zero)",
    "露西亚娜•奥克希斯•提奥多•德•蒙特夫": "rina_(zenless_zone_zero)",
    "派派•韦尔":    "piper_(zenless_zone_zero)",
    "波可娜•费雷尼": "pomona_(zenless_zone_zero)",
    "雨果•维拉德":  "hugo_(zenless_zone_zero)",
    "薇薇安•班希":  "vivian_(zenless_zone_zero)",
    "爱丽丝・泰姆菲尔德": "alice_(zenless_zone_zero)",
    "卢西娅・艾洛温": "lucia_(zenless_zone_zero)",
    "伊德海莉・墨菲": "yideli_(zenless_zone_zero)",
    "亚历山德丽娜•莎芭丝缇安": "alexandrina_(zenless_zone_zero)",
    "可琳•威克斯":  "corin_(zenless_zone_zero)",
    "艾莲•乔":     "ellen_(zenless_zone_zero)",
    "冯•莱卡恩":    "von_lycaon_(zenless_zone_zero)",
    "简•杜":       "jane_(zenless_zone_zero)",
    "赛斯•洛威尔":  "seth_(zenless_zone_zero)",
    "奥菲丝・马格努森": "anby_(zenless_zone_zero)",
    "鬼火":         "oni_fire_(zenless_zone_zero)",
    "席德":         "sid_(zenless_zone_zero)",
    "芙罗拉":       "flora_(zenless_zone_zero)",
    "老席德":       "old_sid_(zenless_zone_zero)",
    "夏潾":         "xialin_(zenless_zone_zero)",
    "普罗米娅":     "promia_(zenless_zone_zero)",
    "照":           "teri_(zenless_zone_zero)",
    "耀嘉音":       "yaojiayin_(zenless_zone_zero)",
    "伊芙琳•舒瓦利耶": "evelyn_(zenless_zone_zero)",
    "诺姆・霍洛维尔": "norm_(zenless_zone_zero)",
    "维琳娜・艾嘉德": "verina_(zenless_zone_zero)",
    "希希芙":       "sishifu_(zenless_zone_zero)",
    # Tower of Fantasy 补充
    "乌丸":         "karasu_(tower_of_fantasy)",
    "西萝":         "xiluo_(tower_of_fantasy)",
    "四枫院羽":     "shifengyuan_yu_(tower_of_fantasy)",
    "艾达星":       "aida_(tower_of_fantasy)",
    "奈·亚伯拉罕":  "nai_(tower_of_fantasy)",
    "希里":         "xili_(tower_of_fantasy)",
    "银岚":         "yinlan_(tower_of_fantasy)",
    "白月魁":       "baiyuekui_(tower_of_fantasy)",
    "蕾蒂":         "lady_(tower_of_fantasy)",
    "卡穆":         "kamu_(tower_of_fantasy)",
    # Snowbreak 补充
    "威尔·安德森":  "wilson_(snowbreak)",
    "莉兹":         "liz_(snowbreak)",
    "伊芙琳":       "evelyn_(snowbreak)",
    "梅丽莎":       "melissa_(snowbreak)",
    "奥利维亚":     "olivia_(snowbreak)",
    "蕾切尔":       "rachel_(snowbreak)",
    "索菲娅":       "sophia_(snowbreak)",
    "维多利亚":     "victoria_(snowbreak)",
    "艾娃":         "ava_(snowbreak)",
    "琳达":         "linda_(snowbreak)",
    "娜塔莉":       "natalie_(snowbreak)",
    "晴·藏锋":     "haru_(snowbreak)",
    "辰星·云篆":   "chenxing_(snowbreak)",
    "安卡希雅·辉夜": "ankaxiya_(snowbreak)",
    "芙提雅·缄默": "fritia_(snowbreak)",
    "肴·冬至":     "yao_(snowbreak)",
    "里芙·狂猎":   "lifu_(snowbreak)",
    "茉莉安":       "mariya_(snowbreak)",
    "芬妮":         "fenni_(snowbreak)",
    "芙提雅":       "fritia_(snowbreak)",
    "苔丝·魔术师": "tess_(snowbreak)",
    "茉莉安·雨燕": "mariya_(snowbreak)",
    # 千恋万花 补充
    "春希":         "haruki_(k_o_u_s_h_i_k_i)",
    "夏树":         "natsuki_(k_o_u_s_h_i_k_i)",
    "奈·绪方":     "ogata_(k_o_u_s_h_i_k_i)",
    "宫泽谦吾":     "miyazawa_kengo_(k_o_u_s_h_i_k_i)",
    "成濑川奈·美": "narusegawa_(k_o_u_s_h_i_k_i)",
    "成濑川雏子":   "narusegawa_(k_o_u_s_h_i_k_i)",
    # 千恋万花 相模家
    "相模兰":   "sagami_ran_(k_o_u_s_h_i_k_i)",
    "相模绫":   "sagami_aya_(k_o_u_s_h_i_k_i)",
    "相模茜":   "sagami_akane_(k_o_u_s_h_i_k_i)",
    "相模朱":   "sagami_ake_(k_o_u_s_h_i_k_i)",
    "相模枫":   "sagami_kaede_(k_o_u_s_h_i_k_i)",
    "相模桂":   "sagami_katsura_(k_o_u_s_h_i_k_i)",
    "相模椿":   "sagami_tsubaki_(k_o_u_s_h_i_k_i)",
    "相模栞":   "sagami_shiori_(k_o_u_s_h_i_k_i)",
    "相模桧":   "sagami_hinoki_(k_o_u_s_h_i_k_i)",
    "相模樱":   "sagami_sakura_(k_o_u_s_h_i_k_i)",
    "相模桃":   "sagami_momo_(k_o_u_s_h_i_k_i)",
    "相模李":   "sagami_sumomo_(k_o_u_s_h_i_k_i)",
    "相模梅":   "sagami_ume_(k_o_u_s_h_i_k_i)",
    "相模杏":   "sagami_anzu_(k_o_u_s_h_i_k_i)",
    "相模梨":   "sagami_nashi_(k_o_u_s_h_i_k_i)",
    "相模柊":   "sagami_hiiragi_(k_o_u_s_h_i_k_i)",
    "相模棹":   "sagami_sao_(k_o_u_s_h_i_k_i)",
    "相模桀":   "sagami_ketsu_(k_o_u_s_h_i_k_i)",
    "相模槙":   "sagami_maki_(k_o_u_s_h_i_k_i)",
    "相模桜":   "sagami_sakura_(k_o_u_s_h_i_k_i)",
    # 萍姥姥
    "萍姥姥":       "ping_(genshin_impact)",
}

# ── 读取所有 keyword 文件 ─────────────────────────────────
def collect_chinese_names(keyword_dir: str) -> Dict[str, str]:
    """
    读取所有 keyword 文件，返回 {中文名: 来源文件}
    """
    names = OrderedDict()
    keyword_path = Path(keyword_dir)
    if not keyword_path.exists():
        print(f"⚠️ keyword 目录不存在: {keyword_dir}")
        return names

    for f in sorted(keyword_path.glob("*.txt")):
        game_key = None
        for pattern, gk in KEYWORD_GAME_MAP.items():
            if pattern in f.name:
                game_key = gk
                break

        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line not in names:
                    names[line] = f.name
    return names


# ── 全局去重 ────────────────────────────────────────────────
HASH_DB_PATH = PROJECT_ROOT / "data" / "image_hashes.db"


def _load_global_hash_db(db_path: str = None) -> Tuple[set, int]:
    """
    从 RDS MySQL 加载全局去重索引。
    替代原有 SQLite 存储。

    返回: (哈希集合, 记录数)
    """
    try:
        return DB.load_all_hashes()
    except Exception as e:
        print(f"  ⚠️ 从 RDS 加载哈希失败，降级到空集合: {e}")
        return set(), 0


def _append_hashes_to_db(new_hashes: Set[str], role_name: str, db_path: str = None) -> None:
    """
    将新增哈希持久化到 RDS MySQL，保证重启后全局去重基准不丢失。

    使用 INSERT IGNORE 避免重复写入，同时更新角色的文件计数和角色列表。
    替代原有 SQLite 存储。
    """
    try:
        DB.append_hashes(new_hashes, role_name)
    except Exception as e:
        print(f"    ⚠️ 持久化哈希到 RDS 失败: {e}")


# ── 下载函数 ────────────────────────────────────────────────
def _compute_existing_hashes(save_dir: str) -> set:
    """计算目录中现有文件的 sha256 哈希集合，用于本角色内容去重"""
    import hashlib
    hashes = set()
    dir_path = Path(save_dir)
    if not dir_path.exists():
        return hashes
    for f in dir_path.iterdir():
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
            try:
                h = hashlib.sha256(f.read_bytes()).hexdigest()
                hashes.add(h)
            except Exception:
                pass
    return hashes


def download_character(target_tag: str, save_dir: str,
                        max_count: int = 100,
                        global_hashes: Optional[set] = None) -> Tuple[int, int]:
    """
    使用 Spider 系统下载单个角色的图片（直接存入 save_dir，不创建嵌套子目录）。
    下载后通过 sha256 内容去重（本角色 + 全局跨角色），避免重复图片。

    参数:
        global_hashes: 全局去重索引（跨角色），由 _load_global_hash_db() 从 SQLite 加载
    返回 (成功数, 失败数)
    """
    from danbooru_mirror_spider import DanbooruMirrorSpider

    # 用于记录本次新增的哈希，更新全局去重基准
    new_hashes = set()

    # 预计算现有文件哈希
    existing_hashes = _compute_existing_hashes(save_dir)
    # 合并全局去重索引（跨角色）
    if global_hashes:
        old_count = len(existing_hashes)
        existing_hashes.update(global_hashes)
        print(f"   本角色 {old_count} 张 + 全局 {len(global_hashes)} 个哈希 = {len(existing_hashes)} 去重基准")

    spider = DanbooruMirrorSpider(site="safebooru", max_workers=4)

    sites_to_try = ["safebooru", "konachan", "yande.re", "lolibooru", "gelbooru"]

    total_success = 0
    total_fail = 0
    remaining = max_count

    for site in sites_to_try:
        if remaining <= 0:
            break
        print(f"    → 尝试站点: {site}")
        try:
            spider.site = site
            spider.site_info = spider.MIRROR_SITES[site]
            spider.session = spider._create_session()

            # 搜索帖子（不经过 spider.download_character_images，避免嵌套目录）
            tag_name = target_tag.lower().replace(' ', '_')
            tags = tag_name
            if site == 'danbooru':
                tags += ' -rating:explicit -rating:questionable'
            elif site in ('konachan', 'yande.re', 'gelbooru'):
                tags += ' rating:safe'

            posts = spider.get_all_posts(tags, remaining)
            if not posts:
                print(f"      {site}: 未找到匹配图片")
                continue

            # 逐张下载，直接存到 save_dir
            site_success = 0
            site_fail = 0
            download_start = time.time()
            for i, post in enumerate(posts, 1):
                # 单角色总超时检查（5 分钟，超时则跳到下一站点）
                if time.time() - download_start > 300:
                    print(f"      {site}: 下载超时（已下载 {site_success} 张），跳到下一站点")
                    break

                if site_success >= remaining:
                    break

                # 定期打印进度
                if i % 5 == 1 or i == len(posts):
                    elapsed = time.time() - download_start
                    print(f"      下载 [{site_success}/{remaining}] 第{i}张... ({elapsed:.0f}s)")
                    sys.stdout.flush()

                try:
                    image_url = spider.get_image_url(post)
                    if not image_url:
                        site_fail += 1
                        continue

                    ext = image_url.split('.')[-1].lower()
                    if ext not in ('jpg', 'jpeg', 'png', 'webp'):
                        ext = 'jpg'

                    post_id = post.get('id', post.get('md5', f"unknown_{i}"))
                    file_path = Path(save_dir) / f"{post_id}.{ext}"

                    # 检查 post_id 去重
                    if file_path.exists():
                        site_success += 1
                        continue

                    # 下载（带超时和重试）
                    for retry in range(3):
                        try:
                            resp = spider.session.get(image_url, stream=True, timeout=(10, 30))
                            resp.raise_for_status()
                            data = resp.content
                            break
                        except Exception as e:
                            if retry < 2:
                                time.sleep(1)
                                continue
                            raise

                    # 内容去重：sha256 检查
                    img_hash = hashlib.sha256(data).hexdigest()
                    if img_hash in existing_hashes:
                        site_success += 1
                        continue

                    with open(file_path, 'wb') as f:
                        f.write(data)

                    existing_hashes.add(img_hash)
                    new_hashes.add(img_hash)
                    site_success += 1
                except Exception as e:
                    site_fail += 1

            print(f"      {site}: 成功={site_success}, 失败={site_fail}")
            total_success += site_success
            total_fail += site_fail
        except Exception as e:
            print(f"    ⚠️ {site} 失败: {e}")

        remaining = max_count - total_success
        time.sleep(random.uniform(0.5, 1.5))

    # 去重基准更新：把本次新增的哈希带回去 + 持久化到数据库
    if global_hashes is not None and new_hashes:
        global_hashes.update(new_hashes)
        print(f"    去重基准已更新: +{len(new_hashes)} 个新哈希 (累计 {len(global_hashes)})")
        # 持久化到 SQLite，保证重启后不丢失
        role_name = Path(save_dir).name
        _append_hashes_to_db(new_hashes, role_name)

    return total_success, total_fail

def main():
    import argparse

    parser = argparse.ArgumentParser(description="整合采集 - keyword + tag_resolver + DanbooruMirrorSpider")
    parser.add_argument("--site", type=str, default="safebooru", help="首选站点")
    parser.add_argument("--workers", type=int, default=4, help="并发下载线程数")
    parser.add_argument("--max-count", type=int, default=100, help="每个角色目标张数")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已有≥max-count张的角色")
    parser.add_argument("--skip-db-threshold", type=int, default=100,
                        help="从数据库查询，跳过已采集>=此阈值的角色（默认100）")
    parser.add_argument("--skip-training", action="store_true", default=True,
                        help="从数据库查询 training_dataset 中已有的角色并跳过")
    args = parser.parse_args()

    # 确保 final_dataset 目录存在
    os.makedirs(FINAL_DATASET_DIR, exist_ok=True)

    # 1. 读取 keyword 文件
    print("=" * 60)
    print("📋 读取 keyword 文件...")
    chinese_names = collect_chinese_names(str(KEYWORD_DIR))
    print(f"   共 {len(chinese_names)} 个不重复角色")

    # 2. 获取 final_dataset 已有状态（本地扫描 + 数据库查询）
    existing_counts = {}

    # 2a. 从数据库查询已采集角色数据
    if args.skip_db_threshold > 0:
        try:
            print("   从 RDS 查询已采集角色数据...")
            db_rows = DB._fetchall(
                "SELECT role_name, training_count, final_count, total_count FROM role_stats"
            )
            for row in db_rows:
                # 使用 total_count 作为判断依据
                existing_counts[row['role_name']] = row['total_count']
            print(f"   ✅ 数据库查询成功: {len(db_rows)} 个角色")
        except Exception as e:
            print(f"   ⚠️ 数据库查询失败: {e}，回退到本地扫描")

    # 2b. 如果数据库没有，回退到本地扫描
    if not existing_counts:
        for d in FINAL_DATASET_DIR.iterdir():
            if d.is_dir():
                count = len([f for f in d.iterdir()
                             if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')])
                existing_counts[d.name] = count

    # 2c. 统计要跳过的角色（根据阈值）
    skip_threshold = args.skip_db_threshold
    roles_to_skip = {name: count for name, count in existing_counts.items() if count >= skip_threshold}

    print(f"   final_dataset 已有 {len(existing_counts)} 个角色目录，共 {sum(existing_counts.values())} 张")
    if roles_to_skip:
        print(f"   ⚠️ 将跳过 {len(roles_to_skip)} 个角色（>= {skip_threshold} 张）:")
        for name in sorted(roles_to_skip.keys(), key=lambda x: roles_to_skip[x], reverse=True)[:10]:
            print(f"      {name}: {roles_to_skip[name]} 张")
        if len(roles_to_skip) > 10:
            print(f"      ... 及其他 {len(roles_to_skip) - 10} 个角色")

    # 2b. 加载全局去重索引（从 SQLite，无需扫描图片）
    print("加载全局跨角色去重索引...")
    global_hashes, hash_count = _load_global_hash_db()
    print(f"   全局索引: {hash_count} 个唯一哈希（db: {HASH_DB_PATH.name}）")
    print("=" * 60)

    # 3. 逐角色处理
    tag_resolver = None  # 延迟导入，按需解析
    total_done = 0
    total_skip = 0
    total_notfound = 0

    for idx, (chinese_name, source_file) in enumerate(chinese_names.items(), 1):
        print(f"\n[{idx}/{len(chinese_names)}] {chinese_name} (来源: {source_file})")

        # 3a. 解析英文 tag
        target_tag = CHARACTER_MAP.get(chinese_name)
        if not target_tag:
            # 回退: 使用 tag_resolver
            if tag_resolver is None:
                from tag_resolver import DanbooruTagResolver
                tag_resolver = DanbooruTagResolver()
            try:
                resolved = tag_resolver.resolve(chinese_name)
                if resolved:
                    target_tag = resolved
                    print(f"   tag_resolver 解析成功: {target_tag}")
                else:
                    print(f"   ⚠️ 无法解析: {chinese_name}，跳过")
                    total_notfound += 1
                    continue
            except Exception as e:
                print(f"   ⚠️ tag_resolver 异常: {e}，跳过")
                total_notfound += 1
                continue

        # 3b. 目录名（去掉括号内容, 保留纯英文名）
        dir_name = target_tag.split("(")[0].strip().rstrip("_")

        # 3c. 检查是否已满足
        existing = existing_counts.get(dir_name, 0)

        # 3c-1. 检查是否超过数据库阈值（从 training_dataset 等来源已有足够数据）
        if args.skip_db_threshold > 0 and existing >= args.skip_db_threshold:
            print(f"   ⚠️ {dir_name} 已有 {existing} 张 >= {args.skip_db_threshold}（数据库阈值），跳过采集")
            total_skip += 1
            continue

        # 3c-2. 检查是否超过目标数量
        need = args.max_count - existing
        if args.skip_existing and need <= 0:
            print(f"   已有 {existing} ≥ {args.max_count}, 跳过")
            total_skip += 1
            continue
        if need <= 0:
            print(f"   已有 {existing} ≥ {args.max_count}, 跳过")
            total_skip += 1
            continue

        print(f"   tag={target_tag}, dir={dir_name}, need={need}/{args.max_count} (已有{existing})")

        # 3d. 下载
        save_dir = str(FINAL_DATASET_DIR / dir_name)
        os.makedirs(save_dir, exist_ok=True)

        try:
            success, fail = download_character(target_tag, save_dir,
                                                max_count=need,
                                                global_hashes=global_hashes)
            print(f"   ✅ {chinese_name}: 成功={success}, 失败={fail}")
            total_done += 1

            # 1. 写入 RDS 采集记录
            try:
                DB.add_collection_record(
                    role_name=chinese_name,
                    role_tag=target_tag,
                    site=args.site or "",
                    success_count=success,
                    fail_count=fail,
                    total_needed=need,
                    existing_before=existing,
                    new_hashes_added=success,
                )
            except Exception as e:
                print(f"   ⚠️ 写入采集记录失败: {e}")

            # 2. 更新角色统计数据到 role_stats 表
            try:
                # 重新统计该角色的图片数量
                new_total = existing + success

                # 检查是否已存在
                existing_row = DB._fetchone(
                    "SELECT id, training_count, final_count FROM role_stats WHERE role_name = %s",
                    (dir_name,)
                )

                if existing_row:
                    # 更新现有记录
                    DB._execute(
                        "UPDATE role_stats SET final_count=%s, total_count=%s, "
                        "updated_at=NOW() WHERE role_name=%s",
                        (new_total, new_total, dir_name)
                    )
                else:
                    # 插入新记录
                    DB._execute(
                        "INSERT INTO role_stats (role_name, training_count, final_count, "
                        "total_count, skip_threshold) VALUES (%s, %s, %s, %s, %s)",
                        (dir_name, 0, new_total, new_total, 100)
                    )

                print(f"   💾 已同步角色统计: {dir_name} {existing} → {new_total} 张")
            except Exception as e:
                print(f"   ⚠️ 更新角色统计失败: {e}")

        except Exception as e:
            print(f"   ❌ {chinese_name} 下载失败: {e}")

        # 角色间延迟
        time.sleep(random.uniform(1.0, 2.0))

    # 4. 汇总
    print("\n" + "=" * 60)
    print("📊 采集完成汇总")
    print(f"   总角色: {len(chinese_names)}")
    print(f"   已采集/补充: {total_done}")
    print(f"   跳过(已满足): {total_skip}")
    print(f"   未找到标签: {total_notfound}")

    # final state
    final_counts = {}
    for d in FINAL_DATASET_DIR.iterdir():
        if d.is_dir():
            final_counts[d.name] = len([f for f in d.iterdir()
                                        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')])
    print(f"\n   final_dataset 最终: {len(final_counts)} 角色, {sum(final_counts.values())} 张")


if __name__ == "__main__":
    main()