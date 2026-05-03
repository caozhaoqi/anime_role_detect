#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""65角色完整拼音映射表 - 中英日文名"""

ROLE_MAPPING = [
    {"chinese": "阿洛娜", "english": "Arona", "japanese": "アロナ", "game": "蔚蓝档案", "pinyin": "a1luo4na4"},
    {"chinese": "普拉娜", "english": "Plana", "japanese": "プラナ", "game": "蔚蓝档案", "pinyin": "pu3la1na4"},
    {"chinese": "纳西妲", "english": "Nahida", "japanese": "ナヒダ", "game": "原神", "pinyin": "na4xi1da2"},
    {"chinese": "缇宝", "english": "Princess", "japanese": "Princess", "game": "崩坏星穹铁道", "pinyin": "ti2bao3"},
    {"chinese": "可莉", "english": "Klee", "japanese": "クレー", "game": "原神", "pinyin": "ke3li4"},
    {"chinese": "迪奥娜", "english": "Diona", "japanese": "ディオナ", "game": "原神", "pinyin": "di2ao4na4"},
    {"chinese": "瑶瑶", "english": "Yaoyao", "japanese": "瑶瑶", "game": "原神", "pinyin": "yao2yao2"},
    {"chinese": "希格雯", "english": "Sigewinne", "japanese": "シーewinne", "game": "原神", "pinyin": "xi1ge2wen2"},
    {"chinese": "蕾贝", "english": "Rebe", "japanese": "レべ", "game": "幻塔", "pinyin": "lei3bei4"},
    {"chinese": "黑塔", "english": "Herta", "japanese": "ヘルタ", "game": "崩坏星穹铁道", "pinyin": "hei1ta3"},
    {"chinese": "符玄", "english": "Fu Xuan", "japanese": "符玄", "game": "崩坏星穹铁道", "pinyin": "fu2xuan2"},
    {"chinese": "七七", "english": "Qiqi", "japanese": "七七", "game": "原神", "pinyin": "qi1qi1"},
    {"chinese": "早柚", "english": "Sayu", "japanese": "小砂", "game": "原神", "pinyin": "zao3you4"},
    {"chinese": "多莉", "english": "Dori", "japanese": "ドリ", "game": "原神", "pinyin": "duo1li4"},
    {"chinese": "卡齐娜", "english": "Kachina", "japanese": "カチナ", "game": "原神", "pinyin": "ka3qi2na4"},
    {"chinese": "三月七", "english": "March 7th", "japanese": "マルセブンス", "game": "崩坏星穹铁道", "pinyin": "san1yue4qi1"},
    {"chinese": "花火", "english": "Sparkle", "japanese": "スパークル", "game": "崩坏星穹铁道", "pinyin": "hua1huo3"},
    {"chinese": "银狼", "english": "Silver Wolf", "japanese": "シルバーワOLF", "game": "崩坏星穹铁道", "pinyin": "yin2lang2"},
    {"chinese": "天童爱丽丝", "english": "Aris", "japanese": "ありす", "game": "蔚蓝档案", "pinyin": "tian1tong2ai4li4si1"},
    {"chinese": "早雾", "english": "Hayiri", "japanese": "ハヤiri", "game": "异环", "pinyin": "zao3wu4"},
    {"chinese": "维里奈", "english": "Verina", "japanese": "ヴェリーナ", "game": "鸣潮", "pinyin": "wei2li3nai4"},
    {"chinese": "安可", "english": "Encore", "japanese": "アンコア", "game": "鸣潮", "pinyin": "an1ke3"},
    {"chinese": "釉壶", "english": "Youhu", "japanese": "ユウホ", "game": "鸣潮", "pinyin": "you4hu2"},
    {"chinese": "洛可可", "english": "Roccia", "japanese": "ロchia", "game": "鸣潮", "pinyin": "luo4ke3ke3"},
    {"chinese": "鹿目圆", "english": "Madoka Kaname", "japanese": "鹿目まどか", "game": "魔法少女小圆", "pinyin": "lu4mu4yuan2"},
    {"chinese": "晓美焰", "english": "Homura Akemi", "japanese": "暁美ほむら", "game": "魔法少女小圆", "pinyin": "xiao3mei3yan4"},
    {"chinese": "血小板", "english": "Platelet", "japanese": "血小板", "game": "工作细胞", "pinyin": "xue4xiao3ban3"},
    {"chinese": "雷姆", "english": "Rem", "japanese": "レム", "game": "Re:从零开始的异世界生活", "pinyin": "lei2mu3"},
    {"chinese": "拉姆", "english": "Ram", "japanese": "ラム", "game": "Re:从零开始的异世界生活", "pinyin": "la1mu3"},
    {"chinese": "康娜", "english": "Kanna", "japanese": "カンナ", "game": "小林家的龙女仆", "pinyin": "kang1na4"},
    {"chinese": "四糸乃", "english": "Yoshino", "japanese": "四糸乃", "game": "约会大作战", "pinyin": "si4mi4nai3"},
    {"chinese": "凯露", "english": "Kyaru", "japanese": "キャル", "game": "公主连接", "pinyin": "kai3lu4"},
    {"chinese": "克萝萝", "english": "Klor", "japanese": "クロロ", "game": "千年之旅", "pinyin": "ke4luo2luo2"},
    {"chinese": "小闪", "english": "Flash", "japanese": "フラッシュ", "game": "千年之旅", "pinyin": "xiao3shan3"},
    {"chinese": "伊莉雅", "english": "Illya", "japanese": "イリー", "game": "魔法少女伊莉雅", "pinyin": "yi1li4ya3"},
    {"chinese": "忍野忍", "english": "Oshino Shinobu", "japanese": "忍野忍", "game": "物语系列", "pinyin": "ren3ye3ren3"},
    {"chinese": "智乃", "english": "Chino", "japanese": "チノ", "game": "请问您今天要来点兔子吗", "pinyin": "zhi4nai3"},
    {"chinese": "小埋", "english": "Tsumugi", "japanese": "埋", "game": "干物妹小埋", "pinyin": "xiao3mai2"},
    {"chinese": "纱雾", "english": "Sagiri", "japanese": "さぎり", "game": "埃罗芒阿老师", "pinyin": "sha1wu4"},
    {"chinese": "猫宫又奈", "english": "Yanagi", "japanese": "ヤなぎ", "game": "绝区零", "pinyin": "mao1gong1you4nai4"},
    {"chinese": "德丽莎", "english": "Theresa", "japanese": "テレサ", "game": "崩坏3", "pinyin": "de2li4sha1"},
    {"chinese": "布洛妮娅", "english": "Bronya", "japanese": "ブロニア", "game": "崩坏3", "pinyin": "bu4luo4ni2ya4"},
    {"chinese": "可琳", "english": "Kira", "japanese": "キラ", "game": "绝区零", "pinyin": "ke3lin2"},
    {"chinese": "爱丽儿", "english": "Ariel", "japanese": "アリエル", "game": "No Game No Life", "pinyin": "ai4li4er3"},
    {"chinese": "神乐", "english": "Kagura", "japanese": "カグラ", "game": "阴阳师", "pinyin": "shen2le4"},
    {"chinese": "白上吹雪", "english": "Shirogane Noel", "japanese": "白上吹雪", "game": "虚拟YouTuber", "pinyin": "bai2shang4chui1xue3"},
    {"chinese": "月千夜", "english": "Tsukiyo", "japanese": "月千夜", "game": "偶像荣耀", "pinyin": "yue4qian1ye4"},
    {"chinese": "芙丽希娅", "english": "Furisia", "japanese": "フルシア", "game": "灵魂潮汐", "pinyin": "fu2li4xi1ya4"},
    {"chinese": "莉塔拉", "english": "Lita", "japanese": "リタ", "game": "少女前线2：追放", "pinyin": "li4ta3la1"},
    {"chinese": "维普蕾", "english": "Viprey", "japanese": "ヴィプレイ", "game": "少女前线2：追放", "pinyin": "wei2pu3lei3"},
    {"chinese": "夏克里", "english": "Shakri", "japanese": "Shakri", "game": "少女前线2：追放", "pinyin": "xia4ke4li3"},
    {"chinese": "纳甘", "english": "Nagan", "japanese": "ナガン", "game": "少女前线2：追放", "pinyin": "na4gan1"},
    {"chinese": "科谢尼娅", "english": "Koshenia", "japanese": "コシェニア", "game": "少女前线2：追放", "pinyin": "ke1xie4ni2ya4"},
    {"chinese": "奇塔", "english": "Kita", "japanese": "キタ", "game": "少女前线2：追放", "pinyin": "qi2ta3"},
    {"chinese": "寇尔芙", "english": "Korvu", "japanese": "コルヴ", "game": "少女前线2：追放", "pinyin": "kou4er3fu2"},
    {"chinese": "克罗丽科", "english": "Krokri", "japanese": "クロクリ", "game": "少女前线2：追放", "pinyin": "ke4luo2li4ke1"},
    {"chinese": "佩里缇亚", "english": "Peritia", "japanese": "ペリシア", "game": "少女前线2：追放", "pinyin": "pei4li3ti2ya4"},
    {"chinese": "阿尼亚", "english": "Anya Forger", "japanese": "アーニャ・フォージャー", "game": "间谍过家家", "pinyin": "a1ni4ya4"},
    {"chinese": "洛茜", "english": "Rosci", "japanese": "ロsci", "game": "明日方舟终末地", "pinyin": "luo4qian4"},
    {"chinese": "祢豆子", "english": "Nezuko Kamado", "japanese": "竈門祢豆子", "game": "鬼灭之刃", "pinyin": "mi3dou4zi5"},
    {"chinese": "希儿", "english": "Seele Vollerei", "japanese": "シール・フォンレーレ", "game": "崩坏学园2", "pinyin": "xi1er3"},
    {"chinese": "杏", "english": "An Makhall", "japanese": "アン・マール", "game": "崩坏学园2", "pinyin": "xing4"},
    {"chinese": "伊瑟琳", "english": "Iselin LeviSius", "japanese": "イコラー・リーヴィシウス", "game": "崩坏学园2", "pinyin": "yi1se4lin2"},
    {"chinese": "芙兰", "english": "Fran", "japanese": "フラン", "game": "崩坏学园2", "pinyin": "fu2lan2"},
    {"chinese": "菲米莉丝", "english": "Fimilis", "japanese": "フィミリス", "game": "崩坏学园2", "pinyin": "fei1mi3li4si1"},
]

def get_all_names(role):
    names = []
    names.append(role["chinese"])
    if role["english"] and role["english"] != "-":
        names.append(role["english"])
    if role["japanese"] and role["japanese"] != "-":
        names.append(role["japanese"])
    names.append(role["pinyin"])
    names.append(f"{role['chinese']} {role['game']}")
    names.append(f"{role['english']} {role['game']}")
    return list(set(names))

if __name__ == "__main__":
    from pathlib import Path
    URL_DIR = Path("spider_image_system/data/img_url")

    existing = {}
    for f in URL_DIR.glob("*_img.txt"):
        role = f.stem.replace("_img", "")
        with open(f, "r", encoding="utf-8") as fp:
            existing[role] = len([l for l in fp if l.strip()])

    print("=" * 80)
    print("📋 65角色完整映射表")
    print("=" * 80)
    print(f"{'序号':<4} {'中文名':<10} {'拼音':<20} {'英文':<20} {'日文':<15} {'URL数':<8} {'状态'}")
    print("-" * 80)

    for i, role in enumerate(ROLE_MAPPING, 1):
        cnt = existing.get(role["pinyin"], 0)
        status = "✅" if cnt >= 100 else ("⚠️" if cnt > 0 else "❌")
        eng = role["english"][:18] if len(role["english"]) > 18 else role["english"]
        jap = role["japanese"][:13] if len(role["japanese"]) > 13 else role["japanese"]
        print(f"{i:<4} {role['chinese']:<10} {role['pinyin']:<20} {eng:<20} {jap:<15} {cnt:<8} {status}")

    print("-" * 80)
    matched = sum(1 for r in ROLE_MAPPING if existing.get(r["pinyin"], 0) > 0)
    sufficient = sum(1 for r in ROLE_MAPPING if existing.get(r["pinyin"], 0) >= 100)
    print(f"\n✅ 已匹配URL: {matched}/65")
    print(f"✅ URL充足(>=100): {sufficient}/65")
    print(f"⚠️ URL不足(<100): {matched - sufficient}/65")
    print(f"❌ 尚未采集: {65 - matched}/65")
