#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 loli-role-new.txt 生成带真实中文的 role_info.json（2026-08-09）"""
import json
import re

TXT = "archived/auto_spider_img/loli-role-new.txt"
OUT = "src/core/data/role_info.json"

entries = []
with open(TXT, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("铃兰、仙狐"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        cn, anime = parts[0], parts[1]
        en_toks = [parts[2]]
        for t in parts[3:]:
            if re.search(r"[a-zA-Z]", t):
                en_toks.append(t)
            else:
                break
        entries.append((cn, anime, " ".join(en_toks)))


def norm_tokens(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


reg = json.load(open("configs/class_registry_v2.json"))
registry_names = [c["name"] for c in reg["classes"] if c.get("status") == "ACTIVE"]
model_names = list(json.load(open("models/efficientnet_b3_v4/class_to_idx.json")).keys())
all_names = sorted(set(registry_names) | set(model_names))

entry_by_tokens = {}
for cn, anime, en in entries:
    entry_by_tokens.setdefault(frozenset(norm_tokens(en)), (cn, anime, en))


def match(name):
    nt = norm_tokens(name)
    if not nt:
        return None
    exact = entry_by_tokens.get(frozenset(nt))
    if exact:
        return exact
    best = None
    for key, e in entry_by_tokens.items():
        inter = nt & key
        if inter and len(inter) == min(len(nt), len(key)):
            if best is None or (len(nt), len(key)) < (best[0], best[1]):
                best = (len(nt), len(key), e)
    return best[2] if best else None


role_info, matched, unmatched = {}, 0, []
for name in all_names:
    e = match(name)
    if e:
        cn, anime, en = e
        role_info[name] = {"cn": cn, "en": en, "jp": "", "anime": anime}
        matched += 1
    else:
        role_info[name] = {"cn": name, "en": name, "jp": "", "anime": ""}
        unmatched.append(name)

# 手动补全（高置信度知名角色；txt 未收录或名字写法差异）。可按需增改。
MANUAL = {
    "kokomi": ("珊瑚宫心海", "原神"),
    "keqing": ("刻晴", "原神"),
    "mona": ("莫娜", "原神"),
    "noelle": ("诺艾尔", "原神"),
    "fischl": ("菲谢尔", "原神"),
    "amber": ("安柏", "原神"),
    "jean": ("琴", "原神"),
    "lisa": ("丽莎", "原神"),
    "nilou": ("妮露", "原神"),
    "ningguang": ("凝光", "原神"),
    "rosaria": ("罗莎莉亚", "原神"),
    "xiangling": ("香菱", "原神"),
    "yae_miko": ("八重神子", "原神"),
    "yanfei": ("烟绯", "原神"),
    "yoimiya": ("宵宫", "原神"),
    "kujou_sara": ("九条裟罗", "原神"),
    "faruzan": ("珐露珊", "原神"),
    "layla": ("莱依拉", "原神"),
    "arlecchino": ("阿蕾奇诺", "原神"),
    "clorinde": ("克洛琳德", "原神"),
    "charlotte": ("夏洛蒂", "原神"),
    "kafka": ("卡芙卡", "崩坏星穹铁道"),
    "black_swan": ("黑天鹅", "崩坏星穹铁道"),
    "sushang": ("素裳", "崩坏星穹铁道"),
    "tingyun": ("停云", "崩坏星穹铁道"),
    "topaz": ("托帕", "崩坏星穹铁道"),
    "lynx": ("玲可", "崩坏星穹铁道"),
    "yukong": ("驭空", "崩坏星穹铁道"),
    "nishimiya_shouko": ("西宫硝子", "声之形"),
    "tsushima_yoshiko": ("津岛善子", "Love Live!"),
    "hoshimachi_suisei": ("星街彗星", "Hololive"),
    "houshou_marine": ("宝钟玛琳", "Hololive"),
    "yuuka": ("优香", "蔚蓝档案"),
    "illyasviel_von_einzbern": ("伊莉雅", "Fate/kaleid liner Prisma Illya"),
    "lucia_(punishing:_gray_raven)": ("露西亚", "战双帕弥什"),
    "yashajin_teni": ("夜叉神天衣", "龙王的工作"),
}
for name, (cn, anime) in MANUAL.items():
    if name in all_names and name in role_info and role_info[name]["cn"] == name:
        role_info[name] = {"cn": cn, "en": name, "jp": "", "anime": anime}
        matched += 1
        unmatched.remove(name)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(role_info, f, ensure_ascii=False, indent=2)

print(f"matched: {matched}/{len(all_names)} ({matched / len(all_names) * 100:.1f}%)")
print(f"unmatched {len(unmatched)}: {unmatched}")
