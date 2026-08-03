#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版角色数据采集脚本 v2

核心改进：
1. 多标签策略：短名、短名+系列、完整名 依次尝试
2. 多站点：Safebooru → Danbooru → Gelbooru → Yande.re
3. 网络重试：失败自动重试 3 次
4. 标签缓存：记住每个角色命中标签，后续直接使用

Usage:
    python3 scripts/data_collection/collect_v2.py --dry-run
    python3 scripts/data_collection/collect_v2.py --target 100
    python3 scripts/data_collection/collect_v2.py --character 可莉
"""

import os
import sys
import time
import json
import hashlib
import random
import argparse
import urllib.parse
import requests
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Callable

# ── 项目路径 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
ROLE_FILE = PROJECT_ROOT / "archived" / "auto_spider_img" / "loli-role-new.txt"
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"
CACHE_FILE = PROJECT_ROOT / "data" / ".tag_cache.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

# ── 系列标签 ─────────────────────────────────────────────
SERIES_TAG = {
    "蔚蓝档案": "blue_archive", "原神": "genshin_impact",
    "崩坏星穹铁道": "honkai:_star_rail", "崩坏3": "honkai_impact_3rd",
    "崩坏学园2": "honkai_gakuen", "鸣潮": "wuthering_waves",
    "异环": "neverness_to_everness", "魔法少女小圆": "mahou_shoujo_madoka_magica",
    "Re:从零开始的异世界生活": "re:zero", "小林家的龙女仆": "maid_dragon",
    "约会大作战": "date_a_live", "公主连接": "princess_connect!",
    "Fate/kaleid liner Prisma Illya": "fate/kaleid_liner",
    "物语系列": "monogatari", "请问您今天要来点兔子吗": "gochuumon_wa_usagi_desu_ka?",
    "干物妹小埋": "himouto!_umaru-chan", "埃罗芒阿老师": "eromanga-sensei",
    "间谍过家家": "spy_x_family", "明日方舟": "arknights",
    "明日方舟终末地": "arknights:_endfield", "阴阳师": "onmyouji",
    "Hololive": "hololive", "东方Project": "touhou",
    "少女前线2：追放": "girls'_frontline", "工作细胞": "hataraku_saibou",
    "天使降临我身边": "watashi_ni_tenshi_ga_maiorita!",
    "龙王的工作": "ryuuou_no_oshigoto!", "偶像荣耀": "idol_pride",
    "绝区零": "zenless_zone_zero", "幸运星": "lucky_star",
    "Love Live!": "love_live!", "为美好的世界献上祝福": "konosuba",
    "悠哉日常大王": "non_non_biyori", "你的名字": "kimi_no_na_wa.",
    "声之形": "koe_no_katachi",
}

# ── 角色多标签映射（中文名 → [短名, 短名+系列, 完整名, ...]）────
# Safebooru 只接受短名，Danbooru 接受完整名，按优先级排列
CHARACTER_TAGS = {
    "阿洛娜": ["arona", "arona_(blue_archive)"],
    "普拉娜": ["plana", "plana_(blue_archive)"],
    "砂狼白子": ["shiroko", "shiroko_(blue_archive)", "sunaookami_shiroko"],
    "伊吹": ["ibuki", "ibuki_(blue_archive)"],
    "白洲梓": ["azusa", "azusa_(blue_archive)", "shirasu_azusa"],
    "久田泉奈": ["izuna", "izuna_(blue_archive)", "kuda_izuna"],
    "阿慈谷日富美": ["hifumi", "hifumi_(blue_archive)", "ajitani_hifumi"],
    "天见和香": ["nodoka", "nodoka_(blue_archive)", "amami_nodoka"],
    "空崎日奈": ["hina", "hina_(blue_archive)", "sorasaki_hina"],
    "圣园未花": ["mika", "mika_(blue_archive)", "misono_mika"],
    "小鸟游星野": ["hoshino", "hoshino_(blue_archive)", "takanashi_hoshino"],
    "天童爱丽丝": ["aris", "aris_(blue_archive)", "tendou_aris"],
    "黑见芹香": ["serika", "serika_(blue_archive)", "kuromi_serika"],
    "十六夜野乃美": ["nonomi", "nonomi_(blue_archive)", "izayoi_nonomi"],
    "奥空绫音": ["ayane", "ayane_(blue_archive)", "okusora_ayane"],
    "黑馆羽留奈": ["haruna", "haruna_(blue_archive)", "kurodate_haruna"],
    "天雨亚子": ["ako", "ako_(blue_archive)", "amau_ako"],
    "陆八魔爱露": ["aru", "aru_(blue_archive)", "rikuhachima_aru"],
    "浅黄睦月": ["mutsuki", "mutsuki_(blue_archive)", "asagi_mutsuki"],
    "鬼方佳代子": ["kayoko", "kayoko_(blue_archive)", "onikata_kayoko"],
    "下江小春": ["koharu", "koharu_(blue_archive)", "shimoe_koharu"],
    "狐坂若藻": ["wakamo", "wakamo_(blue_archive)", "kosaka_wakamo"],
    "纳西妲": ["nahida", "nahida_(genshin_impact)"],
    "可莉": ["klee", "klee_(genshin_impact)"],
    "迪奥娜": ["diona", "diona_(genshin_impact)"],
    "瑶瑶": ["yaoyao", "yaoyao_(genshin_impact)"],
    "希格雯": ["sigewinne", "sigewinne_(genshin_impact)"],
    "七七": ["qiqi", "qiqi_(genshin_impact)"],
    "早柚": ["sayu", "sayu_(genshin_impact)"],
    "多莉": ["dori", "dori_(genshin_impact)"],
    "派蒙": ["paimon", "paimon_(genshin_impact)"],
    "卡齐娜": ["kachina", "kachina_(genshin_impact)"],
    "芙宁娜": ["furina", "furina_(genshin_impact)"],
    "胡桃": ["hu_tao", "hu_tao_(genshin_impact)"],
    "绮良良": ["kirara", "kirara_(genshin_impact)"],
    "柯莱": ["collei", "collei_(genshin_impact)"],
    "黑塔": ["herta", "herta_(honkai:_star_rail)"],
    "符玄": ["fu_xuan", "fu_xuan_(honkai:_star_rail)"],
    "三月七": ["march_7th", "march_7th_(honkai:_star_rail)"],
    "花火": ["sparkle", "sparkle_(honkai:_star_rail)"],
    "银狼": ["silver_wolf", "silver_wolf_(honkai:_star_rail)"],
    "克拉拉": ["clara", "clara_(honkai:_star_rail)"],
    "虎克": ["hook", "hook_(honkai:_star_rail)"],
    "云璃": ["yunli", "yunli_(honkai:_star_rail)"],
    "缇宝": ["tribbie", "tribbie_(honkai:_star_rail)"],
    "白露": ["bailu", "bailu_(honkai:_star_rail)"],
    "流萤": ["firefly", "firefly_(honkai:_star_rail)"],
    "德丽莎": ["theresa", "theresa_apocalypse"],
    "布洛妮娅": ["bronya", "bronya_zaychik"],
    "格蕾修": ["griseo", "griseo_(honkai_impact)"],
    "萝莎莉娅": ["rozaliya", "rozaliya_(honkai_impact)"],
    "莉莉娅": ["liliya", "liliya_(honkai_impact)"],
    "希儿": ["seele", "seele_vollerei"],
    "杏": ["sin_mal"],
    "伊瑟琳": ["einstein", "einstein_(honkai_impact)"],
    "芙兰": ["fuka", "fuka_(honkai_impact)"],
    "菲米莉丝": ["fumilis"],
    "维里奈": ["verina", "verina_(wuthering_waves)"],
    "安可": ["encore", "encore_(wuthering_waves)"],
    "釉瑚": ["youhu", "youhu_(wuthering_waves)"],
    "蕾姆": ["rem_(re:zero)", "rem"],
    "拉姆": ["ram_(re:zero)", "ram"],
    "碧翠丝": ["beatrice_(re:zero)", "beatrice"],
    "康娜": ["kanna_kamui", "kanna"],
    "伊露露": ["ilulu"],
    "托尔": ["tohru_(maid_dragon)", "tohru"],
    "四糸乃": ["yoshino_(date_a_live)", "yoshino"],
    "时崎狂三": ["kurumi_tokisaki", "tokisaki_kurumi"],
    "五河琴里": ["kotori_itsuka", "itsuka_kotori"],
    "夜刀神十香": ["tohka_yatogami", "yatogami_tohka"],
    "凯露": ["kyaru_(princess_connect!)", "kyaru"],
    "镜华": ["kyouka_(princess_connect!)", "kyouka"],
    "优妮": ["yuni_(princess_connect!)", "yuni"],
    "可可萝": ["kokkoro"],
    "佩可莉姆": ["pecorine"],
    "优衣": ["yui_(princess_connect!)", "yui"],
    "真步": ["maho_(princess_connect!)", "maho"],
    "伊莉雅": ["illyasviel_von_einzbern", "illya"],
    "克洛伊": ["chloe_von_einzbern", "chloe"],
    "美游": ["miyu_edelfelt", "miyu"],
    "忍野忍": ["shinobu_oshino", "oshino_shinobu"],
    "八九寺真宵": ["mayoi_hachikuji", "hachikuji_mayoi"],
    "香风智乃": ["chino_kafuu", "kafuu_chino"],
    "香风心爱": ["cocoa_hoto", "hoto_cocoa"],
    "天天座理世": ["rize_tedeza", "tedeza_rize"],
    "宇治松千夜": ["chiya_ujimatsu", "ujimatsu_chiya"],
    "桐间纱路": ["syaro_kirima", "kirima_syaro"],
    "条河麻耶": ["maya_joga", "joga_maya"],
    "小埋": ["umaru_doma", "doma_umaru"],
    "纱雾": ["sagiri_izumi", "izumi_sagiri"],
    "阿尼亚": ["anya_forger", "anya"],
    "铃兰": ["suzuran_(arknights)", "suzuran"],
    "艾雅法拉": ["eyjafjalla"],
    "迷迭香": ["rosmontis"],
    "刻俄柏": ["ceobe"],
    "泡普卡": ["popukar"],
    "神乐": ["kagura", "kagura_(onmyouji)"],
    "白上吹雪": ["shirakami_fubuki", "fubuki"],
    "芙兰朵露": ["flandre_scarlet", "flandre"],
    "蕾米莉亚": ["remilia_scarlet", "remilia"],
    "琪露诺": ["cirno"],
    "古明地恋": ["komeiji_koishi", "koishi"],
    "古明地觉": ["komeiji_satori", "satori"],
    "博丽灵梦": ["hakurei_reimu", "reimu"],
    "雾雨魔理沙": ["kirisame_marisa", "marisa"],
    "血小板": ["platelet"],
    "白咲花": ["shirosaki_hana", "hana"],
    "星野日向": ["hoshino_hinata", "hinata"],
    "姬坂乃爱": ["himesaka_noa", "noa"],
    "种村小依": ["tanemura_koharu"],
    "小之森夏音": ["konomori_kanon"],
    "雏鹤爱": ["hinatsuru_ai", "ai"],
    "阿库娅": ["aqua_(konosuba)", "aqua"],
    "惠惠": ["megumin"],
    "悠悠": ["yunyun_(konosuba)", "yunyun"],
    "矢泽妮可": ["yazawa_nico", "nico"],
    "津岛善子": ["tsushima_yoshiko", "yoshiko"],
    "莲华": ["miyauchi_renge", "renge"],
    "宫水三叶": ["miyamizu_mitsuha", "mitsuha"],
    "西宫硝子": ["nishimiya_shouko", "shouko"],
    "可琳·威克斯": ["corin_wickes", "corin"],
    "珂蕾妲": ["koleda_belobog", "koleda"],
    "柳（猫宫又奈）": ["nekomiya_mana", "mana"],
    "泉此方": ["izumi_konata", "konata"],
    "柊镜": ["hiiragi_kagami", "kagami"],
    "柊司": ["hiiragi_tsukasa", "tsukasa"],
    "月千夜": ["tsukiyo_(idol_pride)", "tsukiyo"],
    "一之濑明日奈": ["ichinose_asuna", "asuna"],
    "莉塔拉": ["lita_(girls_frontline)", "lita"],
    "维普蕾": ["vipery_(girls_frontline)", "vipery"],
    "夏克里": ["shakri_(girls_frontline)", "shakri"],
    "纳甘": ["nagan_(girls_frontline)", "nagan"],
    "科谢尼娅": ["koshenia_(girls_frontline)", "koshenia"],
    "寇尔芙": ["corvu_(girls_frontline)", "corvu"],
    "克罗丽科": ["krolik_(girls_frontline)", "krolik"],
    "佩里缇亚": ["peritia_(girls_frontline)", "peritia"],
    "早雾": ["hayiri"],
    "薄荷": ["mint_(neverness_to_everness)", "mint"],
    "鹿目圆": ["kaname_madoka", "madoka"],
    "晓美焰": ["akemi_homura", "homura"],
    "巴麻美": ["tomoe_mami", "mami"],
    "美树沙耶香": ["miki_sayaka", "sayaka"],
    "洛茜": ["rosci_(arknights)", "rosci"],
    "夜叉神天衣": ["yashajin_teni", "teni"],
    "空银子": ["sora_ginko", "ginko"],
}

# ── 工具函数 ─────────────────────────────────────────────


def parse_roles() -> List[Dict]:
    roles = []
    with open(ROLE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                roles.append({
                    "chinese": parts[0], "series": parts[1],
                    "english": parts[2], "japanese": parts[3] if len(parts) > 3 else "",
                })
    return roles


def load_tag_cache() -> Dict[str, str]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_tag_cache(cache: Dict[str, str]):
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def count_existing(dir_name: str) -> int:
    if not FINAL_DIR.exists():
        return 0
    for d in FINAL_DIR.iterdir():
        if d.is_dir() and d.name.lower() == dir_name.lower():
            return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    return 0


def compute_dir_name(tag: str) -> str:
    return tag.split("(")[0].strip().rstrip("_")


def load_existing_hashes(save_dir: Path) -> Set[str]:
    hashes = set()
    for f in save_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            try:
                hashes.add(hashlib.md5(f.read_bytes()).hexdigest())
            except Exception:
                pass
    return hashes


# ── 数据源 ───────────────────────────────────────────────


def fetch_safebooru(tag: str, limit: int = 100) -> List[Dict]:
    """Safebooru: 只接受短名标签"""
    encoded = urllib.parse.quote(tag.replace(" ", "_"))
    url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&limit={limit}&tags={encoded}"
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data
            return []
        except requests.exceptions.JSONDecodeError:
            return []  # 空响应，标签不存在
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return []


def fetch_danbooru(tag: str, limit: int = 100) -> List[Dict]:
    """Danbooru: 接受完整标签，无需 API key"""
    encoded = urllib.parse.quote(tag.replace(" ", "_"))
    url = f"https://danbooru.donmai.us/posts.json?tags={encoded}&limit={limit}"
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
            return []
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return []


def fetch_gelbooru(tag: str, limit: int = 100) -> List[Dict]:
    """Gelbooru: 替代源"""
    encoded = urllib.parse.quote(tag.replace(" ", "_"))
    url = f"https://gelbooru.com/index.php?page=dapi&s=post&q=index&json=1&limit={limit}&tags={encoded}"
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if isinstance(data, list):
                        return data
                except Exception:
                    pass
            return []
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return []


def fetch_yande(tag: str, limit: int = 100) -> List[Dict]:
    """Yande.re: 需要短标签"""
    encoded = urllib.parse.quote(tag.replace(" ", "_"))
    url = f"https://yande.re/post.json?tags={encoded}&limit={limit}"
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return []


# ── 图片下载 ──────────────────────────────────────────────


def download_safebooru(post: Dict, save_dir: Path, hashes: Set[str]) -> Tuple[bool, str]:
    try:
        if "file_url" in post and post["file_url"]:
            url = post["file_url"]
            if not url.startswith("http"):
                url = f"https://safebooru.org/images/{post['directory']}/{post['image']}"
        else:
            return False, ""
        ext = os.path.splitext(url)[1] or ".jpg"
        if ext.lower() not in IMAGE_EXTS:
            ext = ".jpg"
        file_path = save_dir / f"{post.get('id', random.randint(100000, 999999))}{ext}"
        if file_path.exists():
            return False, ""
        resp = requests.get(url, headers=HEADERS, timeout=(10, 30))
        data = resp.content
        img_hash = hashlib.md5(data).hexdigest()
        if img_hash in hashes:
            return False, img_hash
        file_path.write_bytes(data)
        return True, img_hash
    except Exception:
        return False, ""


def download_danbooru(post: Dict, save_dir: Path, hashes: Set[str]) -> Tuple[bool, str]:
    try:
        url = post.get("file_url") or post.get("large_file_url")
        if not url:
            return False, ""
        ext = os.path.splitext(url)[1] or ".jpg"
        if ext.lower() not in IMAGE_EXTS:
            ext = ".jpg"
        file_path = save_dir / f"{post.get('id', random.randint(100000, 999999))}{ext}"
        if file_path.exists():
            return False, ""
        resp = requests.get(url, headers=HEADERS, timeout=(10, 30))
        data = resp.content
        img_hash = hashlib.md5(data).hexdigest()
        if img_hash in hashes:
            return False, img_hash
        file_path.write_bytes(data)
        return True, img_hash
    except Exception:
        return False, ""


def download_gelbooru(post: Dict, save_dir: Path, hashes: Set[str]) -> Tuple[bool, str]:
    try:
        url = post.get("file_url")
        if not url:
            return False, ""
        ext = os.path.splitext(url)[1] or ".jpg"
        if ext.lower() not in IMAGE_EXTS:
            ext = ".jpg"
        file_path = save_dir / f"{post.get('id', random.randint(100000, 999999))}{ext}"
        if file_path.exists():
            return False, ""
        resp = requests.get(url, headers=HEADERS, timeout=(10, 30))
        data = resp.content
        img_hash = hashlib.md5(data).hexdigest()
        if img_hash in hashes:
            return False, img_hash
        file_path.write_bytes(data)
        return True, img_hash
    except Exception:
        return False, ""


def download_yande(post: Dict, save_dir: Path, hashes: Set[str]) -> Tuple[bool, str]:
    try:
        url = post.get("file_url")
        if not url:
            return False, ""
        ext = os.path.splitext(url)[1] or ".jpg"
        if ext.lower() not in IMAGE_EXTS:
            ext = ".jpg"
        file_path = save_dir / f"{post.get('id', random.randint(100000, 999999))}{ext}"
        if file_path.exists():
            return False, ""
        resp = requests.get(url, headers=HEADERS, timeout=(10, 30))
        data = resp.content
        img_hash = hashlib.md5(data).hexdigest()
        if img_hash in hashes:
            return False, img_hash
        file_path.write_bytes(data)
        return True, img_hash
    except Exception:
        return False, ""


# ── 采集器 ────────────────────────────────────────────────

SOURCES = [
    ("Safebooru", fetch_safebooru, download_safebooru),
    ("Danbooru", fetch_danbooru, download_danbooru),
    ("Gelbooru", fetch_gelbooru, download_gelbooru),
    ("Yande.re", fetch_yande, download_yande),
]


def find_working_tag(ch_name: str, cache: Dict[str, str]) -> Optional[str]:
    """找到第一个能命中数据的标签"""
    if ch_name in cache:
        return cache[ch_name]

    tags = CHARACTER_TAGS.get(ch_name, [])
    if not tags:
        return None

    for tag in tags:
        for src_name, fetch_fn, _ in SOURCES:
            try:
                posts = fetch_fn(tag, limit=3)
                if posts:
                    cache[ch_name] = tag
                    save_tag_cache(cache)
                    print(f"    标签命中: {tag} (via {src_name})")
                    return tag
            except Exception:
                pass
            time.sleep(0.2)

    return None


def collect_character(ch_name: str, tag: str, dir_name: str, target: int, existing: int) -> Tuple[int, int]:
    """采集单个角色"""
    need = max(0, target - existing)
    if need <= 0:
        return 0, existing

    save_dir = FINAL_DIR / dir_name
    os.makedirs(save_dir, exist_ok=True)
    hashes = load_existing_hashes(save_dir)

    downloaded = 0
    for src_name, fetch_fn, download_fn in SOURCES:
        if downloaded >= need:
            break
        try:
            posts = fetch_fn(tag, limit=min(need * 3, 100))
            if not posts:
                continue
            print(f"    {src_name}: {len(posts)} 个帖子")
            for post in posts:
                if downloaded >= need:
                    break
                success, img_hash = download_fn(post, save_dir, hashes)
                if success and img_hash:
                    hashes.add(img_hash)
                    downloaded += 1
                time.sleep(random.uniform(0.2, 0.5))
        except Exception as e:
            print(f"    {src_name} 错误: {e}")
        time.sleep(random.uniform(0.5, 1.0))

    return downloaded, existing + downloaded


# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="角色数据采集 v2")
    parser.add_argument("--target", type=int, default=100, help="目标图片数")
    parser.add_argument("--character", type=str, help="单角色采集")
    parser.add_argument("--dry-run", action="store_true", help="预览模式")
    parser.add_argument("--skip-existing", type=int, default=100, help="跳过已有>=N的角色")
    args = parser.parse_args()

    roles = parse_roles()
    print(f"角色总数: {len(roles)}")

    if args.character:
        roles = [r for r in roles if args.character in r["chinese"]]
        if not roles:
            print(f"未找到: {args.character}")
            return

    tag_cache = load_tag_cache()
    print(f"标签缓存: {len(tag_cache)} 条")

    # 扫描已有数据
    existing_cache = {}
    if FINAL_DIR.exists():
        for d in FINAL_DIR.iterdir():
            if d.is_dir():
                cnt = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
                existing_cache[d.name.lower()] = cnt

    # 构建任务
    tasks = []
    for role in roles:
        ch_name = role["chinese"]
        if ch_name not in CHARACTER_TAGS:
            continue
        dir_name = compute_dir_name(CHARACTER_TAGS[ch_name][0])
        existing = existing_cache.get(dir_name.lower(), 0)
        if existing >= args.skip_existing:
            continue
        tasks.append((role, dir_name, existing))

    if args.dry_run:
        print(f"\n待采集: {len(tasks)} 个角色")
        for r, dn, ex in tasks[:10]:
            print(f"  {r['chinese']} ({r['series']}) → {dn}: 已有{ex}")
        if len(tasks) > 10:
            print(f"  ... 共 {len(tasks)} 个")
        return

    print(f"\n待采集: {len(tasks)} 个角色")
    os.makedirs(FINAL_DIR, exist_ok=True)

    total_dl = 0
    for idx, (role, dir_name, existing) in enumerate(tasks, 1):
        ch_name = role["chinese"]
        print(f"\n[{idx}/{len(tasks)}] {ch_name} ({role['series']})")

        tag = find_working_tag(ch_name, tag_cache)
        if not tag:
            print(f"  所有标签均无结果，跳过")
            continue

        try:
            dl, total = collect_character(ch_name, tag, dir_name, args.target, existing)
            total_dl += dl
            print(f"  {dir_name}: 下载 {dl} 张, 总计 {total}")
        except Exception as e:
            print(f"  失败: {e}")

        time.sleep(random.uniform(1.0, 3.0))

    print(f"\n采集完成，共下载 {total_dl} 张")


if __name__ == "__main__":
    main()