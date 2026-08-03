#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 loli-role-new.txt 的角色数据采集脚本

解析角色列表，使用 Safebooru → Yande.re 多站点采集角色图片，
自动去重、质量过滤、跳过已有足够数据的角色。

Usage:
    python3 scripts/data_collection/collect_loli_roles.py
    python3 scripts/data_collection/collect_loli_roles.py --target 100 --workers 3
    python3 scripts/data_collection/collect_loli_roles.py --dry-run          # 仅预览
    python3 scripts/data_collection/collect_loli_roles.py --character Klee   # 单角色
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
from typing import Dict, List, Optional, Set, Tuple

# ── 项目路径 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
ROLE_FILE = PROJECT_ROOT / "archived" / "auto_spider_img" / "loli-role-new.txt"
FINAL_DATASET_DIR = PROJECT_ROOT / "data" / "final_dataset"
CONFIG_PATH = Path(__file__).parent / "config" / "collection_config.json"

# ── 系列 → Danbooru 系列标签映射 ─────────────────────────
SERIES_TAG_MAP = {
    "蔚蓝档案": "blue_archive",
    "原神": "genshin_impact",
    "崩坏星穹铁道": "honkai:_star_rail",
    "崩坏3": "honkai_impact_3rd",
    "崩坏学园2": "honkai_gakuen",
    "鸣潮": "wuthering_waves",
    "异环": "neverness_to_everness",
    "魔法少女小圆": "mahou_shoujo_madoka_magica",
    "Re:从零开始的异世界生活": "re:zero_kara_hajimeru_isekai_seikatsu",
    "小林家的龙女仆": "kobayashi-san_chi_no_maid_dragon",
    "约会大作战": "date_a_live",
    "公主连接": "princess_connect!",
    "Fate/kaleid liner Prisma Illya": "fate/kaleid_liner",
    "物语系列": "monogatari_(series)",
    "请问您今天要来点兔子吗": "gochuumon_wa_usagi_desu_ka?",
    "干物妹小埋": "himouto!_umaru-chan",
    "埃罗芒阿老师": "eromanga-sensei",
    "间谍过家家": "spy_x_family",
    "明日方舟": "arknights",
    "明日方舟终末地": "arknights:_endfield",
    "阴阳师": "onmyouji",
    "Hololive": "hololive",
    "东方Project": "touhou",
    "少女前线2：追放": "girls'_frontline",
    "工作细胞": "hataraku_saibou",
    "天使降临我身边": "watashi_ni_tenshi_ga_maiorita!",
    "龙王的工作": "ryuuou_no_oshigoto!",
    "偶像荣耀": "idol_pride",
    "绝区零": "zenless_zone_zero",
    "幸运星": "lucky_star",
    "Love Live!": "love_live!",
    "为美好的世界献上祝福": "kono_subarashii_sekai_ni_shukufuku_wo!",
    "悠哉日常大王": "non_non_biyori",
    "你的名字": "kimi_no_na_wa.",
    "声之形": "koe_no_katachi",
}

# ── 特殊角色名映射（中文名 → Danbooru tag）────────────────
CHARACTER_TAG_MAP = {
    "阿洛娜": "arona_(blue_archive)",
    "普拉娜": "plana_(blue_archive)",
    "砂狼白子": "sunaookami_shiroko",
    "伊吹": "ibuki_(blue_archive)",
    "白洲梓": "shirasu_azusa",
    "久田泉奈": "kuda_izuna",
    "阿慈谷日富美": "ajitani_hifumi",
    "天见和香": "amami_nodoka",
    "空崎日奈": "sorasaki_hina",
    "圣园未花": "misono_mika",
    "小鸟游星野": "takanashi_hoshino",
    "天童爱丽丝": "tendou_aris",
    "黑见芹香": "kuromi_serika",
    "十六夜野乃美": "izayoi_nonomi",
    "奥空绫音": "okusora_ayane",
    "黑馆羽留奈": "kurodate_haruna",
    "天雨亚子": "amau_ako",
    "陆八魔爱露": "rikuhachima_aru",
    "浅黄睦月": "asagi_mutsuki",
    "鬼方佳代子": "onikata_kayoko",
    "下江小春": "shimoe_koharu",
    "狐坂若藻": "kosaka_wakamo",
    "纳西妲": "nahida_(genshin_impact)",
    "可莉": "klee_(genshin_impact)",
    "迪奥娜": "diona_(genshin_impact)",
    "瑶瑶": "yaoyao_(genshin_impact)",
    "希格雯": "sigewinne_(genshin_impact)",
    "七七": "qiqi_(genshin_impact)",
    "早柚": "sayu_(genshin_impact)",
    "多莉": "dori_(genshin_impact)",
    "派蒙": "paimon_(genshin_impact)",
    "卡齐娜": "kachina_(genshin_impact)",
    "芙宁娜": "furina_(genshin_impact)",
    "胡桃": "hu_tao_(genshin_impact)",
    "绮良良": "kirara_(genshin_impact)",
    "柯莱": "collei_(genshin_impact)",
    "黑塔": "herta_(honkai:_star_rail)",
    "符玄": "fu_xuan_(honkai:_star_rail)",
    "三月七": "march_7th_(honkai:_star_rail)",
    "花火": "sparkle_(honkai:_star_rail)",
    "银狼": "silver_wolf_(honkai:_star_rail)",
    "克拉拉": "clara_(honkai:_star_rail)",
    "虎克": "hook_(honkai:_star_rail)",
    "云璃": "yunli_(honkai:_star_rail)",
    "缇宝": "tribbie_(honkai:_star_rail)",
    "白露": "bailu_(honkai:_star_rail)",
    "流萤": "firefly_(honkai:_star_rail)",
    "德丽莎": "theresa_apocalypse",
    "布洛妮娅": "bronya_zaychik",
    "格蕾修": "griseo_(honkai_impact)",
    "萝莎莉娅": "rozaliya_(honkai_impact)",
    "莉莉娅": "liliya_(honkai_impact)",
    "希儿": "seele_vollerei",
    "杏": "sin_mal",
    "伊瑟琳": "einstein_(honkai_impact)",
    "芙兰": "fuka_(honkai_impact)",
    "菲米莉丝": "fumilis",
    "维里奈": "verina_(wuthering_waves)",
    "安可": "encore_(wuthering_waves)",
    "釉瑚": "youhu_(wuthering_waves)",
    "蕾姆": "rem_(re:zero)",
    "拉姆": "ram_(re:zero)",
    "碧翠丝": "beatrice_(re:zero)",
    "康娜": "kanna_kamui",
    "伊露露": "ilulu",
    "托尔": "tohru_(maid_dragon)",
    "四糸乃": "yoshino_(date_a_live)",
    "时崎狂三": "tokisaki_kurumi",
    "五河琴里": "itsuka_kotori",
    "夜刀神十香": "yatogami_tohka",
    "凯露": "kyaru_(princess_connect!)",
    "镜华": "kyouka_(princess_connect!)",
    "优妮": "yuni_(princess_connect!)",
    "可可萝": "kokkoro",
    "佩可莉姆": "pecorine",
    "优衣": "yui_(princess_connect!)",
    "真步": "maho_(princess_connect!)",
    "伊莉雅": "illyasviel_von_einzbern",
    "克洛伊": "chloe_von_einzbern",
    "美游": "miyu_edelfelt",
    "忍野忍": "oshino_shinobu",
    "八九寺真宵": "hachikuji_mayoi",
    "香风智乃": "kafuu_chino",
    "香风心爱": "hoto_cocoa",
    "天天座理世": "tedeza_rize",
    "宇治松千夜": "ujimatsu_chiya",
    "桐间纱路": "kirima_syaro",
    "条河麻耶": "joga_maya",
    "小埋": "doma_umaru",
    "纱雾": "izumi_sagiri",
    "阿尼亚": "anya_forger",
    "铃兰": "suzuran_(arknights)",
    "艾雅法拉": "eyjafjalla",
    "迷迭香": "rosmontis",
    "刻俄柏": "ceobe",
    "泡普卡": "popukar",
    "神乐": "kagura_(onmyouji)",
    "白上吹雪": "shirakami_fubuki",
    "芙兰朵露": "flandre_scarlet",
    "蕾米莉亚": "remilia_scarlet",
    "琪露诺": "cirno",
    "古明地恋": "komeiji_koishi",
    "古明地觉": "komeiji_satori",
    "博丽灵梦": "hakurei_reimu",
    "雾雨魔理沙": "kirisame_marisa",
    "血小板": "platelet_(hataraku_saibou)",
    "白咲花": "shirosaki_hana",
    "星野日向": "hoshino_hinata",
    "姬坂乃爱": "himesaka_noa",
    "种村小依": "tanemura_koharu",
    "小之森夏音": "konomori_kanon",
    "雏鹤爱": "hinatsuru_ai",
    "阿库娅": "aqua_(konosuba)",
    "惠惠": "megumin",
    "悠悠": "yunyun_(konosuba)",
    "矢泽妮可": "yazawa_nico",
    "津岛善子": "tsushima_yoshiko",
    "莲华": "miyauchi_renge",
    "宫水三叶": "miyamizu_mitsuha",
    "西宫硝子": "nishimiya_shouko",
    "可琳·威克斯": "corin_wickes",
    "珂蕾妲": "koleda_belobog",
    "柳（猫宫又奈）": "nekomiya_mana",
    "泉此方": "izumi_konata",
    "柊镜": "hiiragi_kagami",
    "柊司": "hiiragi_tsukasa",
    "月千夜": "tsukiyo_(idol_pride)",
    "一之濑明日奈": "ichinose_asuna",
    "莉塔拉": "lita_(girls_frontline)",
    "维普蕾": "vipery_(girls_frontline)",
    "夏克里": "shakri_(girls_frontline)",
    "纳甘": "nagan_(girls_frontline)",
    "科谢尼娅": "koshenia_(girls_frontline)",
    "寇尔芙": "corvu_(girls_frontline)",
    "克罗丽科": "krolik_(girls_frontline)",
    "佩里缇亚": "peritia_(girls_frontline)",
    "早雾": "hayiri",
    "薄荷": "mint_(neverness_to_everness)",
    "鹿目圆": "kaname_madoka",
    "晓美焰": "akemi_homura",
    "巴麻美": "tomoe_mami",
    "美树沙耶香": "miki_sayaka",
    "洛茜": "rosci_(arknights)",
    "夜叉神天衣": "yashajin_teni",
    "空银子": "sora_ginko",
}

# ── 请求头 ───────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://safebooru.org/",
}


def parse_role_file(filepath: str) -> List[Dict]:
    """解析 loli-role-new.txt，返回角色列表"""
    roles = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                chinese_name = parts[0]
                series = parts[1]
                english_name = parts[2]
                japanese_name = parts[3] if len(parts) > 3 else ""
                roles.append({
                    "chinese": chinese_name,
                    "series": series,
                    "english": english_name,
                    "japanese": japanese_name,
                })
    return roles


def get_tag(chinese_name: str) -> Optional[str]:
    """获取角色的 Danbooru/Safebooru 标签"""
    return CHARACTER_TAG_MAP.get(chinese_name)


def _build_existing_cache() -> Dict[str, int]:
    """预构建 final_dataset 目录名→图片数 缓存（大小写不敏感）"""
    cache = {}
    if not FINAL_DATASET_DIR.exists():
        return cache
    for d in FINAL_DATASET_DIR.iterdir():
        if d.is_dir():
            key = d.name.lower()
            count = sum(1 for f in d.iterdir()
                        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
            cache[key] = count
    return cache


def count_existing(dir_name: str, cache: Optional[Dict[str, int]] = None) -> int:
    """统计角色已有图片数量（大小写不敏感），使用缓存加速"""
    if cache is not None:
        return cache.get(dir_name.lower(), 0)
    if not FINAL_DATASET_DIR.exists():
        return 0
    for d in FINAL_DATASET_DIR.iterdir():
        if d.is_dir() and d.name.lower() == dir_name.lower():
            return sum(1 for f in d.iterdir()
                       if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
    return 0


def compute_dir_name(tag: str) -> str:
    """从标签生成目录名（去掉括号中的系列名）"""
    return tag.split("(")[0].strip().rstrip("_")


def fetch_safebooru_posts(tag: str, limit: int = 100) -> List[Dict]:
    """从 Safebooru API 获取帖子列表"""
    encoded = urllib.parse.quote(tag.replace(" ", "_"))
    url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&limit={limit}&tags={encoded}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Safebooru API 请求失败: {e}")
        return []


def download_image(post: Dict, save_dir: str, existing_hashes: Set[str]) -> Tuple[bool, str]:
    """下载单张图片，返回 (成功, 哈希)"""
    try:
        # 构建图片URL
        if "file_url" in post and post["file_url"]:
            url = post["file_url"]
            if not url.startswith("http"):
                url = f"https://safebooru.org/images/{post['directory']}/{post['image']}"
        elif "image" in post:
            url = f"https://safebooru.org/images/{post['directory']}/{post['image']}"
        else:
            return False, ""

        ext = os.path.splitext(url)[1] or ".jpg"
        if ext.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
            ext = ".jpg"

        post_id = post.get("id", str(random.randint(100000, 999999)))
        file_path = os.path.join(save_dir, f"{post_id}{ext}")

        if os.path.exists(file_path):
            return True, ""

        resp = requests.get(url, headers=HEADERS, timeout=(10, 30))
        resp.raise_for_status()
        data = resp.content

        # MD5 去重
        img_hash = hashlib.md5(data).hexdigest()
        if img_hash in existing_hashes:
            return True, img_hash  # 已有，不算新下载

        with open(file_path, "wb") as f:
            f.write(data)

        return True, img_hash
    except Exception:
        return False, ""


def collect_character(tag: str, dir_name: str, target: int = 100, existing: int = 0) -> Tuple[int, int]:
    """采集单个角色图片"""
    save_dir = FINAL_DATASET_DIR / dir_name
    os.makedirs(save_dir, exist_ok=True)

    need = max(0, target - existing)

    if need <= 0:
        print(f"  {dir_name}: 已有 {existing} 张 >= {target}，跳过")
        return 0, existing

    print(f"\n  [{dir_name}] tag={tag}, 已有 {existing}, 需 {need} 张")

    # 计算已有哈希
    existing_hashes = set()
    for f in save_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
            try:
                existing_hashes.add(hashlib.md5(f.read_bytes()).hexdigest())
            except Exception:
                pass

    # 多站点采集
    sites = ["safebooru", "yande.re"]
    downloaded = 0
    failed = 0

    for site in sites:
        if downloaded >= need:
            break

        if site == "safebooru":
            posts = fetch_safebooru_posts(tag, limit=min(need * 3, 100))
            if not posts:
                print(f"    Safebooru: 无结果")
                continue

            print(f"    Safebooru: 找到 {len(posts)} 个帖子")
            for i, post in enumerate(posts):
                if downloaded >= need:
                    break
                success, img_hash = download_image(post, str(save_dir), existing_hashes)
                if success:
                    if img_hash:
                        existing_hashes.add(img_hash)
                        downloaded += 1
                else:
                    failed += 1
                time.sleep(random.uniform(0.3, 0.8))

        elif site == "yande.re":
            # Yande.re 作为备选
            encoded = urllib.parse.quote(tag.replace(" ", "_"))
            url = f"https://yande.re/post.json?tags={encoded}&limit={min(need * 2, 100)}"
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                if resp.status_code == 200:
                    posts = resp.json()
                    print(f"    Yande.re: 找到 {len(posts)} 个帖子")
                    for post in posts:
                        if downloaded >= need:
                            break
                        if "file_url" in post:
                            try:
                                img_url = post["file_url"]
                                ext = os.path.splitext(img_url)[1] or ".jpg"
                                post_id = post.get("id", str(random.randint(100000, 999999)))
                                file_path = save_dir / f"{post_id}{ext}"

                                if file_path.exists():
                                    continue

                                resp2 = requests.get(img_url, headers=HEADERS, timeout=(10, 30))
                                data = resp2.content
                                img_hash = hashlib.md5(data).hexdigest()
                                if img_hash in existing_hashes:
                                    continue
                                with open(file_path, "wb") as f:
                                    f.write(data)
                                existing_hashes.add(img_hash)
                                downloaded += 1
                                time.sleep(0.5)
                            except Exception:
                                failed += 1
            except Exception as e:
                print(f"    Yande.re 请求失败: {e}")

        time.sleep(random.uniform(1.0, 2.0))

    print(f"  {dir_name}: 下载 {downloaded} 张, 失败 {failed} 张, 总计 {existing + downloaded}")
    return downloaded, existing + downloaded


def main():
    parser = argparse.ArgumentParser(description="loli-role 角色图片采集")
    parser.add_argument("--target", type=int, default=100, help="每个角色目标图片数")
    parser.add_argument("--workers", type=int, default=3, help="并发数（暂未使用）")
    parser.add_argument("--character", type=str, help="只采集指定角色（中文名）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际下载")
    parser.add_argument("--skip-existing", type=int, default=100, help="跳过已有>=N张的角色")
    args = parser.parse_args()

    # 1. 解析角色列表
    print("=" * 60)
    print("解析角色列表...")
    roles = parse_role_file(str(ROLE_FILE))
    print(f"共 {len(roles)} 个角色")

    # 2. 过滤
    if args.character:
        roles = [r for r in roles if args.character in r["chinese"]]
        if not roles:
            print(f"未找到角色: {args.character}")
            return

    # 3. 预构建目录缓存
    existing_cache = _build_existing_cache() if not args.dry_run else {}
    print(f"final_dataset 已有 {len(existing_cache)} 个角色目录\n")

    # 4. 构建角色→标签映射
    print("角色标签映射:")
    role_tasks = []
    skipped = 0
    not_found = 0

    for role in roles:
        tag = get_tag(role["chinese"])
        if not tag:
            print(f"  ✗ {role['chinese']} ({role['series']}) — 无标签映射")
            not_found += 1
            continue

        dir_name = compute_dir_name(tag)
        existing = count_existing(dir_name, existing_cache) if not args.dry_run else 0

        if existing >= args.skip_existing:
            skipped += 1
            print(f"  ⊘ {role['chinese']} → {tag} ({dir_name}): 已有 {existing} 张，跳过")
            continue

        role_tasks.append((role, tag, dir_name, existing))
        print(f"  ✓ {role['chinese']} → {tag} ({dir_name}): 已有 {existing} 张")

    print(f"\n统计: 有标签 {len(role_tasks)} 个 | 跳过 {skipped} 个 | 无标签 {not_found} 个")

    if args.dry_run:
        print("\n[Dry-run 模式] 不执行实际下载")
        return

    if not role_tasks:
        print("\n所有角色已有足够数据，无需采集")
        return

    # 4. 开始采集
    os.makedirs(FINAL_DATASET_DIR, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"开始采集 {len(role_tasks)} 个角色 (目标: {args.target} 张/角色)")
    print(f"输出目录: {FINAL_DATASET_DIR}")
    print(f"{'=' * 60}")

    total_downloaded = 0
    total_failed = 0

    for idx, (role, tag, dir_name, existing) in enumerate(role_tasks, 1):
        print(f"\n[{idx}/{len(role_tasks)}] {role['chinese']} ({role['series']})")
        try:
            dl, total = collect_character(tag, dir_name, target=args.target, existing=existing)
            total_downloaded += dl
        except Exception as e:
            print(f"  采集失败: {e}")
            total_failed += 1

        # 角色间延迟
        time.sleep(random.uniform(1.0, 3.0))

    # 5. 汇总
    print(f"\n{'=' * 60}")
    print(f"采集完成")
    print(f"  成功下载: {total_downloaded} 张")
    print(f"  失败角色: {total_failed} 个")

    # 最终统计
    final_counts = {}
    for d in FINAL_DATASET_DIR.iterdir():
        if d.is_dir():
            final_counts[d.name] = len([f for f in d.iterdir()
                                        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')])
    print(f"  final_dataset 最终: {len(final_counts)} 角色, {sum(final_counts.values())} 张")


if __name__ == "__main__":
    main()