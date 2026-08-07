#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0 优先级角色定向采集脚本

目标：37 个 P0 角色补到 total ≥ 40 张
- 长尾 21 类（1-10 张）
- 测试全错 16 类（混淆严重）

策略：
1. 每个角色 2-4 个标签变体（短名 → 短名+系列 → 完整名 → 别名）
2. 多站点：Safebooru → Gelbooru → Yande.re → Danbooru
3. 每个站点翻页采集（最多 5 页）
4. MD5 去重
5. 失败自动重试 3 次

Usage:
    python3 scripts/data_collection/collect_p0.py --dry-run
    python3 scripts/data_collection/collect_p0.py --target 40
    python3 scripts/data_collection/collect_p0.py --target 40 --group longtail
    python3 scripts/data_collection/collect_p0.py --target 40 --group testfail
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
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"
CACHE_FILE = PROJECT_ROOT / "data" / ".tag_cache_p0.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

# ── P0 角色标签映射 ──────────────────────────────────────
# dir_name → [tag_variants in priority order]
P0_ROLES = {
    # ── 长尾 21 类 ──
    "amber": ["amber_(genshin_impact)", "amber"],
    "ellen": ["ellen_(zenless_zone_zero)", "ellen"],
    "evelyn": ["evelyn_(wuthering_waves)", "evelyn"],
    "fischl": ["fischl_(genshin_impact)", "fischl"],
    "hoshimachi_suisei": ["hoshimachi_suisei", "suisei", "suisei_(hololive)"],
    "houshou_marine": ["houshou_marine", "marine_(hololive)", "marine"],
    "jean": ["jean_(genshin_impact)", "jean_gunnhildr", "jean"],
    "layla": ["layla_(genshin_impact)", "layla"],
    "lisa": ["lisa_(genshin_impact)", "lisa"],
    "lumine": ["lumine_(genshin_impact)", "lumine", "traveler_(genshin_impact)"],
    "lynx": ["lynx_(honkai:_star_rail)", "lynx"],
    "pelagia": ["pelagia_(girls'_frontline)", "pelagia"],
    "piper": ["piper_(zenless_zone_zero)", "piper"],
    "rina": ["rina_(zenless_zone_zero)", "rina"],
    "sapphire": ["sapphire"],
    "seth": ["seth_(zenless_zone_zero)", "seth"],
    "yae_miko": ["yae_miko", "yae_miko_(genshin_impact)"],
    "yashajin_teni": ["yashajin_teni", "teni"],
    "youhu": ["youhu_(wuthering_waves)", "youhu"],
    "yuni_(princess_connect!)": ["yuni_(princess_connect!)", "yuni"],
    "yuuka": ["yuuka_(blue_archive)", "hayase_yuuka", "yuuka"],

    # ── 测试全错 16 类 ──
    "Tribbie": ["tribbie_(honkai:_star_rail)", "tribbie"],
    "amami_nodoka": ["amami_nodoka", "nodoka_(blue_archive)", "nodoka"],
    "anya_forger": ["anya_forger", "anya_(spy_x_family)", "anya"],
    "arlecchino": ["arlecchino_(genshin_impact)", "arlecchino", "the_knist"],
    "dunyarzad": ["dunyarzad_(genshin_impact)", "dunyarzad"],
    "einstein": ["einstein_(honkai_impact)", "einstein"],
    "encore": ["encore_(wuthering_waves)", "encore"],
    "fern": ["fern_(sousou_no_frieren)", "fern", "fern_(frieren)"],
    "fuka": ["fuka_(honkai_impact)", "fuka", "fuka_(herrscher_of_ocean)"],
    "jade": ["jade_(honkai:_star_rail)", "jade"],
    "komeiji_satori": ["komeiji_satori", "satori_(touhou)", "satori"],
    "misono_mika": ["misono_mika", "mika_(blue_archive)", "mika"],
    "miyamizu_mitsuha": ["miyamizu_mitsuha", "mitsuha", "mitsuha_(kimi_no_na_wa.)"],
    "theresa_apocalypse": ["theresa_apocalypse", "theresa_(honkai_impact)", "theresa"],
    "tsukiyo": ["tsukiyo_(idol_pride)", "tsukiyo"],
    "verina": ["verina_(wuthering_waves)", "verina"],
    # ── P1/P2 轻度长尾（余力补，非阻塞）──
    "aino": ["aino", "aino_(blue_archive)"],
    "kyaru_(princess_connect!)": ["kyaru_(princess_connect!)", "kyaru"],
    "kafuu_chino": ["kafuu_chino", "chino_(gochuumon_wa_usagi_desu_ka?)", "chino"],
    "hanya": ["hanya_(honkai:_star_rail)", "hanya"],
    "yunjin": ["yunjin_(genshin_impact)", "yunjin"],
    "izumi_sagiri": ["izumi_sagiri", "sagiri_(eromanga_sensei)", "sagiri"],
    "robin": ["robin_(honkai:_star_rail)", "robin"],
    "Bailu": ["bailu_(honkai:_star_rail)", "bailu_(genshin_impact)", "bailu"],
    "Rosmontis": ["rosmontis", "rosmontis_(arknights)"],
    "cyrene": ["cyrene_(honkai:_star_rail)", "cyrene"],
    "lita": ["lita", "lita_(blue_archive)"],
    "popukar": ["popukar", "popukar_(arknights)"],
}

# 分组
LONGTAIL_ROLES = [
    "amber", "ellen", "evelyn", "fischl", "hoshimachi_suisei",
    "houshou_marine", "jean", "layla", "lisa", "lumine",
    "lynx", "pelagia", "piper", "rina", "sapphire",
    "seth", "yae_miko", "yashajin_teni", "youhu",
    "yuni_(princess_connect!)", "yuuka",
]

TESTFAIL_ROLES = [
    "Tribbie", "amami_nodoka", "anya_forger", "arlecchino",
    "dunyarzad", "einstein", "encore", "fern",
    "fuka", "jade", "komeiji_satori", "misono_mika",
    "miyamizu_mitsuha", "theresa_apocalypse", "tsukiyo", "verina",
]

# ── P1/P2 分组（余力补，非阻塞）──
P1P2_ROLES = [
    "aino", "kyaru_(princess_connect!)", "kafuu_chino", "hanya",
    "yunjin", "izumi_sagiri", "robin", "Bailu",
    "Rosmontis", "cyrene", "lita", "popukar",
]


# ── 工具函数 ─────────────────────────────────────────────

def load_tag_cache() -> Dict[str, str]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_tag_cache(cache: Dict[str, str]):
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def count_existing(dir_name: str) -> int:
    """统计角色已有图片数（在 final_dataset 中，忽略大小写匹配目录名）"""
    if not FINAL_DIR.exists():
        return 0
    for d in FINAL_DIR.iterdir():
        if d.is_dir() and d.name.lower() == dir_name.lower():
            return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    return 0


def load_existing_hashes(save_dir: Path) -> Set[str]:
    hashes = set()
    if not save_dir.exists():
        return hashes
    for f in save_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            try:
                hashes.add(hashlib.md5(f.read_bytes()).hexdigest())
            except Exception:
                pass
    return hashes


# ── 数据源（带翻页）──────────────────────────────────────

def fetch_safebooru(tag: str, page: int = 0, limit: int = 100) -> List[Dict]:
    encoded = urllib.parse.quote(tag.replace(" ", "_"))
    url = (f"https://safebooru.org/index.php?page=dapi&s=post&q=index"
           f"&json=1&limit={limit}&pid={page}&tags={encoded}")
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data
            return []
        except requests.exceptions.JSONDecodeError:
            return []
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return []


def fetch_gelbooru(tag: str, page: int = 0, limit: int = 100) -> List[Dict]:
    encoded = urllib.parse.quote(tag.replace(" ", "_"))
    url = (f"https://gelbooru.com/index.php?page=dapi&s=post&q=index"
           f"&json=1&limit={limit}&pid={page}&tags={encoded}")
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if isinstance(data, list):
                        return data
                    # Gelbooru 有时返回 @attributes 包裹
                    if isinstance(data, dict) and "post" in data:
                        return data["post"] if isinstance(data["post"], list) else [data["post"]]
                except Exception:
                    pass
            return []
        except Exception:
            if attempt < 2:
                time.sleep(2)
    return []


def fetch_yande(tag: str, page: int = 0, limit: int = 100) -> List[Dict]:
    encoded = urllib.parse.quote(tag.replace(" ", "_"))
    url = f"https://yande.re/post.json?tags={encoded}&limit={limit}&page={page}"
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


def fetch_danbooru(tag: str, page: int = 0, limit: int = 100) -> List[Dict]:
    encoded = urllib.parse.quote(tag.replace(" ", "_"))
    url = f"https://danbooru.donmai.us/posts.json?tags={encoded}&limit={limit}&page={page}"
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


# ── 图片下载 ──────────────────────────────────────────────

def download_image(url: str, save_dir: Path, post_id: any, hashes: Set[str]) -> Tuple[bool, str]:
    try:
        ext = os.path.splitext(url)[1] or ".jpg"
        if ext.lower() not in IMAGE_EXTS:
            ext = ".jpg"
        file_path = save_dir / f"{post_id}{ext}"
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


def extract_url_safebooru(post: Dict) -> Optional[str]:
    if "file_url" in post and post["file_url"]:
        url = post["file_url"]
        if not url.startswith("http"):
            url = f"https://safebooru.org/images/{post.get('directory','')}/{post.get('image','')}"
        return url
    return None


def extract_url_gelbooru(post: Dict) -> Optional[str]:
    return post.get("file_url")


def extract_url_yande(post: Dict) -> Optional[str]:
    return post.get("file_url")


def extract_url_danbooru(post: Dict) -> Optional[str]:
    return post.get("file_url") or post.get("large_file_url")


# ── 采集逻辑 ─────────────────────────────────────────────

# 顺序：Yande.re 优先（不限流），Safebooru 次之，Gelbooru/Danbooru 兜底
SOURCES = [
    ("Yande.re", fetch_yande, extract_url_yande),
    ("Safebooru", fetch_safebooru, extract_url_safebooru),
    ("Gelbooru", fetch_gelbooru, extract_url_gelbooru),
    ("Danbooru", fetch_danbooru, extract_url_danbooru),
]


def collect_character(dir_name: str, tags: List[str], target: int, existing: int) -> Tuple[int, int]:
    """采集单个角色，跳过探测阶段，直接用每个标签尝试下载"""
    need = max(0, target - existing)
    if need <= 0:
        return 0, existing

    save_dir = FINAL_DIR / dir_name
    os.makedirs(save_dir, exist_ok=True)
    hashes = load_existing_hashes(save_dir)

    downloaded = 0
    max_pages = 3

    # 逐标签尝试，每个标签在所有源上翻页
    for tag in tags:
        if downloaded >= need:
            break

        for src_name, fetch_fn, extract_fn in SOURCES:
            if downloaded >= need:
                break

            for page in range(max_pages):
                if downloaded >= need:
                    break

                try:
                    posts = fetch_fn(tag, page=page, limit=min(need * 2, 100))
                    if not posts:
                        break

                    print(f"    {src_name} [{tag}] p{page}: {len(posts)} posts")

                    for post in posts:
                        if downloaded >= need:
                            break

                        url = extract_fn(post)
                        if not url:
                            continue

                        post_id = post.get("id", random.randint(100000, 999999))
                        success, img_hash = download_image(url, save_dir, post_id, hashes)
                        if success and img_hash:
                            hashes.add(img_hash)
                            downloaded += 1

                        time.sleep(random.uniform(0.1, 0.3))

                except Exception as e:
                    print(f"    {src_name} [{tag}] p{page} 错误: {e}")
                    break

            time.sleep(random.uniform(1.0, 2.0))

    return downloaded, existing + downloaded


# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P0 优先级角色定向采集")
    parser.add_argument("--target", type=int, default=40, help="目标图片数 (默认 40)")
    parser.add_argument("--group", type=str, default="all",
                        choices=["all", "longtail", "testfail", "p1p2"], help="采集分组")
    parser.add_argument("--dry-run", action="store_true", help="预览模式")
    args = parser.parse_args()

    # 选择角色
    if args.group == "longtail":
        roles = LONGTAIL_ROLES
    elif args.group == "testfail":
        roles = TESTFAIL_ROLES
    elif args.group == "p1p2":
        roles = P1P2_ROLES
    else:
        roles = LONGTAIL_ROLES + TESTFAIL_ROLES + P1P2_ROLES

    print(f"P0 角色定向采集 — 目标: {args.target} 张/角色")
    print(f"分组: {args.group} ({len(roles)} 角色)")
    print("=" * 60)

    # 检查已有数据
    tasks = []
    for dir_name in roles:
        existing = count_existing(dir_name)
        tags = P0_ROLES.get(dir_name, [dir_name])
        tasks.append((dir_name, tags, existing))

    # 打印任务列表
    print(f"\n{'角色目录':<35s} {'已有':>5s} {'需补':>5s} {'标签':>5s}")
    print("-" * 60)
    total_need = 0
    for dir_name, tags, existing in tasks:
        need = max(0, args.target - existing)
        total_need += need
        mark = "✅" if existing >= args.target else "❌"
        print(f"  {dir_name:<33s} {existing:>5d} {need:>5d} {len(tags):>5d} {mark}")

    print(f"\n总需补: {total_need} 张")

    if args.dry_run:
        print("\n[dry-run] 预览模式，不执行下载")
        return

    # 开始采集
    total_dl = 0

    for idx, (dir_name, tags, existing) in enumerate(tasks, 1):
        need = max(0, args.target - existing)
        if need <= 0:
            print(f"\n[{idx}/{len(tasks)}] {dir_name}: 已有 {existing} ✅ 跳过")
            continue

        print(f"\n[{idx}/{len(tasks)}] {dir_name}: 已有 {existing}, 需补 {need}")

        try:
            dl, total = collect_character(dir_name, tags, args.target, existing)
            total_dl += dl
            status = "✅" if total >= args.target else "⚠️"
            print(f"  {status} 下载 {dl} 张, 总计 {total}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")

        time.sleep(random.uniform(3.0, 5.0))

    # 最终统计
    print(f"\n{'=' * 60}")
    print(f"采集完成，共下载 {total_dl} 张")
    print(f"\n最终统计:")

    ok = 0
    for dir_name, _, _ in tasks:
        cnt = count_existing(dir_name)
        mark = "✅" if cnt >= args.target else "❌"
        if cnt >= args.target:
            ok += 1
        print(f"  {dir_name:<35s} {cnt:>5d}  {mark}")

    print(f"\n达标: {ok}/{len(tasks)} ({ok/len(tasks)*100:.1f}%)")


if __name__ == "__main__":
    main()
