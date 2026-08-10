#!/usr/bin/env python3
"""
基于 curl 子进程的采集脚本 — 解决 macOS LibreSSL 兼容性问题

特性：
- 使用 curl subprocess 替代 Python requests，规避 SSL 问题
- 多标签变体自动匹配（短名 → 完整名 → 别名）
- 多数据源（Safebooru → Yande.re → Gelbooru）
- 支持翻页
- MD5 去重
- 并发下载控制

Usage:
    python3 scripts/data_collection/collect_curl.py --target 40
    python3 scripts/data_collection/collect_curl.py --target 40 --roles ellen,fischl,jean
"""

import os
import sys
import json
import hashlib
import random
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"
TAG_CACHE = PROJECT_ROOT / "data" / ".tag_cache_curl.json"
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
CURL_TIMEOUT = 30
CURL_CONNECT = 15

# ── 标签映射（按优先级排序，第一个命中的会被缓存） ──
# 每个角色: [优先标签, 备选1, 备选2, ...]
ROLE_TAGS = {
    # 原神
    "amber": ["amber_(genshin_impact)", "amber"],
    "fischl": ["fischl_(genshin_impact)", "fischl"],
    "jean": ["jean_(genshin_impact)", "jean_gunnhildr", "jean"],
    "layla": ["layla_(genshin_impact)", "layla"],
    "lisa": ["lisa_(genshin_impact)", "lisa_minci", "lisa"],
    "lumine": ["lumine_(genshin_impact)", "lumine", "traveler_(genshin_impact)"],
    "yae_miko": ["yae_miko", "yae_miko_(genshin_impact)"],
    "arlecchino": ["arlecchino_(genshin_impact)", "arlecchino", "the_knight"],
    "dunyarzad": ["dunyarzad_(genshin_impact)", "dunyarzad"],

    # 绝区零
    "ellen": ["ellen_(zenless_zone_zero)", "ellen_joe", "eleanor_(zenless_zone_zero)"],
    "evelyn": ["evelyn_(wuthering_waves)", "evelyn_(pgr)", "evelyn"],
    "piper": ["piper_(zenless_zone_zero)", "piper"],
    "rina": ["rina_(zenless_zone_zero)", "rina"],
    "seth": ["seth_(zenless_zone_zero)", "seth"],

    # 星穹铁道
    "lynx": ["lynx_(honkai:_star_rail)", "lynx"],
    "Tribbie": ["tribbie_(honkai:_star_rail)", "tribbie"],
    "jade": ["jade_(honkai:_star_rail)", "jade"],

    # 蓝档案
    "amami_nodoka": ["amami_nodoka", "nodoka_(blue_archive)", "nodoka"],
    "yuuka": ["hayase_yuuka", "yuuka_(blue_archive)", "yuuka"],
    "kafuu_chino": ["chino_kafuu", "chino_(is_the_order_a_rabbit?)", "chino"],

    # Hololive
    "hoshimachi_suisei": ["hoshimachi_suisei", "suisei_(hololive)", "suisei"],
    "houshou_marine": ["houshou_marine", "marine_(hololive)", "marine"],

    # 崩崩崩
    "einstein": ["einstein_(honkai_impact)", "einstein"],
    "fuka": ["fuka_(honkai_impact)", "fuka", "fuka_(herrscher_of_ocean)"],
    "theresa_apocalypse": ["theresa_apocalypse", "theresa_(honkai_impact)", "theresa"],

    # 其他
    "pelagia": ["pelagia_(girls'_frontline)", "pelagia"],
    "sapphire": ["sapphire_(princess_connect!)", "sapphire"],
    "yashajin_teni": ["yashajin_teni", "teni"],
    "youhu": ["youhu_(wuthering_waves)", "youhu"],
    "yuni_(princess_connect!)": ["yuni_(princess_connect!)", "yuni"],
    "anya_forger": ["anya_forger", "anya_(spy_x_family)", "anya"],
    "encore": ["encore_(wuthering_waves)", "encore"],
    "fern": ["fern_(sousou_no_frieren)", "fern", "fern_(frieren)"],
    "komeiji_satori": ["komeiji_satori", "satori_(touhou)", "satori"],
    "misono_mika": ["misono_mika", "mika_(blue_archive)", "mika"],
    "miyamizu_mitsuha": ["miyamizu_mitsuha", "mitsuha", "mitsuha_(kimi_no_na_wa.)"],
    "tsukiyo": ["tsukiyo_(idol_pride)", "tsukiyo"],
    "verina": ["verina_(wuthering_waves)", "verina"],
    "aino": ["aino_(princess_connect!)", "aino"],
    "kyaru_(princess_connect!)": ["kyaru_(princess_connect!)", "kyaru"],
    "hanya": ["hanya_(jujutsu_kaisen)", "hanya"],
    "yunjin": ["yun_jin_(genshin_impact)", "yunjin", "yun_jin"],
    "izumi_sagiri": ["izumi_sagiri", "sagiri_(anohana)", "sagiri"],
    "robin": ["robin_(fire_emblem)", "robin"],
    "Bailu": ["bailu_(honkai:_star_rail)", "bailu"],
    "Rosmontis": ["rosmontis_(arknights)", "rosmontis"],
    "cyrene": ["cyrene_(wuthering_waves)", "cyrene"],
    "lita": ["lita_(honkai_impact)", "lita"],
    "popukar": ["popukar_(honkai_impact)", "popukar"],
}


# ── 工具函数 ──

def curl_get(url: str) -> Optional[object]:
    """用 curl 做 GET 请求，返回解析后的 JSON"""
    try:
        result = subprocess.run(
            ["curl", "-s", "--connect-timeout", str(CURL_CONNECT),
             "--max-time", str(CURL_TIMEOUT),
             "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
             url],
            capture_output=True, text=True, timeout=CURL_TIMEOUT + 5
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return None
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def curl_download(url: str, save_path: str) -> bool:
    """用 curl 下载文件"""
    try:
        result = subprocess.run(
            ["curl", "-s", "--connect-timeout", "15",
             "--max-time", "60",
             "-o", save_path,
             "-H", "User-Agent: Mozilla/5.0",
             url],
            capture_output=True, timeout=65
        )
        return result.returncode == 0 and os.path.getsize(save_path) > 0
    except Exception:
        return False


def load_cache() -> Dict[str, str]:
    if TAG_CACHE.exists():
        return json.loads(TAG_CACHE.read_text())
    return {}


def save_cache(cache: Dict[str, str]):
    TAG_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def count_images(dir_name: str) -> int:
    if not FINAL_DIR.exists():
        return 0
    for d in FINAL_DIR.iterdir():
        if d.is_dir() and d.name.lower() == dir_name.lower():
            return sum(1 for f in d.iterdir()
                       if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    return 0


def load_existing_hashes(save_dir: Path) -> set:
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


# ── 数据源 ──

def fetch_safebooru(tag: str, page: int = 0, limit: int = 100) -> List[Dict]:
    enc = urllib.parse.quote(tag)
    url = (f"https://safebooru.org/index.php?page=dapi&s=post&q=index"
           f"&json=1&limit={limit}&pid={page}&tags={enc}")
    data = curl_get(url)
    if isinstance(data, list):
        return data
    return []


def fetch_yande(tag: str, page: int = 0, limit: int = 100) -> List[Dict]:
    enc = urllib.parse.quote(tag)
    url = f"https://yande.re/post.json?tags={enc}&limit={limit}&page={page}"
    data = curl_get(url)
    if isinstance(data, list):
        return data
    return []


def fetch_gelbooru(tag: str, page: int = 0, limit: int = 100) -> List[Dict]:
    enc = urllib.parse.quote(tag)
    url = (f"https://gelbooru.com/index.php?page=dapi&s=post&q=index"
           f"&json=1&limit={limit}&pid={page}&tags={enc}")
    data = curl_get(url)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "post" in data:
        posts = data["post"]
        return posts if isinstance(posts, list) else [posts]
    return []


# 数据源配置（顺序按命中率排序）
SOURCES = [
    ("Yande.re", fetch_yande),
    ("Safebooru", fetch_safebooru),
    ("Gelbooru", fetch_gelbooru),
]

SOURCE_DOWNLOAD = {
    "Yande.re": lambda p: p.get("file_url", ""),
    "Safebooru": lambda p: p.get("file_url", ""),
    "Gelbooru": lambda p: p.get("file_url", ""),
}


# ── 采集逻辑 ──

def find_best_tag(dir_name: str, tags: List[str], cache: Dict[str, str]) -> Optional[str]:
    """从标签列表中找到第一个有数据的"""
    if dir_name in cache:
        return cache[dir_name]

    for tag in tags:
        for src_name, fetch_fn in SOURCES:
            posts = fetch_fn(tag, page=0, limit=3)
            if posts:
                cache[dir_name] = tag
                save_cache(cache)
                print(f"    命中标签: {tag} (via {src_name}, {len(posts)} posts)")
                return tag
            time.sleep(0.3)

    return None


def download_posts(tag: str, target: int, save_dir: Path, hashes: set) -> int:
    """从所有源下载图片，支持翻页"""
    downloaded = 0
    max_pages = 3

    for src_name, fetch_fn in SOURCES:
        if downloaded >= target:
            break

        for page in range(max_pages):
            if downloaded >= target:
                break

            posts = fetch_fn(tag, page=page, limit=min(target * 2, 100))
            if not posts:
                break

            print(f"    {src_name} p{page}: {len(posts)} posts", end="")

            for post in posts:
                if downloaded >= target:
                    break

                url = SOURCE_DOWNLOAD[src_name](post)
                if not url or not url.startswith("http"):
                    continue

                post_id = post.get("id", random.randint(100000, 999999))
                ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
                if ext.lower() not in IMAGE_EXTS:
                    ext = ".jpg"
                file_path = save_dir / f"{post_id}{ext}"

                if file_path.exists():
                    continue

                if curl_download(url, str(file_path)):
                    try:
                        img_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
                        if img_hash in hashes:
                            file_path.unlink()
                            continue
                        hashes.add(img_hash)
                        downloaded += 1
                        print(".", end="")
                    except Exception:
                        file_path.unlink()

            print(f" -> 下载 {downloaded}")
            time.sleep(0.3)

        time.sleep(1.0)

    return downloaded


# ── 主流程 ──

def main():
    import argparse
    parser = argparse.ArgumentParser(description="基于 curl 的角色采集")
    parser.add_argument("--target", type=int, default=40)
    parser.add_argument("--roles", type=str, default="", help="逗号分隔的角色名")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.roles:
        role_names = [r.strip() for r in args.roles.split(",")]
    else:
        # 默认: 所有不足 40 张的角色
        role_names = []
        for dir_name in os.listdir(str(FINAL_DIR)):
            dp = FINAL_DIR / dir_name
            if not dp.is_dir():
                continue
            cnt = sum(1 for f in dp.iterdir()
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
            if cnt < args.target:
                role_names.append(dir_name)

    print(f"目标: {args.target} 张/角色, 共 {len(role_names)} 个角色")
    print("=" * 60)

    # 任务列表
    tasks = []
    for dir_name in role_names:
        tags = ROLE_TAGS.get(dir_name, [dir_name])
        existing = count_images(dir_name)
        tasks.append((dir_name, tags, existing))

    for dir_name, tags, existing in tasks:
        need = max(0, args.target - existing)
        mark = "✅" if existing >= args.target else "❌"
        print(f"  {dir_name:<30s} {existing:>4d} -> 目标 {args.target}  {mark}")

    if args.dry_run:
        return

    cache = load_cache()
    total_dl = 0

    for idx, (dir_name, tags, existing) in enumerate(tasks, 1):
        need = max(0, args.target - existing)
        if need <= 0:
            print(f"\n[{idx}/{len(tasks)}] {dir_name}: 已有 {existing} ✅")
            continue

        print(f"\n[{idx}/{len(tasks)}] {dir_name}: 已有 {existing}, 需补 {need}")

        save_dir = FINAL_DIR / dir_name
        os.makedirs(save_dir, exist_ok=True)
        hashes = load_existing_hashes(save_dir)

        tag = find_best_tag(dir_name, tags, cache)
        if not tag:
            print(f"  ⚠️ 所有标签均无结果，跳过")
            continue

        try:
            dl = download_posts(tag, need, save_dir, hashes)
            total_dl += dl
            final_cnt = count_images(dir_name)
            mark = "✅" if final_cnt >= args.target else "⚠️"
            print(f"  {mark} 下载 {dl} 张, 总计 {final_cnt}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")

        time.sleep(random.uniform(3.0, 5.0))

    print(f"\n{'=' * 60}")
    print(f"采集完成，共下载 {total_dl} 张")

    # 最终统计
    print(f"\n最终结果:")
    ok = 0
    for dir_name, _, _ in tasks:
        cnt = count_images(dir_name)
        mark = "✅" if cnt >= args.target else "❌"
        if cnt >= args.target:
            ok += 1
        print(f"  {dir_name:<30s} {cnt:>4d}  {mark}")

    print(f"\n达标: {ok}/{len(tasks)} ({ok / len(tasks) * 100:.1f}%)")


if __name__ == "__main__":
    main()
