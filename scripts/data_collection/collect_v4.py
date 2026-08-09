#!/usr/bin/env python3
"""
多线程并发采集脚本 v4

优化:
- 多线程并发处理多个角色
- 每角色内下载线程数可调
- 智能标签缓存 + 回退
- 速率限制 + 断点续采

Usage:
    python3 collect_v4.py --target 100
    python3 collect_v4.py --target 100 --roles amber,fischl
"""

import os, sys, json, hashlib, random, subprocess, time, urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

PROJECT_ROOT = Path(__file__).parent.parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"
TAG_CACHE = PROJECT_ROOT / "data" / ".tag_cache_curl.json"
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

CURL_TIMEOUT = 30
CURL_CONNECT = 15
DL_WORKERS = 3
ROLE_WORKERS = 3
PAGE_LIMIT = 100
MAX_PAGES = 3
SOURCE_DELAY = 1.5
REQUEST_DELAY = 0.8

hash_lock = Lock()
print_lock = Lock()


# ── 标签映射 ──
def build_role_tags():
    tags = {}
    genshin = ["amber", "fischl", "jean", "layla", "lisa", "lumine", "yae_miko",
               "arlecchino", "dunyarzad", "yunjin", "furina", "navia", "neuvillette",
               "clorinde", "chiori", "raiden_shogun", "zhongli", "venti", "diluc",
               "alhaitham", "dehya", "ganyu", "hutao", "keqing", "kokomi", "qiqi",
               "xiangling", "nahida", "freminet", "lynette", "lyney", "sigewinne",
               "emilie", "wriothesley", "charlotte", "chevreuse", "collei",
               "tighnari", "cyno", "candace", "nilou", "kaveh", "faruzan",
               "mika", "kinich", "mualani", "xilonen", "chasca", "iori", "gorou",
               "thoma"]
    for r in genshin:
        tags[r] = [f"{r}_(genshin_impact)", r]

    zz0 = ["ellen", "evelyn", "piper", "rina", "seth", "youhu", "verina",
           "encore", "cyrene", "fuka"]
    for r in zz0:
        tags[r] = [f"{r}_(zenless_zone_zero)", f"{r}_(wuthering_waves)", r]

    starrail = ["lynx", "Tribbie", "jade", "Bailu", "sparkle", "hanabi",
                "black_swan", "hanya"]
    for r in starrail:
        tags[r] = [f"{r}_(honkai:_star_rail)", r]

    blue_archive = ["amami_nodoka", "yuuka", "kafuu_chino", "misono_mika",
                   "yoshino", "hifumi", "hanako", "sorasaki_hina", "sunaookami_shiroko"]
    for r in blue_archive:
        tags[r] = [f"{r}", f"{r}_(blue_archive)"]

    hololive = ["hoshimachi_suisei", "houshou_marine"]
    for r in hololive:
        tags[r] = [r, f"{r}_(hololive)"]

    honkai = ["einstein", "tesla", "newton", "darwin", "plato", "aristotle",
              "madame_A", "theresa_apocalypse", "lita", "popukar", "griseo"]
    for r in honkai:
        tags[r] = [f"{r}_(honkai_impact)", r]

    fate = ["kallen", "lelouch", "c.c.", "cornelia", "euphy", "suzaku", "rivalz",
            "ashford", "monica", "shirley", "kunlun", "jiudong", "laoya",
            "liyuu", "kayneth", "caeneus", "abigail", "olga", "yuriko"]
    for r in fate:
        tags[r] = [f"{r}", r]

    one_piece = ["chopper", "robin_(one_piece)", "franky", "brooklyn", "jinbe",
                 "vivi", "perona", "bonney", "boa_hancock"]
    for r in one_piece:
        tags[r] = [r, f"{r}_(one_piece)"]

    madoka = ["akemi_homura", "kaname_madoka", "miki_sayaka", "tomoe_mami"]
    for r in madoka:
        tags[r] = [f"{r}", f"{r}_(madoka_magica)"]

    sousou = ["frieren", "fern", "senshi", "starter"]
    for r in sousou:
        tags[r] = [f"{r}_(sousou_no_frieren)", r]

    others = {
        "pelagia": ["pelagia_(girls'_frontline)", "pelagia"],
        "sapphire": ["sapphire_(princess_connect!)", "sapphire"],
        "yashajin_teni": ["yashajin_teni", "teni"],
        "yuni_(princess_connect!)": ["yuni_(princess_connect!)", "yuni"],
        "anya_forger": ["anya_forger", "anya_(spy_x_family)", "anya"],
        "komeiji_satori": ["komeiji_satori", "satori_(touhou)", "satori"],
        "miyamizu_mitsuha": ["miyamizu_mitsuha", "mitsuha"],
        "tsukiyo": ["tsukiyo_(idol_pride)", "tsukiyo"],
        "aino": ["aino_(princess_connect!)", "aino"],
        "kyaru_(princess_connect!)": ["kyaru_(princess_connect!)", "kyaru"],
        "izumi_sagiri": ["izumi_sagiri", "sagiri"],
        "robin": ["robin_(fire_emblem)", "robin"],
        "Rosmontis": ["rosmontis_(arknights)", "rosmontis"],
        "kagami": ["hiiragi_kagami", "kagami"],
        "tsukasa": ["hiiragi_tsukasa", "tsukasa"],
        "sparkle": ["sparkle_(honkai:_star_rail)", "sparkle"],
        "hanabi": ["hanabi_(honkai:_star_rail)", "hanabi"],
        "black_swan": ["black_swan_(honkai:_star_rail)", "black_swan"],
        "hanya": ["hanya_(jujutsu_kaisen)", "hanya"],
        "griseo": ["griseo_(honkai_impact)", "griseo"],
        "hanako": ["hanako_(blue_archive)", "hanako"],
        "hifumi": ["ajitani_hifumi", "hifumi_(blue_archive)"],
        "fuka": ["fuka_(honkai_impact)", "fuka"],
    }
    tags.update(others)
    return tags


ROLE_TAGS = build_role_tags()


# ── 工具函数 ──
def curl_get(url: str) -> Optional[object]:
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
    except (subprocess.TimeoutExpired, Exception):
        return None


def curl_download(url: str, save_path: str) -> bool:
    try:
        result = subprocess.run(
            ["curl", "-s", "--connect-timeout", "15",
             "--max-time", "60",
             "-o", save_path,
             "-H", "User-Agent: Mozilla/5.0"],
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
def fetch_safebooru(tag: str, page: int = 0, limit: int = PAGE_LIMIT) -> List[Dict]:
    enc = urllib.parse.quote(tag)
    url = (f"https://safebooru.org/index.php?page=dapi&s=post&q=index"
           f"&json=1&limit={limit}&pid={page}&tags={enc}")
    data = curl_get(url)
    return data if isinstance(data, list) else []


def fetch_yande(tag: str, page: int = 0, limit: int = PAGE_LIMIT) -> List[Dict]:
    enc = urllib.parse.quote(tag)
    url = f"https://yande.re/post.json?tags={enc}&limit={limit}&page={page}"
    data = curl_get(url)
    return data if isinstance(data, list) else []


def fetch_gelbooru(tag: str, page: int = 0, limit: int = PAGE_LIMIT) -> List[Dict]:
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


def find_best_tag(dir_name: str, tags: List[str], cache: Dict[str, str]) -> Optional[str]:
    if dir_name in cache:
        return cache[dir_name]
    for tag in tags:
        for src_name, fetch_fn in SOURCES:
            posts = fetch_fn(tag, page=0, limit=5)
            if posts:
                cache[dir_name] = tag
                save_cache(cache)
                with print_lock:
                    print(f"    命中标签: {tag} (via {src_name}, {len(posts)} posts)")
                return tag
            time.sleep(REQUEST_DELAY)
    return None


def download_single_post(args: Tuple) -> Tuple:
    """下载单个帖子（顺序调用，但可被线程池调度）"""
    dir_name, url, file_path_str, post_id = args
    ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
    if ext.lower() not in IMAGE_EXTS:
        ext = ".jpg"
    full_path = f"{file_path_str}{ext}"
    if os.path.exists(full_path):
        return (dir_name, post_id, ext, False, None)
    try:
        # 用curl下载，注意args顺序
        result = subprocess.run(
            ["curl", "-sSL", "--connect-timeout", "15",
             "--max-time", "60",
             "-o", full_path,
             "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
             url],
            capture_output=True, timeout=65
        )
        if result.returncode != 0 or not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
            if os.path.exists(full_path) and os.path.getsize(full_path) == 0:
                os.unlink(full_path)
            return (dir_name, post_id, ext, False, None)
        try:
            with hash_lock:
                file_hash = hashlib.md5(Path(full_path).read_bytes()).hexdigest()
            return (dir_name, post_id, ext, True, file_hash)
        except Exception:
            if os.path.exists(full_path):
                os.unlink(full_path)
            return (dir_name, post_id, ext, False, None)
    except Exception as e:
        with print_lock:
            print(f"[ERR] download: {e}")
        return (dir_name, post_id, ext, False, None)


def collect_character(dir_name: str, tags: List[str], target: int,
                      existing: int, cache: Dict[str, str]) -> Tuple[int, int]:
    """采集单个角色，返回 (已下载, 最终数量)"""
    need = max(0, target - existing)
    if need <= 0:
        return 0, existing

    save_dir = FINAL_DIR / dir_name
    os.makedirs(save_dir, exist_ok=True)
    hashes = load_existing_hashes(save_dir)

    # 尝试所有标签
    effective_tags = tags[:]
    if dir_name in cache:
        effective_tags.insert(0, cache[dir_name])
    effective_tags = list(dict.fromkeys(effective_tags))  # 去重保序

    downloaded = 0
    tried_tags = set()

    for tag_idx, tag in enumerate(effective_tags):
        if downloaded >= need:
            break
        if tag in tried_tags:
            continue
        tried_tags.add(tag)

        with print_lock:
            print(f"  📥 {dir_name}: tag={tag} (已下载 {downloaded}/{need})")

        tag_found = False
        for src_name, fetch_fn in SOURCES:
            if downloaded >= need:
                break

            # 探测标签是否有数据
            probe = fetch_fn(tag, page=0, limit=3)
            if not probe:
                continue
            tag_found = True
            if tag not in cache or cache[dir_name] != tag:
                cache[dir_name] = tag
                save_cache(cache)

            for page in range(MAX_PAGES):
                if downloaded >= need:
                    break
                posts = fetch_fn(tag, page=page, limit=min(need * 2, PAGE_LIMIT))
                if not posts:
                    break

                with print_lock:
                    print(f"    {src_name} p{page}: {len(posts)} posts", end="")

                tasks = []
                for post in posts:
                    if downloaded + len(tasks) >= need + 50:
                        break
                    url = SOURCE_DOWNLOAD[src_name](post)
                    if not url or not url.startswith("http"):
                        continue
                    post_id = post.get("id", random.randint(100000, 999999))
                    file_path_str = str(save_dir / str(post_id))
                    if os.path.exists(f"{file_path_str}.jpg") or os.path.exists(f"{file_path_str}.png") or os.path.exists(f"{file_path_str}.webp"):
                        continue
                    tasks.append((dir_name, url, file_path_str, post_id))

                if not tasks:
                    with print_lock:
                        print(" -> 全部已存在")
                    continue

                # 并行下载
                page_dl = 0
                with ThreadPoolExecutor(max_workers=DL_WORKERS) as executor:
                    futures = {executor.submit(download_single_post, t): t for t in tasks}
                    for future in as_completed(futures):
                        result = future.result()
                        _, post_id, ext, success, file_hash = result
                        if success:
                            if file_hash not in hashes:
                                with hash_lock:
                                    hashes.add(file_hash)
                                downloaded += 1
                                page_dl += 1
                                with print_lock:
                                    print(".", end="", flush=True)
                            else:
                                # MD5重复
                                fpath = str(save_dir / f"{post_id}{ext}")
                                if os.path.exists(fpath):
                                    os.unlink(fpath)

                with print_lock:
                    print(f" -> 本轮下载 {page_dl}, 累计 {downloaded}")

                if downloaded >= need:
                    break
                time.sleep(SOURCE_DELAY)

            if downloaded >= need:
                break
            time.sleep(SOURCE_DELAY)

        if not tag_found:
            with print_lock:
                print(f"    标签 {tag} 无数据，尝试下一个")
            time.sleep(REQUEST_DELAY)

    final_cnt = count_images(dir_name)
    with print_lock:
        mark = "✅" if final_cnt >= target else "⚠️"
        print(f"  {mark} {dir_name}: 下载 {downloaded} 张, 总计 {final_cnt}")
    return downloaded, final_cnt


# ── 主流程 ──
def main():
    import argparse
    global DL_WORKERS, ROLE_WORKERS
    parser = argparse.ArgumentParser(description="多线程并发角色采集")
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--roles", type=str, default="", help="逗号分隔角色名")
    parser.add_argument("--role-workers", type=int, default=3)
    parser.add_argument("--dl-workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min", type=int, default=0, help="最低现有数量阈值")
    parser.add_argument("--max", type=int, default=200, help="最高现有数量阈值")
    args = parser.parse_args()

    DL_WORKERS = args.dl_workers
    ROLE_WORKERS = args.role_workers

    if args.roles:
        role_names = [r.strip() for r in args.roles.split(",")]
    else:
        role_names = []
        if FINAL_DIR.exists():
            for d in FINAL_DIR.iterdir():
                if not d.is_dir():
                    continue
                cnt = sum(1 for f in d.iterdir()
                          if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
                if args.min <= cnt < args.max:
                    role_names.append(d.name)

    if not role_names:
        print("没有需要采集的角色")
        return

    print(f"目标: {args.target} 张/角色, 共 {len(role_names)} 个角色")
    print(f"角色并发: {ROLE_WORKERS}, 下载并发: {DL_WORKERS}")
    print("=" * 60)

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

    # 多线程并发处理角色
    with ThreadPoolExecutor(max_workers=ROLE_WORKERS) as executor:
        futures = {}
        for dir_name, tags, existing in tasks:
            need = max(0, args.target - existing)
            if need <= 0:
                with print_lock:
                    print(f"  ✅ {dir_name}: 已有 {existing} 张，跳过")
                continue
            future = executor.submit(
                collect_character, dir_name, tags, args.target, existing, cache
            )
            futures[future] = dir_name

        for future in as_completed(futures):
            dir_name = futures[future]
            try:
                dl, final = future.result()
                total_dl += dl
            except Exception as e:
                with print_lock:
                    print(f"  ❌ {dir_name}: 异常 - {e}")

    print(f"\n{'=' * 60}")
    print(f"采集完成，共下载 {total_dl} 张")

    # 最终统计
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
