#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多源采集脚本 v3 — 确保每个角色至少 100 张

数据源（按优先级）:
1. Yande.re — 支持完整日文名标签，主力源
2. Safebooru — 仅短名标签，辅助源
3. Gelbooru — 限流，需间隔 2s

标签策略:
- 完整名 (ajitani_hifumi) → Yande.re
- 短名 (hifumi) → Safebooru
- 短名+系列 (hifumi_(blue_archive)) → Safebooru

Usage:
    python3 scripts/data_collection/collect_v3.py --dry-run
    python3 scripts/data_collection/collect_v3.py --target 100 --workers 3
    python3 scripts/data_collection/collect_v3.py --character cirno
"""

import os
import sys
import json
import time
import random
import hashlib
import argparse
import urllib.parse
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).parent.parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"
STATE_FILE = PROJECT_ROOT / "data" / ".collect_state.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# ── 会话管理 ─────────────────────────────────────────────

def create_session():
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(HEADERS)
    return s

# ── 数据源 ───────────────────────────────────────────────

def fetch_yande(session: requests.Session, tag: str, limit: int = 100) -> List[Dict]:
    """Yande.re — 支持完整日文名标签"""
    encoded = urllib.parse.quote(tag.replace(" ", "_"))
    url = f"https://yande.re/post.json?tags={encoded}&limit={limit}"
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
            return []
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return []

def fetch_safebooru(session: requests.Session, tag: str, limit: int = 100) -> List[Dict]:
    """Safebooru — 仅短名标签"""
    encoded = urllib.parse.quote(tag.replace(" ", "_"))
    url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&limit={limit}&tags={encoded}"
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=15)
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

def fetch_gelbooru(session: requests.Session, tag: str, limit: int = 50) -> List[Dict]:
    """Gelbooru — 需要限流"""
    encoded = urllib.parse.quote(tag.replace(" ", "_"))
    url = f"https://gelbooru.com/index.php?page=dapi&s=post&q=index&json=1&limit={limit}&tags={encoded}"
    time.sleep(2)  # 避免 429
    for attempt in range(2):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if isinstance(data, list):
                        return data
                except Exception:
                    pass
            return []
        except Exception:
            if attempt < 1:
                time.sleep(2)
    return []

# ── 下载器 ───────────────────────────────────────────────

def download_yande(session, post, save_dir: Path, hashes: Set[str]) -> Tuple[bool, str]:
    try:
        url = post.get("file_url", "")
        if not url:
            return False, ""
        ext = os.path.splitext(url)[1] or ".jpg"
        if ext.lower() not in IMAGE_EXTS:
            ext = ".jpg"
        fname = save_dir / f"{post.get('id', random.randint(100000, 999999))}{ext}"
        if fname.exists():
            return False, ""
        resp = session.get(url, timeout=(10, 30))
        data = resp.content
        if len(data) < 1024:
            return False, ""
        h = hashlib.md5(data).hexdigest()
        if h in hashes:
            return False, h
        fname.write_bytes(data)
        return True, h
    except Exception:
        return False, ""

def download_safebooru(session, post, save_dir: Path, hashes: Set[str]) -> Tuple[bool, str]:
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
        fname = save_dir / f"{post.get('id', random.randint(100000, 999999))}{ext}"
        if fname.exists():
            return False, ""
        resp = session.get(url, timeout=(10, 30))
        data = resp.content
        if len(data) < 1024:
            return False, ""
        h = hashlib.md5(data).hexdigest()
        if h in hashes:
            return False, h
        fname.write_bytes(data)
        return True, h
    except Exception:
        return False, ""

def download_gelbooru(session, post, save_dir: Path, hashes: Set[str]) -> Tuple[bool, str]:
    try:
        url = post.get("file_url", "")
        if not url:
            return False, ""
        ext = os.path.splitext(url)[1] or ".jpg"
        if ext.lower() not in IMAGE_EXTS:
            ext = ".jpg"
        fname = save_dir / f"{post.get('id', random.randint(100000, 999999))}{ext}"
        if fname.exists():
            return False, ""
        resp = session.get(url, timeout=(10, 30))
        data = resp.content
        if len(data) < 1024:
            return False, ""
        h = hashlib.md5(data).hexdigest()
        if h in hashes:
            return False, h
        fname.write_bytes(data)
        return True, h
    except Exception:
        return False, ""

# ── 标签生成 ──────────────────────────────────────────────

def generate_tag_variants(dir_name: str) -> List[str]:
    """
    根据目录名生成标签变体，按优先级排列

    规则:
    - 纯短名 (cirno) → [cirno, cirno_(touhou)]
    - 完整名 (ajitani_hifumi) → [ajitani_hifumi]
    - 完整名+下划线 (hakurei_reimu) → [hakurei_reimu]
    """
    name = dir_name.strip().lower()
    variants = []

    # 已知的角色名→标签映射
    KNOWN_MAP = {
        # 蔚蓝档案（短名映射）
        "arona": ["arona_(blue_archive)", "arona"],
        "plana": ["plana_(blue_archive)", "plana"],
        "shiroko": ["sunaookami_shiroko", "shiroko_(blue_archive)", "shiroko"],
        "hifumi": ["hifumi_(blue_archive)", "ajitani_hifumi", "hifumi"],
        "hina": ["sorasaki_hina", "hina_(blue_archive)", "hina"],
        "mika": ["misono_mika", "mika_(blue_archive)", "mika"],
        "hoshino": ["takanashi_hoshino", "hoshino_(blue_archive)", "hoshino"],
        "aris": ["tendou_aris", "aris_(blue_archive)", "aris"],
        "azusa": ["shirasu_azusa", "azusa_(blue_archive)", "azusa"],
        "izuna": ["kuda_izuna", "izuna_(blue_archive)", "izuna"],
        "serika": ["kuromi_serika", "serika_(blue_archive)", "serika"],
        "nonomi": ["izayoi_nonomi", "nonomi_(blue_archive)", "nonomi"],
        "ayane": ["okusora_ayane", "ayane_(blue_archive)", "ayane"],
        "haruna": ["kurodate_haruna", "haruna_(blue_archive)", "haruna"],
        "ako": ["amau_ako", "ako_(blue_archive)", "ako"],
        "aru": ["rikuhachima_aru", "aru_(blue_archive)", "aru"],
        "mutsuki": ["asagi_mutsuki", "mutsuki_(blue_archive)", "mutsuki"],
        "kayoko": ["onikata_kayoko", "kayoko_(blue_archive)", "kayoko"],
        "koharu": ["shimoe_koharu", "koharu_(blue_archive)", "koharu"],
        "wakamo": ["kosaka_wakamo", "wakamo_(blue_archive)", "wakamo"],
        "nodoka": ["amami_nodoka", "nodoka_(blue_archive)", "nodoka"],
        "ibuki": ["ibuki_(blue_archive)", "ibuki"],
        # 原神
        "nahida": ["nahida_(genshin_impact)", "nahida"],
        "klee": ["klee_(genshin_impact)", "klee"],
        "diona": ["diona_(genshin_impact)", "diona"],
        "yaoyao": ["yaoyao_(genshin_impact)", "yaoyao"],
        "sigewinne": ["sigewinne_(genshin_impact)", "sigewinne"],
        "qiqi": ["qiqi_(genshin_impact)", "qiqi"],
        "sayu": ["sayu_(genshin_impact)", "sayu"],
        "dori": ["dori_(genshin_impact)", "dori"],
        "paimon": ["paimon_(genshin_impact)", "paimon"],
        "kachina": ["kachina_(genshin_impact)", "kachina"],
        "furina": ["furina_(genshin_impact)", "furina"],
        "hu_tao": ["hu_tao_(genshin_impact)", "hu_tao"],
        "kirara": ["kirara_(genshin_impact)", "kirara"],
        "collei": ["collei_(genshin_impact)", "collei"],
        "amber": ["amber_(genshin_impact)", "amber"],
        "mona": ["mona_(genshin_impact)", "mona"],
        "keqing": ["keqing_(genshin_impact)", "keqing"],
        "xiangling": ["xiangling_(genshin_impact)", "xiangling"],
        "ningguang": ["ningguang_(genshin_impact)", "ningguang"],
        "noelle": ["noelle_(genshin_impact)", "noelle"],
        "rosaria": ["rosaria_(genshin_impact)", "rosaria"],
        "yanfei": ["yanfei_(genshin_impact)", "yanfei"],
        "yoimiya": ["yoimiya_(genshin_impact)", "yoimiya"],
        "nilou": ["nilou_(genshin_impact)", "nilou"],
        "faruzan": ["faruzan_(genshin_impact)", "faruzan"],
        "kujou_sara": ["kujou_sara_(genshin_impact)", "kujou_sara"],
        "kokomi": ["sangonomiya_kokomi", "kokomi_(genshin_impact)", "kokomi"],
        "clorinde": ["clorinde_(genshin_impact)", "clorinde"],
        # 崩铁
        "herta": ["herta_(honkai:_star_rail)", "herta"],
        "fu_xuan": ["fu_xuan_(honkai:_star_rail)", "fu_xuan"],
        "march_7th": ["march_7th_(honkai:_star_rail)", "march_7th"],
        "sparkle": ["sparkle_(honkai:_star_rail)", "sparkle"],
        "silver_wolf": ["silver_wolf_(honkai:_star_rail)", "silver_wolf"],
        "clara": ["clara_(honkai:_star_rail)", "clara"],
        "hook": ["hook_(honkai:_star_rail)", "hook"],
        "yunli": ["yunli_(honkai:_star_rail)", "yunli"],
        "bailu": ["bailu_(honkai:_star_rail)", "bailu"],
        "firefly": ["firefly_(honkai:_star_rail)", "firefly"],
        "black_swan": ["black_swan_(honkai:_star_rail)", "black_swan"],
        "kafka": ["kafka_(honkai:_star_rail)", "kafka"],
        "serval": ["serval_(honkai:_star_rail)", "serval"],
        "sushang": ["sushang_(honkai:_star_rail)", "sushang"],
        "tingyun": ["tingyun_(honkai:_star_rail)", "tingyun"],
        "topaz": ["topaz_(honkai:_star_rail)", "topaz"],
        "xueyi": ["xueyi_(honkai:_star_rail)", "xueyi"],
        "yukong": ["yukong_(honkai:_star_rail)", "yukong"],
        "natasha": ["natasha_(honkai:_star_rail)", "natasha"],
        "qingque": ["qingque_(honkai:_star_rail)", "qingque"],
        "aglaea": ["aglaea_(honkai:_star_rail)", "aglaea"],
        "castorice": ["castorice_(honkai:_star_rail)", "castorice"],
        "tribbie": ["tribbie_(honkai:_star_rail)", "tribbie"],
        # 崩坏3
        "theresa": ["theresa_apocalypse", "theresa_(honkai_impact)", "theresa"],
        "bronya": ["bronya_zaychik", "bronya_(honkai_impact)", "bronya"],
        "seele": ["seele_vollerei", "seele_(honkai_impact)", "seele"],
        "griseo": ["griseo_(honkai_impact)", "griseo"],
        "fuka": ["fuka_(honkai_impact)", "fuka"],
        "sin_mal": ["sin_mal_(honkai_impact)", "sin_mal"],
        "einstein": ["einstein_(honkai_impact)", "einstein"],
        "rozaliya": ["rozaliya_(honkai_impact)", "rozaliya"],
        "liliya": ["liliya_(honkai_impact)", "liliya"],
        # 东方Project
        "cirno": ["cirno_(touhou)", "cirno"],
        "flandre_scarlet": ["flandre_scarlet_(touhou)", "flandre_scarlet"],
        "remilia_scarlet": ["remilia_scarlet_(touhou)", "remilia_scarlet"],
        "hakurei_reimu": ["hakurei_reimu_(touhou)", "hakurei_reimu"],
        "kirisame_marisa": ["kirisame_marisa_(touhou)", "kirisame_marisa"],
        "komeiji_koishi": ["komeiji_koishi_(touhou)", "komeiji_koishi"],
        "komeiji_satori": ["komeiji_satori_(touhou)", "komeiji_satori"],
        # 鸣潮
        "verina": ["verina_(wuthering_waves)", "verina"],
        "encore": ["encore_(wuthering_waves)", "encore"],
        "youhu": ["youhu_(wuthering_waves)", "youhu"],
        # Re:Zero
        "rem": ["rem_(re:zero)", "rem"],
        "ram": ["ram_(re:zero)", "ram"],
        "beatrice": ["beatrice_(re:zero)", "beatrice"],
        # 龙女仆
        "kanna": ["kanna_kamui", "kanna_(maid_dragon)", "kanna"],
        "tohru": ["tohru_(maid_dragon)", "tohru"],
        # 公主连接
        "kyaru": ["kyaru_(princess_connect!)", "kyaru"],
        "kyouka": ["kyouka_(princess_connect!)", "kyouka"],
        "yuni": ["yuni_(princess_connect!)", "yuni"],
        "kokkoro": ["kokkoro_(princess_connect!)", "kokkoro"],
        "pecorine": ["pecorine_(princess_connect!)", "pecorine"],
        "yui": ["yui_(princess_connect!)", "yui"],
        "maho": ["maho_(princess_connect!)", "maho"],
        # 明日方舟
        "suzuran": ["suzuran_(arknights)", "suzuran"],
        "eyjafjalla": ["eyjafjalla_(arknights)", "eyjafjalla"],
        "rosmontis": ["rosmontis_(arknights)", "rosmontis"],
        "ceobe": ["ceobe_(arknights)", "ceobe"],
        "popukar": ["popukar_(arknights)", "popukar"],
        # 绝区零
        "corin": ["corin_wickes", "corin_(zenless_zone_zero)", "corin"],
        "koleda": ["koleda_belobog", "koleda_(zenless_zone_zero)", "koleda"],
        "nekomiya_mana": ["nekomiya_mana_(zenless_zone_zero)", "nekomiya_mana"],
        # 约战
        "yoshino": ["yoshino_(date_a_live)", "yoshino"],
        "kurumi": ["kurumi_tokisaki", "tokisaki_kurumi"],
        "kotori": ["kotori_itsuka", "itsuka_kotori"],
        "tohka": ["tohka_yatogami", "yatogami_tohka"],
        # Fate
        "illya": ["illyasviel_von_einzbern", "illya_(fate)", "illya"],
        "chloe": ["chloe_von_einzbern", "chloe_(fate)", "chloe"],
        "miyu": ["miyu_edelfelt", "miyu_(fate)", "miyu"],
        # 其他
        "asuna": ["yuuki_asuna", "asuna_(blue_archive)", "asuna"],
        "lucia": ["lucia_(punishing:_gray_raven)", "lucia"],
        "pearl": ["pearl_(honkai:_star_rail)", "pearl"],
        "sandrone": ["sandrone_(genshin_impact)", "sandrone"],
        "dunyarzad": ["dunyarzad_(genshin_impact)", "dunyarzad"],
        "charlotte": ["charlotte_(genshin_impact)", "charlotte"],
        "fumilis": ["fumilis"],
        "hayiri": ["hayiri"],
        "mint": ["mint_(neverness_to_everness)", "mint"],
        "rosci": ["rosci_(arknights)", "rosci"],
        "ilulu": ["ilulu_(maid_dragon)", "ilulu"],
        "platelet": ["platelet_(hataraku_saibou)", "platelet"],
        "kagura": ["kagura_(onmyouji)", "kagura"],
        "shirakami_fubuki": ["shirakami_fubuki"],
        "aqua": ["aqua_(konosuba)", "minato_aqua", "aqua"],
        "megumin": ["megumin_(konosuba)", "megumin"],
        "yunyun": ["yunyun_(konosuba)", "yunyun"],
        "umaru": ["doma_umaru", "umaru_(himouto)", "umaru"],
        "sagiri": ["izumi_sagiri", "sagiri_(eromanga)", "sagiri"],
        "anya": ["anya_forger", "anya_(spy_x_family)", "anya"],
        "chino": ["chino_kafuu", "kafuu_chino"],
        "cocoa": ["cocoa_hoto", "hoto_cocoa"],
        "rize": ["rize_tedeza", "tedeza_rize"],
        "chiya": ["chiya_ujimatsu", "ujimatsu_chiya"],
        "syaro": ["syaro_kirima", "kirima_syaro"],
        "shinobu": ["shinobu_oshino", "oshino_shinobu"],
        "mayoi": ["mayoi_hachikuji", "hachikuji_mayoi"],
        "hana": ["hana_(watashi_ni_tenshi)", "shirosaki_hana"],
        "hinata": ["hinata_(watashi_ni_tenshi)", "hoshino_hinata"],
        "noa": ["noa_(watashi_ni_tenshi)", "himesaka_noa"],
        "ai": ["ai_(ryuuou)", "hinatsuru_ai"],
        "renge": ["miyauchi_renge"],
        "mitsuha": ["miyamizu_mitsuha"],
        "shouko": ["nishimiya_shouko"],
        "konata": ["izumi_konata"],
        "kagami": ["hiiragi_kagami"],
        "tsukasa": ["hiiragi_tsukasa"],
        "nico": ["yazawa_nico"],
        "yoshiko": ["tsushima_yoshiko"],
        "madoka": ["kaname_madoka"],
        "homura": ["akemi_homura"],
        "mami": ["tomoe_mami"],
        "sayaka": ["miki_sayaka"],
        "teni": ["yashajin_teni"],
        "ginko": ["sora_ginko"],
        "tsukiyo": ["tsukiyo_(idol_pride)"],
        "aino": ["aino_(idol_pride)"],
        "lita": ["lita_(girls_frontline)"],
        "vipery": ["vipery_(girls_frontline)"],
        "shakri": ["shakri_(girls_frontline)"],
        "nagan": ["nagan_(girls_frontline)"],
        "koshenia": ["koshenia_(girls_frontline)"],
        "corvu": ["corvu_(girls_frontline)"],
        "krolik": ["krolik_(girls_frontline)"],
        "peritia": ["peritia_(girls_frontline)"],
    }

    if name in KNOWN_MAP:
        return KNOWN_MAP[name]

    # 默认：用目录名本身作为标签
    variants.append(name)
    return variants

# ── 核心采集逻辑 ──────────────────────────────────────────

def load_state() -> Dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}

def save_state(state: Dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))

def load_existing_hashes(save_dir: Path) -> Set[str]:
    hashes = set()
    if save_dir.exists():
        for f in save_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                try:
                    hashes.add(hashlib.md5(f.read_bytes()).hexdigest())
                except Exception:
                    pass
    return hashes

def count_images(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    return sum(1 for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)

def collect_character(
    dir_name: str,
    target: int,
    session: requests.Session,
    lock: threading.Lock,
    progress: Dict,
) -> Tuple[int, int]:
    """采集单个角色，返回 (下载数, 总数)"""
    save_dir = FINAL_DIR / dir_name
    os.makedirs(save_dir, exist_ok=True)

    existing = count_images(save_dir)
    need = max(0, target - existing)
    if need <= 0:
        return 0, existing

    hashes = load_existing_hashes(save_dir)
    tags = generate_tag_variants(dir_name)
    downloaded = 0

    # 数据源：Yande.re → Safebooru → Gelbooru
    sources = [
        ("Yande.re", fetch_yande, download_yande),
        ("Safebooru", fetch_safebooru, download_safebooru),
        ("Gelbooru", fetch_gelbooru, download_gelbooru),
    ]

    for tag in tags:
        if downloaded >= need:
            break
        for src_name, fetch_fn, download_fn in sources:
            if downloaded >= need:
                break
            try:
                posts = fetch_fn(session, tag, limit=min(need * 3, 100))
                if not posts:
                    continue
                for post in posts:
                    if downloaded >= need:
                        break
                    success, h = download_fn(session, post, save_dir, hashes)
                    if success and h:
                        hashes.add(h)
                        downloaded += 1
                    time.sleep(random.uniform(0.1, 0.3))
                if downloaded >= need:
                    break
            except Exception as e:
                if "Gelbooru" in src_name:
                    time.sleep(2)  # 限流后等待
            time.sleep(random.uniform(0.3, 0.8))

    total = existing + downloaded
    with lock:
        progress[dir_name] = {"downloaded": downloaded, "total": total, "need": need}
    return downloaded, total

# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="多源采集 v3")
    parser.add_argument("--target", type=int, default=100, help="目标图片数/角色")
    parser.add_argument("--character", type=str, help="单角色采集")
    parser.add_argument("--dry-run", action="store_true", help="预览模式")
    parser.add_argument("--workers", type=int, default=2, help="并发数")
    parser.add_argument("--max-chars", type=int, default=0, help="最大采集角色数")
    args = parser.parse_args()

    # 扫描需要采集的角色
    tasks = []
    if args.character:
        dir_name = args.character.lower()
        save_dir = FINAL_DIR / dir_name
        if save_dir.exists():
            existing = count_images(save_dir)
            need = max(0, args.target - existing)
            tasks.append((dir_name, existing, need))
        else:
            print(f"目录不存在: {dir_name}")
            return
    else:
        if not FINAL_DIR.exists():
            print("final_dataset 目录不存在")
            return
        for d in sorted(FINAL_DIR.iterdir()):
            if not d.is_dir():
                continue
            existing = count_images(d)
            need = max(0, args.target - existing)
            if need > 0:
                tasks.append((d.name, existing, need))

    # 按缺口排序（缺口大的优先）
    tasks.sort(key=lambda x: -x[2])

    if args.max_chars > 0:
        tasks = tasks[:args.max_chars]

    total_need = sum(n for _, _, n in tasks)
    print(f"角色数: {len(tasks)}, 需下载: {total_need} 张")

    if args.dry_run:
        print(f"\n缺口最大的前20个:")
        for name, existing, need in tasks[:20]:
            print(f"  {name:30s}: 已有{existing:>3} → 缺{need:>3} 张")
        return

    # 采集
    os.makedirs(FINAL_DIR, exist_ok=True)
    progress = {}
    lock = threading.Lock()
    total_dl = 0

    def worker(task):
        name, existing, need = task
        session = create_session()
        try:
            dl, total = collect_character(name, args.target, session, lock, progress)
            return name, dl, total
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, t): t for t in tasks}
        completed = 0

        for future in as_completed(futures):
            name, dl, total = future.result()
            total_dl += dl
            completed += 1
            pct = completed / len(tasks) * 100
            status = "✅" if total >= args.target else "🔶"
            print(f"  [{completed}/{len(tasks)} {pct:.0f}%] {status} {name}: +{dl} → {total} 张")

    print(f"\n采集完成: {total_dl} 张")
    save_state(progress)

    # 最终统计
    final_stats = []
    for d in sorted(FINAL_DIR.iterdir()):
        if d.is_dir():
            cnt = count_images(d)
            final_stats.append((d.name, cnt))

    below = sum(1 for _, c in final_stats if c < args.target)
    at = sum(1 for _, c in final_stats if c >= args.target)
    print(f"达标 ({args.target}+): {at} 角色, 未达标: {below} 角色")


if __name__ == "__main__":
    main()