#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_dataset 多站点数据补充采集

目标：将 final_dataset 所有角色补充到 100 张
数据源：Safebooru → Gelbooru → Konachan → Yande.re（自动回退）
去重：post_id 去重 + MD5 内容去重
"""

import os
import sys
import time
import json
import hashlib
import random
import logging
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"collect_fd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============ 配置 ============
FINAL_DATASET_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
TARGET_COUNT = 100
MAX_PAGES_PER_SITE = 5
PAGE_LIMIT = 100
TIMEOUT = 30
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# ============ 多站点配置 ============
SITES = [
    {
        "name": "safebooru",
        "api": "https://safebooru.org/index.php",
        "format": "xml",
        "delay": 1.0,
        "params": {"page": "dapi", "s": "post", "q": "index"},
    },
    {
        "name": "gelbooru",
        "api": "https://gelbooru.com/index.php",
        "format": "json",
        "delay": 3.0,
        "params": {"page": "dapi", "s": "post", "q": "index"},
    },
    {
        "name": "konachan",
        "api": "https://konachan.com/post.json",
        "format": "json",
        "delay": 1.5,
        "params": {},
    },
    {
        "name": "yandere",
        "api": "https://yande.re/post.json",
        "format": "json",
        "delay": 1.5,
        "params": {},
    },
]

# ============ 角色名 → 搜索标签映射 ============
# 从 archived/ 提取的角色名映射
# key=final_dataset目录名, value=(搜索标签, 备选标签)
CHARACTER_TAG_MAP = {
    # 原神
    "amber":        ("amber_(genshin_impact)", "amber"),
    "barbara":      ("barbara_(genshin_impact)", "barbara"),
    "beidou":       ("beidou", "beidou_(genshin_impact)"),
    "charlotte":    ("charlotte_(genshin_impact)", "charlotte"),
    "clorinde":     ("clorinde_(genshin_impact)", "clorinde"),
    "cloud_retainer": ("cloud_retainer", "cloud_retainer_(genshin_impact)"),
    "collei":       ("collei", "collei_(genshin_impact)"),
    "dehya":        ("dehya", "dehya_(genshin_impact)"),
    "eula":         ("eula", "eula_(genshin_impact)"),
    "faruzan":      ("faruzan", "faruzan_(genshin_impact)"),
    "fischl":       ("fischl", "fischl_(genshin_impact)"),
    "ganyu":        ("ganyu", "ganyu_(genshin_impact)"),
    "hu_tao":       ("hu_tao", "hu_tao_(genshin_impact)"),
    "keqing":       ("keqing", "keqing_(genshin_impact)"),
    "lisa":         ("lisa_(genshin_impact)", "lisa"),
    "lynette":      ("lynette_(genshin_impact)", "lynette"),
    "mona":         ("mona_(genshin_impact)", "mona"),
    "nahida":       ("nahida", "nahida_(genshin_impact)"),
    "nilou":        ("nilou", "nilou_(genshin_impact)"),
    "ningguang":    ("ningguang", "ningguang_(genshin_impact)"),
    "noelle":       ("noelle_(genshin_impact)", "noelle"),
    "qiqi":         ("qiqi", "qiqi_(genshin_impact)"),
    "rosaria":      ("rosaria_(genshin_impact)", "rosaria"),
    "sayu":         ("sayu", "sayu_(genshin_impact)"),
    "shenhe":       ("shenhe", "shenhe_(genshin_impact)"),
    "sucrose":      ("sucrose", "sucrose_(genshin_impact)"),
    "xiangling":    ("xiangling", "xiangling_(genshin_impact)"),
    "xinyan":       ("xinyan", "xinyan_(genshin_impact)"),
    "yae_miko":     ("yae_miko", "yae_miko_(genshin_impact)"),
    "yanfei":       ("yanfei", "yanfei_(genshin_impact)"),
    "yelan":        ("yelan", "yelan_(genshin_impact)"),
    "yoimiya":      ("yoimiya", "yoimiya_(genshin_impact)"),
    "yunjin":       ("yunjin", "yunjin_(genshin_impact)"),
    "ayaka":        ("ayaka", "ayaka_(genshin_impact)"),
    "kokomi":       ("kokomi", "kokomi_(genshin_impact)"),
    "shenhe":       ("shenhe", "shenhe_(genshin_impact)"),
    "kirara":       ("kirara", "kirara_(genshin_impact)"),
    "dori":         ("dori_(genshin_impact)", "dori"),
    # 星穹铁道
    "asta":         ("asta_(honkai:_star_rail)", "asta"),
    "march_7th":    ("march_7th", "march_7th_(honkai:_star_rail)"),
    "serval":       ("serval", "serval_(honkai:_star_rail)"),
    "silver_wolf":  ("silver_wolf", "silver_wolf_(honkai:_star_rail)"),
    "seele":        ("seele", "seele_(honkai:_star_rail)"),
    "kafka":        ("kafka_(honkai:_star_rail)", "kafka"),
    "sushang":      ("sushang", "sushang_(honkai:_star_rail)"),
    "himeko":       ("himeko", "himeko_(honkai:_star_rail)"),
    "qingque":      ("qingque", "qingque_(honkai:_star_rail)"),
    "tingyun":      ("tingyun", "tingyun_(honkai:_star_rail)"),
    "topaz":        ("topaz_(honkai:_star_rail)", "topaz"),
    "yukong":       ("yukong", "yukong_(honkai:_star_rail)"),
    "jingliu":      ("jingliu", "jingliu_(honkai:_star_rail)"),
    "ruan_mei":     ("ruan_mei", "ruan_mei_(honkai:_star_rail)"),
    "fu_xuan":      ("fu_xuan", "fu_xuan_(honkai:_star_rail)"),
    "huohuo":       ("huohuo", "huohuo_(honkai:_star_rail)"),
    "xueyi":        ("xueyi", "xueyi_(honkai:_star_rail)"),
    "yunli":        ("yunli", "yunli_(honkai:_star_rail)"),
    "natasha":      ("natasha_(honkai:_star_rail)", "natasha"),
    "sara":         ("sara_(honkai:_star_rail)", "sara"),
    "black_swan":   ("black_swan", "black_swan_(honkai:_star_rail)"),
    # 崩坏3
    "kiana":        ("kiana", "kiana_(honkai_impact_3rd)"),
    "theresa":      ("theresa_(honkai_impact_3rd)", "theresa"),
    "rita":         ("rita_(honkai_impact_3rd)", "rita"),
    "seele":        ("seele", "seele_(honkai_impact_3rd)"),
    "bronya":       ("bronya", "bronya_(honkai_impact_3rd)"),
    "yae_sakura":   ("yae_sakura", "yae_sakura_(honkai_impact_3rd)"),
    "mobius":       ("mobius_(honkai_impact_3rd)", "mobius"),
    "eden":         ("eden_(honkai_impact_3rd)", "eden"),
    "pardofelis":   ("pardofelis", "pardofelis_(honkai_impact_3rd)"),
    "griseo":       ("griseo", "griseo_(honkai_impact_3rd)"),
    "li_sushang":   ("li_sushang", "li_sushang_(honkai_impact_3rd)"),
    "prometheus":   ("prometheus_(honkai_impact_3rd)", "prometheus"),
    "pelagia":      ("pelagia", "pelagia_(honkai_impact_3rd)"),
    "raven":        ("raven_(honkai_impact_3rd)", "raven"),
    # Blue Archive
    "ako":          ("amau_ako", "ako"),
    "hoshino":      ("takanashi_hoshino", "hoshino"),
    "plana":        ("plana", "plana_(blue_archive)"),
    "shigure_kira": ("shigure_kira", "shigure_kira_(blue_archive)"),
    # 其他
    "sacred_garden_mikoto": ("sacred_garden_mikoto", "mikoto"),
}


def build_tags(role_name: str) -> list:
    """构建多组标签（优先使用精确映射）"""
    name = role_name.lower().replace(" ", "_")
    if role_name in CHARACTER_TAG_MAP:
        primary, fallback = CHARACTER_TAG_MAP[role_name]
        return [
            [primary, "solo", "rating:safe"],
            [primary, "-group"],
            [fallback, "solo", "rating:safe"],
            [fallback, "-group"],
            [name, "solo", "rating:safe"],
        ]
    return [
        [name, "solo", "rating:safe"],
        [name, "-group"],
    ]


# ============ 核心逻辑 ============


def get_all_roles() -> list:
    """获取 final_dataset 中需要补充的角色"""
    roles = []
    if not FINAL_DATASET_DIR.exists():
        logger.error(f"目录不存在: {FINAL_DATASET_DIR}")
        return roles
    for d in sorted(FINAL_DATASET_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        existing_ids = set()
        for f in d.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
                existing_ids.add(f.stem)
        count = len(existing_ids)
        if count < TARGET_COUNT:
            roles.append({
                "name": d.name,
                "count": count,
                "needed": TARGET_COUNT - count,
                "existing_ids": existing_ids,
            })
    roles.sort(key=lambda x: x["needed"], reverse=True)
    return roles


def load_md5s(role_name: str) -> set:
    """加载角色目录所有图片的 MD5"""
    md5s = set()
    role_dir = FINAL_DATASET_DIR / role_name
    if not role_dir.exists():
        return md5s
    for f in role_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            continue
        try:
            h = hashlib.md5()
            with open(f, "rb") as fp:
                for chunk in iter(lambda: fp.read(8192), b""):
                    h.update(chunk)
            md5s.add(h.hexdigest())
        except Exception:
            pass
    return md5s


def search_site(site_cfg: dict, tags: list, page: int, session: requests.Session) -> list:
    """搜索单个站点，返回 post 列表"""
    for attempt in range(MAX_RETRIES):
        time.sleep(site_cfg["delay"] * (1 + attempt * 0.5) + random.uniform(0, 1))
        try:
            params = dict(site_cfg["params"])
            params["tags"] = " ".join(tags)
            params["limit"] = PAGE_LIMIT

            if site_cfg["format"] == "xml":
                # Safebooru/Gelbooru XML: pid
                params["pid"] = page
                resp = session.get(site_cfg["api"], params=params, timeout=TIMEOUT)
                if resp.status_code == 200:
                    root = ElementTree.fromstring(resp.content)
                    return [p.attrib for p in root.findall("post")]
            else:
                # JSON: page
                params["page"] = page + 1
                resp = session.get(site_cfg["api"], params=params, timeout=TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    return data if isinstance(data, list) else []

            if resp.status_code == 429:
                logger.warning(f"  [{site_cfg['name']}] 429, retry({attempt+1})")
                time.sleep(site_cfg["delay"] * 5 * (attempt + 1))
                continue

        except Exception as e:
            logger.debug(f"  [{site_cfg['name']}] 搜索失败: {e}")
            time.sleep(site_cfg["delay"] * 2)

    return []


def download_image(post: dict, site_name: str, role_name: str,
                   existing_ids: set, existing_md5s: set, session: requests.Session) -> bool:
    """下载单张图片（post_id + MD5 双重去重）"""
    post_id = str(post.get("id", ""))
    # file_url 字段：gelbooru/safebooru=file_url, konachan/yandere=file_url
    file_url = post.get("file_url", "")
    if not post_id or not file_url:
        return False
    if post_id in existing_ids:
        return False

    ext = ".jpg"
    for e in (".png", ".webp", ".gif", ".jpeg"):
        if file_url.lower().endswith(e):
            ext = e
            break

    for attempt in range(2):
        try:
            resp = session.get(file_url, timeout=TIMEOUT)
            if resp.status_code != 200:
                return False
            data = resp.content
            md5 = hashlib.md5(data).hexdigest()
            if md5 in existing_md5s:
                return False

            role_dir = FINAL_DATASET_DIR / role_name
            role_dir.mkdir(parents=True, exist_ok=True)
            with open(role_dir / f"{post_id}{ext}", "wb") as f:
                f.write(data)
            existing_ids.add(post_id)
            existing_md5s.add(md5)
            logger.info(f"  [{site_name}] ✅ {role_name}/{post_id}{ext}")
            return True
        except Exception:
            time.sleep(1)
    return False


def collect_role(role: dict, session: requests.Session) -> dict:
    """采集单个角色（多站点依次尝试）"""
    name = role["name"]
    needed = role["needed"]
    existing_ids = role["existing_ids"]
    existing_md5s = load_md5s(name)
    tag_sets = build_tags(name)

    logger.info(f"\n{'='*50}")
    logger.info(f"{name}: {role['count']}→{TARGET_COUNT} (需{needed})")

    downloaded = 0

    for site_cfg in SITES:
        if downloaded >= needed:
            break
        logger.info(f"  → 站点: {site_cfg['name']}")

        for tag_set in tag_sets:
            if downloaded >= needed:
                break
            empty_pages = 0
            for page in range(MAX_PAGES_PER_SITE):
                if downloaded >= needed:
                    break
                posts = search_site(site_cfg, tag_set, page, session)
                if not posts:
                    empty_pages += 1
                    if empty_pages >= 2:
                        break
                    continue
                empty_pages = 0
                for post in posts:
                    if downloaded >= needed:
                        break
                    if download_image(post, site_cfg["name"], name, existing_ids, existing_md5s, session):
                        downloaded += 1
                logger.info(f"    [{site_cfg['name']}] pg{page}: +{downloaded}/{needed}")

    return {"role": name, "downloaded": downloaded, "needed": needed}


def main():
    logger.info("=" * 60)
    logger.info(f"final_dataset 多站点采集 (目标:{TARGET_COUNT}/角色)")
    logger.info(f"站点: {[s['name'] for s in SITES]}")
    logger.info("=" * 60)

    roles = get_all_roles()
    if not roles:
        logger.info("全部达标！")
        return

    total = sum(r["needed"] for r in roles)
    logger.info(f"共 {len(roles)} 角色需补充, 总计 {total} 张")
    for r in roles:
        logger.info(f"  {r['name']:>25}: {r['count']}/{TARGET_COUNT}")

    session = requests.Session()
    session.headers.update(HEADERS)

    results = []
    total_ok = 0
    for i, r in enumerate(roles, 1):
        logger.info(f"\n[{i}/{len(roles)}]")
        res = collect_role(r, session)
        results.append(res)
        total_ok += res["downloaded"]
        time.sleep(2)

    # 汇总
    not_met = sum(x["needed"] - x["downloaded"] for x in results if x["downloaded"] < x["needed"])
    logger.info("\n" + "=" * 60)
    logger.info(f"完成！下载 {total_ok} 张, 未满足 {not_met} 张")
    for x in results:
        s = "✅" if x["downloaded"] >= x["needed"] else "⚠️"
        logger.info(f"  {s} {x['role']:>25}: {x['downloaded']}/{x['needed']}")

    out = {
        "timestamp": datetime.now().isoformat(),
        "target": TARGET_COUNT,
        "ok": total_ok,
        "not_met": not_met,
        "results": results,
    }
    fp = f"collect_fd_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果: {fp}")


if __name__ == "__main__":
    main()