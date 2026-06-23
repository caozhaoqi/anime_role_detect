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
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import OrderedDict

# ── 项目路径 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DANBOORU_DIR = PROJECT_ROOT / "archived" / "spider_image_system" / "src" / "danbooru"
CONFIG_DIR = Path(__file__).parent / "config"

from db_utils import DB
sys.path.insert(0, str(DANBOORU_DIR))
sys.path.insert(0, str(DANBOORU_DIR.parent))

FINAL_DATASET_DIR = PROJECT_ROOT / "data" / "final_dataset"
KEYWORD_DIR = PROJECT_ROOT / "archived" / "auto_spider_img" / "keywords"

# ── 从 JSON 配置文件加载映射 ──────────────────────────────
def load_config():
    """从 JSON 配置文件加载映射"""
    config_path = CONFIG_DIR / "collection_config.json"
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        print("请先运行: python3 scripts/data_collection/config/generate_config.py")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

_config = load_config()

GAME_TAG_MAP = _config.get('game_tag_map', {})
KEYWORD_GAME_MAP = _config.get('keyword_game_map', {})
CHARACTER_MAP = _config.get('character_map', {})
ALL_CHARACTERS = _config.get('all_characters', {})


# ── 读取所有 keyword 文件 ─────────────────────────────────
def collect_chinese_names(keyword_dir: str) -> Dict[str, str]:
    """
    读取所有 keyword 文件，返回 {中文名: 来源文件}
    """
    names = OrderedDict()
    keyword_path = Path(keyword_dir)
    if not keyword_path.exists():
        print(f"️ keyword 目录不存在: {keyword_dir}")
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
        print(f"从 RDS 加载哈希失败，降级到空集合: {e}")
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
        print(f" 持久化哈希到 RDS 失败: {e}")


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
                        global_hashes: Optional[set] = None,
                        skip_global_duplicates: bool = True) -> Tuple[int, int, int]:
    """
    使用 Spider 系统下载单个角色的图片（直接存入 save_dir，不创建嵌套子目录）。
    下载后通过 sha256 内容去重（本角色 + 全局跨角色），避免重复图片。

    参数:
        global_hashes: 全局去重索引（跨角色），由 _load_global_hash_db() 从 RDS 加载
        skip_global_duplicates: 是否跳过全局重复图片（默认True，设为False可强制保存）
    返回 (成功数, 失败数)
    """
    from danbooru_mirror_spider import DanbooruMirrorSpider

    new_hashes = set()
    deduplicated = 0

    existing_hashes = _compute_existing_hashes(save_dir)
    if global_hashes and skip_global_duplicates:
        old_count = len(existing_hashes)
        existing_hashes.update(global_hashes)
        print(f"本角色 {old_count} 张 + 全局 {len(global_hashes)} 个哈希 = {len(existing_hashes)} 去重基准")

    spider = DanbooruMirrorSpider(site="safebooru", max_workers=4)

    sites_to_try = ["safebooru", "konachan", "yande.re", "lolibooru", "gelbooru"]

    total_success = 0
    total_fail = 0
    remaining = max_count

    for site in sites_to_try:
        if remaining <= 0:
            break
        print(f" → 尝试站点: {site}")
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
                print(f"{site}: 未找到匹配图片")
                continue

            # 逐张下载，直接存到 save_dir
            site_success = 0
            site_fail = 0
            download_start = time.time()
            for i, post in enumerate(posts, 1):
                # 单角色总超时检查（5 分钟，超时则跳到下一站点）
                if time.time() - download_start > 300:
                    print(f"{site}: 下载超时（已下载 {site_success} 张），跳到下一站点")
                    break

                if site_success >= remaining:
                    break

                # 定期打印进度
                if i % 5 == 1 or i == len(posts):
                    elapsed = time.time() - download_start
                    print(f"下载 [{site_success}/{remaining}] 第{i}张... ({elapsed:.0f}s)")
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
                        deduplicated += 1
                        if deduplicated <= 5:
                            print(f"  内容去重跳过: {file_path.name} (哈希已存在)")
                        continue

                    with open(file_path, 'wb') as f:
                        f.write(data)

                    existing_hashes.add(img_hash)
                    new_hashes.add(img_hash)
                    site_success += 1
                except Exception as e:
                    site_fail += 1

            print(f"{site}: 成功={site_success}, 失败={site_fail}")
            total_success += site_success
            total_fail += site_fail
        except Exception as e:
            print(f" {site} 失败: {e}")

        remaining = max_count - total_success
        time.sleep(random.uniform(0.5, 1.5))

    # 去重基准更新：把本次新增的哈希带回去 + 持久化到数据库
    db_inserted = 0
    if global_hashes is not None and new_hashes:
        global_hashes.update(new_hashes)
        print(f" 去重基准已更新: +{len(new_hashes)} 个新哈希 (累计 {len(global_hashes)})")
        role_name = Path(save_dir).name
        db_inserted = _append_hashes_to_db(new_hashes, role_name)

    # 打印去重统计
    if deduplicated > 0:
        print(f" ⚠️ 内容去重跳过 {deduplicated} 张（哈希已存在于全局索引）")

    return total_success, total_fail, db_inserted

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
    parser.add_argument("--no-global-dedup", action="store_true",
                        help="禁用全局跨角色内容去重，强制保存所有下载的图片")
    args = parser.parse_args()

    # 确保 final_dataset 目录存在
    os.makedirs(FINAL_DATASET_DIR, exist_ok=True)

    # 1. 读取 keyword 文件
    print("=" * 60)
    print(" 读取 keyword 文件...")
    chinese_names = collect_chinese_names(str(KEYWORD_DIR))
    print(f"共 {len(chinese_names)} 个不重复角色")

    # 2. 获取 final_dataset 已有状态（本地扫描 + 数据库查询）
    existing_counts = {}

    # 2a. 从数据库查询已采集角色数据
    if args.skip_db_threshold > 0:
        try:
            print("从 RDS 查询已采集角色数据...")
            db_rows = DB._fetchall(
                "SELECT role_name, training_count, final_count, total_count FROM role_stats"
            )
            for row in db_rows:
                # 使用 total_count 作为判断依据
                existing_counts[row['role_name']] = row['total_count']
            print(f" 数据库查询成功: {len(db_rows)} 个角色")
        except Exception as e:
            print(f"数据库查询失败: {e}，回退到本地扫描")

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

    print(f"final_dataset 已有 {len(existing_counts)} 个角色目录，共 {sum(existing_counts.values())} 张")
    if roles_to_skip:
        print(f"将跳过 {len(roles_to_skip)} 个角色（>= {skip_threshold} 张）:")
        for name in sorted(roles_to_skip.keys(), key=lambda x: roles_to_skip[x], reverse=True)[:10]:
            print(f"{name}: {roles_to_skip[name]} 张")
        if len(roles_to_skip) > 10:
            print(f"... 及其他 {len(roles_to_skip) - 10} 个角色")

    # 2b. 加载全局去重索引（从 SQLite，无需扫描图片）
    print("加载全局跨角色去重索引...")
    global_hashes, hash_count = _load_global_hash_db()
    print(f"全局索引: {hash_count} 个唯一哈希（db: {HASH_DB_PATH.name}）")
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
                    print(f"tag_resolver 解析成功: {target_tag}")
                else:
                    print(f"无法解析: {chinese_name}，跳过")
                    total_notfound += 1
                    continue
            except Exception as e:
                print(f"tag_resolver 异常: {e}，跳过")
                total_notfound += 1
                continue

        # 3b. 目录名（去掉括号内容, 保留纯英文名）
        dir_name = target_tag.split("(")[0].strip().rstrip("_")

        # 3c. 检查是否已满足
        existing = existing_counts.get(dir_name, 0)

        # 3c-1. 检查是否超过数据库阈值（从 training_dataset 等来源已有足够数据）
        if args.skip_db_threshold > 0 and existing >= args.skip_db_threshold:
            print(f"{dir_name} 已有 {existing} 张 >= {args.skip_db_threshold}（数据库阈值），跳过采集")
            total_skip += 1
            continue

        # 3c-2. 检查是否超过目标数量
        need = args.max_count - existing
        if args.skip_existing and need <= 0:
            print(f"已有 {existing} ≥ {args.max_count}, 跳过")
            total_skip += 1
            continue
        if need <= 0:
            print(f"已有 {existing} ≥ {args.max_count}, 跳过")
            total_skip += 1
            continue

        print(f"tag={target_tag}, dir={dir_name}, need={need}/{args.max_count} (已有{existing})")

        # 3d. 下载
        save_dir = str(FINAL_DATASET_DIR / dir_name)
        os.makedirs(save_dir, exist_ok=True)

        try:
            success, fail, db_inserted = download_character(target_tag, save_dir,
                                                              max_count=need,
                                                              global_hashes=global_hashes,
                                                              skip_global_duplicates=not args.no_global_dedup)
            print(f" {chinese_name}: 成功={success}, 失败={fail}, 数据库新增={db_inserted}")
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
                    new_hashes_added=db_inserted,
                )
            except Exception as e:
                print(f"写入采集记录失败: {e}")

            # 2. 更新角色统计数据到 role_stats 表
            try:
                # 重新扫描目录获取真实的图片数量
                actual_count = len([f for f in Path(save_dir).iterdir()
                                   if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')])

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
                        (actual_count, actual_count, dir_name)
                    )
                else:
                    # 插入新记录
                    DB._execute(
                        "INSERT INTO role_stats (role_name, training_count, final_count, "
                        "total_count, skip_threshold) VALUES (%s, %s, %s, %s, %s)",
                        (dir_name, 0, actual_count, actual_count, 100)
                    )

                print(f" 已同步角色统计: {dir_name} {existing} → {actual_count} 张 (实际扫描)")
            except Exception as e:
                print(f"更新角色统计失败: {e}")

        except Exception as e:
            print(f" {chinese_name} 下载失败: {e}")

        # 角色间延迟
        time.sleep(random.uniform(1.0, 2.0))

    # 4. 汇总
    print("\n" + "=" * 60)
    print(" 采集完成汇总")
    print(f"总角色: {len(chinese_names)}")
    print(f"已采集/补充: {total_done}")
    print(f"跳过(已满足): {total_skip}")
    print(f"未找到标签: {total_notfound}")

    # final state
    final_counts = {}
    for d in FINAL_DATASET_DIR.iterdir():
        if d.is_dir():
            final_counts[d.name] = len([f for f in d.iterdir()
                                        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')])
    print(f"\n   final_dataset 最终: {len(final_counts)} 角色, {sum(final_counts.values())} 张")


if __name__ == "__main__":
    main()