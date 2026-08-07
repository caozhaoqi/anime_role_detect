#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计角色数据采集情况，写入本地 SQLite 数据库

功能:
    1. 扫描 training_dataset 和 final_dataset 目录
    2. 统计每个角色的图片数量、文件大小、分辨率
    3. 写入本地 SQLite 数据库 role_stats.db
    4. 输出数据分布报告

用法:
    python3 scripts/data_collection/sync_role_stats_local.py
    python3 scripts/data_collection/sync_role_stats_local.py --db /path/to/role_stats.db
"""

import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_DIR = PROJECT_ROOT / "data" / "training_dataset"
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"
DEFAULT_DB = PROJECT_ROOT / "data" / "role_stats.db"

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

# 短名 → 全名映射（统一角色主键）
NAME_MAP = {
    "aris": "tendou_aris", "Aru": "rikuhachima_aru", "ako": "amau_ako",
    "seele": "seele_vollerei", "hina": "sorasaki_hina", "mika": "misono_mika",
    "hoshino": "takanashi_hoshino", "azusa": "shirasu_azusa", "izuna": "kuda_izuna",
    "shiroko": "sunaookami_shiroko", "hifumi": "ajitani_hifumi", "nodoka": "amami_nodoka",
    "kayoko": "onikata_kayoko", "mutsuki": "asagi_mutsuki", "serika": "kuromi_serika",
    "nonomi": "izayoi_nonomi", "ayane": "okusora_ayane", "haruna": "kurodate_haruna",
    "koharu": "shimoe_koharu", "wakamo": "kosaka_wakamo", "theresa": "theresa_apocalypse",
    "bronya": "bronya_zaychik", "kanna": "kanna_kamui",
    "kyaru": "kyaru_(princess_connect!)", "kyouka": "kyouka_(princess_connect!)",
    "yuni": "yuni_(princess_connect!)", "rem": "rem_(re:zero)", "ram": "ram_(re:zero)",
    "beatrice": "beatrice_(re:zero)", "illya": "illyasviel_von_einzbern",
    "chloe": "chloe_von_einzbern", "sagiri": "izumi_sagiri", "umaru": "doma_umaru",
    "asuna": "yuuki_asuna", "hinata": "hoshino_hinata", "hana": "shirosaki_hana",
    "noa": "himesaka_noa", "ai": "hinatsuru_ai", "kurumi": "kurumi_tokisaki",
    "kotori": "kotori_itsuka", "tohka": "tohka_yatogami",
    "kagura": "kagura_(onmyouji)", "lucia": "lucia_(punishing:_gray_raven)",
    "shinobu": "oshino_shinobu", "mayoi": "hachikuji_mayoi",
    "kagami": "hiiragi_kagami", "tsukasa": "hiiragi_tsukasa", "nico": "yazawa_nico",
    "yoshiko": "tsushima_yoshiko", "madoka": "kaname_madoka", "homura": "akemi_homura",
    "mami": "tomoe_mami", "sayaka": "miki_sayaka", "renge": "miyauchi_renge",
    "cocoa": "cocoa_hoto", "rize": "rize_tedeza", "chiya": "chiya_ujimatsu",
    "syaro": "syaro_kirima", "mitsuha": "miyamizu_mitsuha", "chino": "chino_kafuu",
    "ilulu": "ilulu_(maid_dragon)", "yoshino": "yoshino_(date_a_live)",
}


def normalize_name(name):
    return NAME_MAP.get(name, name)


def count_images(dataset_dir):
    """统计目录中每个角色的图片数量和总大小"""
    stats = {}
    if not dataset_dir.exists():
        return stats

    for role_dir in dataset_dir.iterdir():
        if not role_dir.is_dir():
            continue

        count = 0
        total_size = 0
        for f in role_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                count += 1
                try:
                    total_size += f.stat().st_size
                except OSError:
                    pass

        if count > 0:
            norm = normalize_name(role_dir.name)
            if norm in stats:
                stats[norm]["count"] += count
                stats[norm]["size_bytes"] += total_size
            else:
                stats[norm] = {
                    "count": count,
                    "size_bytes": total_size,
                    "dir_name": role_dir.name,
                }

    return stats


def create_tables(conn):
    """创建数据库表"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS role_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role_name TEXT UNIQUE NOT NULL,
            dir_name TEXT,
            training_count INTEGER DEFAULT 0,
            final_count INTEGER DEFAULT 0,
            total_count INTEGER DEFAULT 0,
            size_mb REAL DEFAULT 0,
            status TEXT DEFAULT 'insufficient',
            skip_threshold INTEGER DEFAULT 100,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_role_name ON role_stats(role_name);
        CREATE INDEX IF NOT EXISTS idx_total_count ON role_stats(total_count);
        CREATE INDEX IF NOT EXISTS idx_status ON role_stats(status);
    """)


def upsert_role(conn, role_name, dir_name, training_count, final_count, total_count, size_mb):
    """插入或更新角色统计"""
    now = datetime.now().isoformat()

    if total_count >= 100:
        status = "sufficient"
    elif total_count >= 50:
        status = "adequate"
    elif total_count >= 30:
        status = "minimal"
    elif total_count >= 10:
        status = "insufficient"
    else:
        status = "critical"

    conn.execute("""
        INSERT INTO role_stats (role_name, dir_name, training_count, final_count,
                                  total_count, size_mb, status, skip_threshold,
                                  created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 100, ?, ?)
        ON CONFLICT(role_name) DO UPDATE SET
            dir_name=excluded.dir_name,
            training_count=excluded.training_count,
            final_count=excluded.final_count,
            total_count=excluded.total_count,
            size_mb=excluded.size_mb,
            status=excluded.status,
            updated_at=excluded.updated_at
    """, (role_name, dir_name, training_count, final_count, total_count,
          size_mb, status, now, now))


def print_distribution(conn):
    """输出数据分布报告"""
    cur = conn.cursor()

    # 总体
    cur.execute("SELECT COUNT(*), SUM(total_count), AVG(total_count) FROM role_stats")
    row = cur.fetchone()
    total_roles = row[0] or 0
    total_images = row[1] or 0
    avg_images = row[2] or 0

    # 各状态
    cur.execute("""
        SELECT status, COUNT(*), SUM(total_count)
        FROM role_stats
        GROUP BY status
        ORDER BY MIN(total_count)
    """)
    status_rows = cur.fetchall()

    print("=" * 70)
    print("         角色数据采集统计报告 (SQLite)")
    print("=" * 70)

    print(f"\n  数据库: {DEFAULT_DB}")
    print(f"  角色总数: {total_roles}")
    print(f"  图片总数: {total_images}")
    print(f"  平均/角色: {avg_images:.1f}")

    # 分布
    print(f"\n{'─'*70}")
    print(f"  数据分布")
    print(f"{'─'*70}")

    status_labels = {
        "critical": "1-9 张 (严重不足)",
        "insufficient": "10-29 张 (不足)",
        "minimal": "30-49 张 (最低)",
        "adequate": "50-99 张 (充足)",
        "sufficient": "100+ 张 (达标)",
    }

    for status, count, images in status_rows:
        label = status_labels.get(status, status)
        pct = count / total_roles * 100 if total_roles > 0 else 0
        bar = "█" * max(1, int(count * 0.6))
        print(f"  {label:22s}: {count:>3} 角色 ({pct:5.1f}%)  {bar}")

    # 达标率
    cur.execute("SELECT COUNT(*) FROM role_stats WHERE total_count >= 100")
    at_100 = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM role_stats WHERE total_count >= 50")
    at_50 = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM role_stats WHERE total_count >= 30")
    at_30 = cur.fetchone()[0]

    print(f"\n{'─'*70}")
    print(f"  达标情况")
    print(f"{'─'*70}")
    print(f"  ≥100 张 (目标): {at_100}/{total_roles} ({at_100/total_roles*100:.1f}%)")
    print(f"  ≥50 张        : {at_50}/{total_roles} ({at_50/total_roles*100:.1f}%)")
    print(f"  ≥30 张 (最低) : {at_30}/{total_roles} ({at_30/total_roles*100:.1f}%)")

    # 缺口
    cur.execute("SELECT SUM(100 - total_count) FROM role_stats WHERE total_count < 100")
    need = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(*) FROM role_stats WHERE total_count < 100")
    need_roles = cur.fetchone()[0]
    print(f"  距 100 张缺口: {need} 张 ({need_roles} 角色未达标)")

    # Top 10
    print(f"\n{'─'*70}")
    print(f"  Top 10 数据最多")
    print(f"{'─'*70}")
    print(f"  {'角色名':<35s} {'训练':>5s} {'最终':>5s} {'合计':>5s} {'大小':>8s}")
    print(f"  {'─'*58}")
    cur.execute("""
        SELECT role_name, training_count, final_count, total_count, size_mb
        FROM role_stats ORDER BY total_count DESC LIMIT 10
    """)
    for row in cur.fetchall():
        print(f"  {row[0]:<35s} {row[1]:>5d} {row[2]:>5d} {row[3]:>5d} {row[4]:>7.1f}MB")

    # Bottom 10
    print(f"\n{'─'*70}")
    print(f"  Bottom 10 数据最少")
    print(f"{'─'*70}")
    print(f"  {'角色名':<35s} {'训练':>5s} {'最终':>5s} {'合计':>5s} {'状态':>12s}")
    print(f"  {'─'*58}")
    cur.execute("""
        SELECT role_name, training_count, final_count, total_count, status
        FROM role_stats ORDER BY total_count ASC LIMIT 10
    """)
    for row in cur.fetchall():
        print(f"  {row[0]:<35s} {row[1]:>5d} {row[2]:>5d} {row[3]:>5d} {row[4]:>12s}")

    # 数据来源
    cur.execute("SELECT COUNT(*) FROM role_stats WHERE training_count > 0 AND final_count > 0")
    both = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM role_stats WHERE training_count > 0 AND final_count = 0")
    train_only = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM role_stats WHERE training_count = 0 AND final_count > 0")
    final_only = cur.fetchone()[0]

    print(f"\n{'─'*70}")
    print(f"  数据来源")
    print(f"{'─'*70}")
    print(f"  仅训练集: {train_only} 角色")
    print(f"  仅最终集: {final_only} 角色")
    print(f"  两者都有: {both} 角色")

    # 总大小
    cur.execute("SELECT SUM(size_mb) FROM role_stats")
    total_mb = cur.fetchone()[0] or 0
    print(f"\n  数据总大小: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="角色数据统计 → SQLite")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB), help="SQLite 数据库路径")
    parser.add_argument("--json", action="store_true", help="同时导出 JSON")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"扫描训练集: {TRAINING_DIR}")
    training_stats = count_images(TRAINING_DIR)
    print(f"  → {len(training_stats)} 个角色")

    print(f"扫描最终集: {FINAL_DIR}")
    final_stats = count_images(FINAL_DIR)
    print(f"  → {len(final_stats)} 个角色")

    # 合并
    all_roles = set(list(training_stats.keys()) + list(final_stats.keys()))
    print(f"合并去重后: {len(all_roles)} 个角色")

    # 写入数据库
    conn = sqlite3.connect(str(db_path))
    create_tables(conn)

    # 清空旧数据
    conn.execute("DELETE FROM role_stats")

    synced = 0
    for role in sorted(all_roles):
        tr = training_stats.get(role, {"count": 0, "size_bytes": 0, "dir_name": role})
        fi = final_stats.get(role, {"count": 0, "size_bytes": 0, "dir_name": role})

        total = tr["count"] + fi["count"]
        size_mb = (tr["size_bytes"] + fi["size_bytes"]) / (1024 * 1024)
        dir_name = fi["dir_name"] if fi["count"] > 0 else tr["dir_name"]

        upsert_role(conn, role, dir_name, tr["count"], fi["count"], total, round(size_mb, 2))
        synced += 1

    conn.commit()
    print(f"已写入 {synced} 条记录到 {db_path}")

    # 输出报告
    print()
    print_distribution(conn)

    # 导出 JSON
    if args.json:
        cur = conn.cursor()
        cur.execute("""
            SELECT role_name, dir_name, training_count, final_count,
                   total_count, size_mb, status
            FROM role_stats ORDER BY total_count DESC
        """)
        rows = cur.fetchall()
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_roles": len(rows),
            "total_images": sum(r[4] for r in rows),
            "roles": [
                {
                    "role_name": r[0],
                    "dir_name": r[1],
                    "training_count": r[2],
                    "final_count": r[3],
                    "total_count": r[4],
                    "size_mb": r[5],
                    "status": r[6],
                }
                for r in rows
            ],
        }
        json_path = db_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nJSON 报告已保存: {json_path}")

    conn.close()
    print(f"\n✅ 完成")


if __name__ == "__main__":
    main()
