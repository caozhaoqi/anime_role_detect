#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 final_dataset 中的重复角色目录（MD5 去重）

问题：采集脚本使用短名（hifumi）创建目录，已有全名目录（ajitani_hifumi）
      导致同一角色出现两个目录，且图片可能有重复。

策略：
1. 将短名目录的图片移动到全名目录
2. 使用 MD5 检查重复图片，避免重复
3. 合并后裁剪到 150 张上限
4. 删除空的短名目录

Usage:
    python3 scripts/data_cleaning/merge_v2.py --dry-run   # 预览
    python3 scripts/data_cleaning/merge_v2.py             # 执行
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set

PROJECT_ROOT = Path(__file__).parent.parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

# 完整映射：短名 → 全名（保留全名目录）
# key = 短名/当前目录名, value = 目标全名目录名
DUPLICATE_MAP = {
    # 蔚蓝档案 (Blue Archive)
    "aris": "tendou_aris",
    "Aru": "rikuhachima_aru",
    "ako": "amau_ako",
    "seele": "seele_vollerei",
    "hina": "sorasaki_hina",
    "mika": "misono_mika",
    "hoshino": "takanashi_hoshino",
    "azusa": "shirasu_azusa",
    "izuna": "kuda_izuna",
    "shiroko": "sunaookami_shiroko",
    "hifumi": "ajitani_hifumi",
    "nodoka": "amami_nodoka",
    "kayoko": "onikata_kayoko",
    "mutsuki": "asagi_mutsuki",
    "serika": "kuromi_serika",
    "nonomi": "izayoi_nonomi",
    "ayane": "okusora_ayane",
    "haruna": "kurodate_haruna",
    "koharu": "shimoe_koharu",
    "wakamo": "kosaka_wakamo",
    # 崩坏3 (Honkai Impact)
    "theresa": "theresa_apocalypse",
    "bronya": "bronya_zaychik",
    # 龙女仆 (Maid Dragon)
    "kanna": "kanna_kamui",
    # 公主连接 (Princess Connect)
    "kyaru": "kyaru_(princess_connect!)",
    "kyouka": "kyouka_(princess_connect!)",
    "yuni": "yuni_(princess_connect!)",
    "pecorine": "pecorine_(princess_connect!)",
    "yui": "yui_(princess_connect!)",
    "maho": "maho_(princess_connect!)",
    "kokkoro": "kokkoro_(princess_connect!)",
    # Re:Zero
    "rem": "rem_(re:zero)",
    "ram": "ram_(re:zero)",
    "beatrice": "beatrice_(re:zero)",
    # Fate/kaleid
    "illya": "illyasviel_von_einzbern",
    "chloe": "chloe_von_einzbern",
    "miyu": "miyu_edelfelt",
    # 其他
    "sagiri": "izumi_sagiri",
    "umaru": "doma_umaru",
    "asuna": "yuuki_asuna",
    "anya": "anya_forger",
    "hinata": "hoshino_hinata",
    "hana": "shirosaki_hana",
    "noa": "himesaka_noa",
    "ai": "hinatsuru_ai",
    "yoshino": "yoshino_(date_a_live)",
    "kurumi": "kurumi_tokisaki",
    "kotori": "kotori_itsuka",
    "tohka": "tohka_yatogami",
    "kagura": "kagura_(onmyouji)",
    "lucia": "lucia_(punishing:_gray_raven)",
    "shinobu": "oshino_shinobu",
    "mayoi": "hachikuji_mayoi",
    "kagami": "hiiragi_kagami",
    "tsukasa": "hiiragi_tsukasa",
    "nico": "yazawa_nico",
    "yoshiko": "tsushima_yoshiko",
    "madoka": "kaname_madoka",
    "homura": "akemi_homura",
    "mami": "tomoe_mami",
    "sayaka": "miki_sayaka",
    "renge": "miyauchi_renge",
    "cocoa": "cocoa_hoto",
    "rize": "rize_tedeza",
    "chiya": "chiya_ujimatsu",
    "syaro": "syaro_kirima",
    "mitsuha": "miyamizu_mitsuha",
    "chino": "chino_kafuu",
    "ilulu": "ilulu_(maid_dragon)",
}


def compute_md5(filepath: Path) -> str:
    """计算文件 MD5"""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def count_images(dir_path: Path) -> int:
    """统计目录中图片数量"""
    if not dir_path.exists():
        return 0
    return sum(1 for f in dir_path.iterdir()
               if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def get_all_hashes(dir_path: Path) -> Set[str]:
    """获取目录中所有图片的 MD5 集合"""
    hashes = set()
    if not dir_path.exists():
        return hashes
    for f in dir_path.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            hashes.add(compute_md5(f))
    return hashes


def merge_roles(short_name: str, full_name: str, max_count: int = 150, dry_run: bool = True) -> Dict:
    """
    合并两个目录：将 short_name 的图片移动到 full_name
    使用 MD5 去重，合并后裁剪到 max_count
    """
    short_dir = FINAL_DIR / short_name
    full_dir = FINAL_DIR / full_name

    result = {
        "short_name": short_name,
        "full_name": full_name,
        "short_before": 0,
        "full_before": 0,
        "moved": 0,
        "skipped_dup": 0,
        "deleted_after_trim": 0,
        "short_deleted": False,
        "success": False,
    }

    if not short_dir.exists():
        return result

    result["short_before"] = count_images(short_dir)
    result["full_before"] = count_images(full_dir)

    # 短名目录为空，直接删除
    if result["short_before"] == 0:
        if dry_run:
            print(f"  [dry-run] {short_name}(空) → 删除目录")
        else:
            try:
                short_dir.rmdir()
                result["short_deleted"] = True
            except OSError:
                pass
        result["success"] = True
        return result

    # 收集全名目录已有图片的 MD5
    full_hashes = get_all_hashes(full_dir) if full_dir.exists() else set()

    # 创建全名目录
    if not dry_run:
        os.makedirs(full_dir, exist_ok=True)

    # 移动图片（MD5 去重）
    for f in sorted(short_dir.iterdir()):
        if not (f.is_file() and f.suffix.lower() in IMAGE_EXTS):
            continue

        if dry_run:
            result["moved"] += 1
            continue

        # 计算 MD5 并检查重复
        try:
            file_hash = compute_md5(f)
        except Exception:
            continue

        if file_hash in full_hashes:
            result["skipped_dup"] += 1
            continue

        # 移动到全名目录，若同名则加后缀
        dest = full_dir / f.name
        if dest.exists():
            dest = full_dir / f"{f.stem}_dup{f.suffix}"

        shutil.move(str(f), str(dest))
        full_hashes.add(file_hash)
        result["moved"] += 1

    # 合并后裁剪到 max_count
    if not dry_run:
        total_after = result["full_before"] + result["moved"]
        if total_after > max_count:
            # 获取所有图片并随机裁剪
            all_images = sorted([
                (f, compute_md5(f)) for f in full_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            ], key=lambda x: x[0].name)  # 按文件名排序，保证可重复
            to_keep = max_count
            to_delete = total_after - to_keep
            # 保留前 max_count 张（确定性）
            for f, _ in all_images[to_keep:]:
                try:
                    f.unlink()
                    result["deleted_after_trim"] += 1
                except OSError:
                    pass

    # 删除短名目录
    if not dry_run:
        try:
            # 检查是否还有残留文件
            remaining = [f for f in short_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
            if not remaining:
                shutil.rmtree(str(short_dir))
                result["short_deleted"] = True
        except OSError:
            pass

    result["success"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description="合并重复角色目录 v2（MD5去重）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览")
    parser.add_argument("--max", type=int, default=150, help="合并后每角色最大图片数")
    parser.add_argument("--output", type=str, help="保存结果到 JSON 文件")
    args = parser.parse_args()

    mode = "预览" if args.dry_run else "执行"
    print(f"合并重复角色目录 v2 — {mode}")
    print(f"数据集: {FINAL_DIR}")
    print(f"上限: {args.max} 张/角色")
    print("=" * 70)

    # 检查实际存在的重复目录
    existing_pairs = []
    for short_name, full_name in DUPLICATE_MAP.items():
        short_dir = FINAL_DIR / short_name
        full_dir = FINAL_DIR / full_name
        if short_dir.exists() or full_dir.exists():
            sc = count_images(short_dir) if short_dir.exists() else 0
            fc = count_images(full_dir) if full_dir.exists() else 0
            if sc > 0 or fc > 0:
                existing_pairs.append((short_name, full_name, sc, fc))

    if not existing_pairs:
        print("未发现需要合并的重复目录")
        return

    print(f"发现 {len(existing_pairs)} 组重复目录:")
    for sn, fn, sc, fc in existing_pairs:
        arrow = "→" if sc > 0 else " (空)"
        print(f"  {sn}({sc:>3}) {arrow} {fn}({fc:>3})")
    print()

    # 执行合并
    results = []
    total_moved = 0
    total_skipped = 0
    total_trimmed = 0

    for short_name, full_name, sc, fc in existing_pairs:
        result = merge_roles(short_name, full_name, args.max, args.dry_run)
        results.append(result)

        status = "✅" if result["success"] else "❌"
        short_info = f"{short_name}({result['short_before']})"
        full_info = f"{full_name}"
        details = []
        if result["moved"] > 0:
            details.append(f"移动+{result['moved']}")
        if result["skipped_dup"] > 0:
            details.append(f"跳过重复{result['skipped_dup']}")
        if result["deleted_after_trim"] > 0:
            details.append(f"裁剪-{result['deleted_after_trim']}")
        if result["short_deleted"]:
            details.append("删除短名目录")

        detail_str = ", ".join(details) if details else "无变化"
        print(f"  {status} {short_info} → {full_info}: {detail_str}")

        total_moved += result["moved"]
        total_skipped += result["skipped_dup"]
        total_trimmed += result["deleted_after_trim"]

    # 汇总
    print(f"\n{'=' * 70}")
    print(f"汇总 ({mode}):")
    print(f"  合并组数: {len(results)}")
    print(f"  移动图片: {total_moved} 张")
    print(f"  跳过重复: {total_skipped} 张")
    print(f"  裁剪超限: {total_trimmed} 张")

    # 最终统计
    if not args.dry_run:
        final_roles = 0
        final_images = 0
        for d in sorted(FINAL_DIR.iterdir()):
            if not d.is_dir():
                continue
            cnt = count_images(d)
            if cnt > 0:
                final_roles += 1
                final_images += cnt

        print(f"\n合并后统计:")
        print(f"  角色数: {final_roles}")
        print(f"  图片总数: {final_images}")

    # 保存结果
    if args.output:
        output_data = {
            "mode": mode,
            "timestamp": os.popen("date").read().strip(),
            "max_per_role": args.max,
            "total_groups": len(results),
            "total_moved": total_moved,
            "total_skipped_duplicates": total_skipped,
            "total_trimmed": total_trimmed,
            "details": results,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.output}")

    if args.dry_run:
        print(f"\n执行: python3 scripts/data_cleaning/merge_v2.py")


if __name__ == "__main__":
    main()