#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase1 T5: 用户纠错反馈数据统计脚本。

读取 logs/feedback/feedback_*.jsonl，统计纠错分布，供后续增量训练参考。

用法:
    python scripts/load_feedback.py              # 统计全部
    python scripts/load_feedback.py --date 2026-08-02  # 统计指定日期
    python scripts/load_feedback.py --export corrections.csv  # 导出 CSV
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEEDBACK_DIR = os.path.join(PROJECT_ROOT, "logs", "feedback")


def load_records(date_filter: str | None = None) -> list[dict]:
    """加载所有 feedback JSONL 记录。"""
    pattern = os.path.join(FEEDBACK_DIR, "feedback_*.jsonl")
    files = sorted(glob.glob(pattern))
    if date_filter:
        files = [f for f in files if date_filter in os.path.basename(f)]

    records: list[dict] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"  [WARN] 跳过无效 JSON: {fp}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  [ERROR] 读取失败 {fp}: {e}", file=sys.stderr)
    return records


def summarize(records: list[dict]) -> dict:
    """统计纠错分布。"""
    total = len(records)
    if total == 0:
        return {"total": 0, "message": "暂无纠错数据"}

    # 按原始预测 → 纠正标签 统计
    correction_pairs: Counter = Counter()
    # 按纠正标签统计（哪些角色被纠错最多）
    corrected_counts: Counter = Counter()
    # 按原始预测统计（哪些角色最容易出错）
    original_counts: Counter = Counter()
    # 按端点统计
    endpoint_counts: Counter = Counter()
    # 高置信度但被纠错（潜在噪声 / 难例）
    high_conf_corrections: list[dict] = []
    # 带有效图像缓存（image_ref 非空）的记录数，用于确认 Phase2 图像缓存生效
    cached_images: int = 0

    for r in records:
        orig = r.get("original_prediction", "unknown")
        corrected = r.get("corrected_label", "unknown")
        conf = r.get("original_confidence", 0.0)
        endpoint = r.get("endpoint", "unknown")

        correction_pairs[(orig, corrected)] += 1
        corrected_counts[corrected] += 1
        original_counts[orig] += 1
        endpoint_counts[endpoint] += 1

        if r.get("image_ref"):
            cached_images += 1

        if conf >= 0.9:
            high_conf_corrections.append(r)

    return {
        "total": total,
        "unique_corrections": len(correction_pairs),
        "top_corrected_roles": corrected_counts.most_common(10),
        "top_error_roles": original_counts.most_common(10),
        "top_correction_pairs": [
            {"from": p[0], "to": p[1], "count": c}
            for p, c in correction_pairs.most_common(15)
        ],
        "by_endpoint": dict(endpoint_counts),
        "cached_images": cached_images,
        "high_confidence_corrections": len(high_conf_corrections),
        "high_confidence_details": high_conf_corrections[:5],
    }


def export_csv(records: list[dict], output_path: str) -> None:
    """导出为 CSV 供训练脚本消费。"""
    if not records:
        print("无数据可导出")
        return
    fields = [
        "recognition_id", "endpoint", "original_prediction",
        "original_confidence", "corrected_label", "image_ref",
        "timestamp", "server_timestamp",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"已导出 {len(records)} 条记录到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="用户纠错反馈数据统计")
    parser.add_argument("--date", type=str, default=None, help="筛选日期 (YYYY-MM-DD)")
    parser.add_argument("--export", type=str, default=None, help="导出 CSV 路径")
    args = parser.parse_args()

    if not os.path.exists(FEEDBACK_DIR):
        print(f"反馈目录不存在: {FEEDBACK_DIR}")
        print("（尚无用户纠错数据，端点刚上线）")
        return

    records = load_records(args.date)
    print(f"\n{'='*60}")
    print(f"用户纠错反馈统计")
    print(f"{'='*60}")
    print(f"数据目录: {FEEDBACK_DIR}")
    print(f"记录总数: {len(records)}")

    summary = summarize(records)
    if summary["total"] == 0:
        print("\n暂无纠错数据。")
        return

    print(f"独立纠错对: {summary['unique_corrections']}")
    print(f"带缓存图像: {summary['cached_images']} 条")
    print(f"\n--- 被纠错最多的角色（Top 10）---")
    for role, count in summary["top_corrected_roles"]:
        print(f"  {role}: {count} 次")

    print(f"\n--- 原始预测出错最多的角色（Top 10）---")
    for role, count in summary["top_error_roles"]:
        print(f"  {role}: {count} 次")

    print(f"\n--- 最常见纠错对（Top 15）---")
    for pair in summary["top_correction_pairs"]:
        print(f"  {pair['from']} → {pair['to']}: {pair['count']} 次")

    print(f"\n--- 按端点统计 ---")
    for ep, count in summary["by_endpoint"].items():
        print(f"  {ep}: {count}")

    print(f"\n--- 高置信度纠错（原置信≥0.9，潜在噪声/难例）---")
    print(f"  共 {summary['high_confidence_corrections']} 条")
    for r in summary["high_confidence_details"]:
        print(f"  {r.get('original_prediction')} ({r.get('original_confidence',0):.2f}) → {r.get('corrected_label')}")

    if args.export:
        export_csv(records, args.export)

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
