#!/usr/bin/env python3
"""数据集统计脚本 - 分析训练集和验证集分布"""
import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def count_images(directory: Path) -> int:
    """统计目录中的图片数量"""
    count = 0
    for file in directory.iterdir():
        if file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            count += 1
    return count


def analyze_dataset(dataset_dir: Path) -> Dict[str, int]:
    """分析数据集，返回角色图片数量"""
    stats = {}
    for role_dir in sorted(dataset_dir.iterdir()):
        if role_dir.is_dir():
            role_name = role_dir.name
            img_count = count_images(role_dir)
            stats[role_name] = img_count
    return stats


def generate_report(stats: Dict[str, int]) -> str:
    """生成统计报告"""
    total_images = sum(stats.values())
    total_roles = len(stats)
    avg_images = total_images / total_roles if total_roles > 0 else 0

    report = []
    report.append("=" * 60)
    report.append("数据集统计报告")
    report.append("=" * 60)
    report.append(f"总角色数: {total_roles}")
    report.append(f"总图片数: {total_images}")
    report.append(f"平均每角色: {avg_images:.1f} 张")
    report.append("")

    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)

    report.append("角色图片数量分布:")
    report.append("-" * 60)
    for role, count in sorted_stats:
        report.append(f"{role:<30} {count:>4} 张")
    report.append("")

    intervals = [
        (0, 10, "0-10张"),
        (10, 30, "10-30张"),
        (30, 50, "30-50张"),
        (50, 100, "50-100张"),
        (100, float('inf'), "100+张"),
    ]

    report.append("区间分布:")
    report.append("-" * 60)
    for min_val, max_val, label in intervals:
        count = sum(1 for c in stats.values() if min_val <= c < max_val)
        report.append(f"{label:<10} {count:>4} 个角色")
    report.append("")

    report.append("不足50张的角色:")
    report.append("-" * 60)
    insufficient = [role for role, count in stats.items() if count < 50]
    if insufficient:
        for role in sorted(insufficient):
            report.append(f"  - {role} ({stats[role]} 张)")
    else:
        report.append("  无")
    report.append("")

    report.append("超过100张的角色:")
    report.append("-" * 60)
    abundant = [role for role, count in stats.items() if count >= 100]
    if abundant:
        for role in sorted(abundant):
            report.append(f"  - {role} ({stats[role]} 张)")
    else:
        report.append("  无")

    return "\n".join(report)


def save_report(report: str, output_path: Path) -> None:
    """保存报告到文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"报告已保存: {output_path}")


def main():
    project_root = Path(__file__).parent.parent.parent

    datasets = {
        "训练集": project_root / "data" / "training_dataset",
        "最终数据集": project_root / "data" / "final_dataset",
    }

    all_reports = []

    for name, dataset_dir in datasets.items():
        if not dataset_dir.exists():
            logger.warning(f"数据集不存在: {dataset_dir}")
            continue

        logger.info(f"分析 {name}: {dataset_dir}")
        stats = analyze_dataset(dataset_dir)
        report = generate_report(stats)
        all_reports.append(f"\n\n{name}\n{report}")

        report_path = project_root / "logs" / f"{name.lower().replace(' ', '_')}_stats.txt"
        save_report(report, report_path)

    full_report = "\n".join(all_reports)
    full_report_path = project_root / "logs" / "dataset_stats.txt"
    save_report(full_report, full_report_path)

    logger.info("\n" + full_report)


if __name__ == "__main__":
    main()