#!/usr/bin/env python3
"""使用修复后的NSFW检测服务处理数据集"""

import os
import sys
import json
from pathlib import Path

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.nsfw_detector import detect_nsfw

DATASET_PATH = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset')
OUTPUT_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/nsfw_results')

def process_dataset():
    """处理final_dataset"""
    output_path = OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    # 收集所有图片
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = []
    
    for char_dir in DATASET_PATH.iterdir():
        if not char_dir.is_dir():
            continue
        
        for ext in image_extensions:
            image_files.extend(char_dir.glob(f"*{ext}"))

    print(f"📁 找到 {len(image_files)} 张图片")

    nsfw_count = 0
    suggestive_count = 0
    safe_count = 0
    error_count = 0
    results = []
    methods = []

    total = len(image_files)
    for i, img_path in enumerate(image_files):
        result = detect_nsfw(str(img_path))
        
        if result is None:
            results.append({
                "path": str(img_path),
                "character": img_path.parent.name,
                "score": 0.0,
                "label": "错误",
                "method": "error"
            })
            error_count += 1
            methods.append("error")
            continue
        
        methods.append(result.get('method', 'unknown'))
        
        # 根据NSFW得分判断类别
        nsfw_score = result.get('nsfw_score', 0)
        if nsfw_score > 0.6:
            label = "NSFW"
            nsfw_count += 1
        elif nsfw_score > 0.4:
            label = "Suggestive"
            suggestive_count += 1
        else:
            label = "Safe"
            safe_count += 1

        results.append({
            "path": str(img_path),
            "character": img_path.parent.name,
            "score": nsfw_score,
            "label": label,
            "method": result.get('method'),
            "is_nsfw": result.get('is_nsfw'),
            "skin_ratio": result.get('skin_ratio'),
            "details": result.get('details')
        })

        if (i + 1) % 50 == 0:
            print(
                f"   已处理: {i + 1}/{total} | NSFW: {nsfw_count} | Suggestive: {suggestive_count} | Safe: {safe_count}"
            )

    print(f"\n✅ 处理完成!")
    print(f"=" * 60)
    print(f"总图片数: {len(image_files)}")
    print(f"NSFW: {nsfw_count} ({nsfw_count/len(image_files)*100:.1f}%)")
    print(f"Suggestive: {suggestive_count} ({suggestive_count/len(image_files)*100:.1f}%)")
    print(f"Safe: {safe_count} ({safe_count/len(image_files)*100:.1f}%)")
    print(f"错误: {error_count}")
    
    print(f"\n检测方法分布:")
    print(f"   opencv_based: {methods.count('opencv_based')}")
    print(f"   rule_based: {methods.count('rule_based')}")
    print(f"   error: {methods.count('error')}")

    # 保存结果
    with open(output_path / "nsfw_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = {
        "total_images": len(image_files),
        "nsfw_count": nsfw_count,
        "suggestive_count": suggestive_count,
        "safe_count": safe_count,
        "detection_method": "OpenCV-based NSFW Detection",
        "thresholds": {"NSFW": ">0.6", "Suggestive": "0.4-0.6", "Safe": "<0.4"}
    }

    with open(output_path / "detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 按角色统计
    char_stats = {}
    for result in results:
        char = result["character"]
        if char not in char_stats:
            char_stats[char] = {"total": 0, "nsfw": 0, "suggestive": 0, "safe": 0}
        char_stats[char]["total"] += 1
        if result["label"] == "NSFW":
            char_stats[char]["nsfw"] += 1
        elif result["label"] == "Suggestive":
            char_stats[char]["suggestive"] += 1
        else:
            char_stats[char]["safe"] += 1

    print(f"\n📊 各角色NSFW分布:")
    print(f"{'角色名称':<30} {'总数':>6} {'NSFW':>6} {'Suggestive':>12} {'Safe':>6}")
    print(f"-" * 70)
    for char, stats in sorted(char_stats.items(), key=lambda x: x[1]["nsfw"], reverse=True):
        print(f"{char:<30} {stats['total']:>6} {stats['nsfw']:>6} {stats['suggestive']:>12} {stats['safe']:>6}")

    print(f"\n📁 检测结果已保存到 {output_path}")

    return summary


if __name__ == "__main__":
    process_dataset()