#!/usr/bin/env python3
"""标注数据统计分析"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.database.init_db import init_database, Annotation, Sample


def analyze_annotations():
    """分析标注数据"""
    # 初始化数据库
    engine, Session = init_database()
    session = Session()

    # 获取所有标注记录
    annotations = session.query(Annotation).all()
    print(f"📊 共 {len(annotations)} 条标注记录\n")

    if not annotations:
        print("❌ 暂无标注数据")
        return

    # ========== 1. 平均bbox大小 ==========
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []

    for ann in annotations:
        if ann.bbox:
            x1, y1, x2, y2 = ann.bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height

            bbox_widths.append(width)
            bbox_heights.append(height)
            bbox_areas.append(area)

    print("=" * 50)
    print("📐 1. 平均bbox大小")
    print("=" * 50)
    print(f"平均宽度:  {sum(bbox_widths)/len(bbox_widths):.2f} 像素")
    print(f"平均高度:  {sum(bbox_heights)/len(bbox_heights):.2f} 像素")
    print(f"平均面积:  {sum(bbox_areas)/len(bbox_areas):.2f} 平方像素")
    print(f"最小面积:  {min(bbox_areas):.2f} 平方像素")
    print(f"最大面积:  {max(bbox_areas):.2f} 平方像素")
    print(f"面积中位数: {sorted(bbox_areas)[len(bbox_areas)//2]:.2f} 平方像素")

    # ========== 2. 角色在图片中的位置分布 ==========
    bbox_centers_x = []
    bbox_centers_y = []
    bbox_areas_ratio = []

    # 获取所有样本信息
    samples = session.query(Sample).all()
    sample_dict = {s.id: s for s in samples}

    for ann in annotations:
        if ann.bbox and ann.sample_id in sample_dict:
            x1, y1, x2, y2 = ann.bbox
            sample = sample_dict[ann.sample_id]

            if sample.width and sample.height:
                # 计算中心点（归一化到0-1）
                center_x = (x1 + x2) / 2 / sample.width
                center_y = (y1 + y2) / 2 / sample.height
                area_ratio = (x2 - x1) * (y2 - y1) / (sample.width * sample.height)

                bbox_centers_x.append(center_x)
                bbox_centers_y.append(center_y)
                bbox_areas_ratio.append(area_ratio)

    print("\n" + "=" * 50)
    print("📍 2. 角色在图片中的位置分布（归一化坐标）")
    print("=" * 50)
    print(f"中心X坐标:")
    print(f"  - 平均值: {sum(bbox_centers_x)/len(bbox_centers_x):.3f}")
    print(f"  - 最小值: {min(bbox_centers_x):.3f}")
    print(f"  - 最大值: {max(bbox_centers_x):.3f}")
    print(f"  - 中位数: {sorted(bbox_centers_x)[len(bbox_centers_x)//2]:.3f}")

    print(f"\n中心Y坐标:")
    print(f"  - 平均值: {sum(bbox_centers_y)/len(bbox_centers_y):.3f}")
    print(f"  - 最小值: {min(bbox_centers_y):.3f}")
    print(f"  - 最大值: {max(bbox_centers_y):.3f}")
    print(f"  - 中位数: {sorted(bbox_centers_y)[len(bbox_centers_y)//2]:.3f}")

    print(f"\nbbox面积占图片比例:")
    print(f"  - 平均值: {sum(bbox_areas_ratio)/len(bbox_areas_ratio)*100:.2f}%")
    print(f"  - 最小值: {min(bbox_areas_ratio)*100:.2f}%")
    print(f"  - 最大值: {max(bbox_areas_ratio)*100:.2f}%")
    print(f"  - 中位数: {sorted(bbox_areas_ratio)[len(bbox_areas_ratio)//2]*100:.2f}%")

    # 位置分布统计
    x_distribution = {
        "左侧 (<0.33)": sum(1 for x in bbox_centers_x if x < 0.33),
        "中间 (0.33-0.67)": sum(1 for x in bbox_centers_x if 0.33 <= x < 0.67),
        "右侧 (>0.67)": sum(1 for x in bbox_centers_x if x >= 0.67)
    }
    y_distribution = {
        "上方 (<0.33)": sum(1 for y in bbox_centers_y if y < 0.33),
        "中间 (0.33-0.67)": sum(1 for y in bbox_centers_y if 0.33 <= y < 0.67),
        "下方 (>0.67)": sum(1 for y in bbox_centers_y if y >= 0.67)
    }

    print(f"\n水平位置分布:")
    for region, count in x_distribution.items():
        print(f"  - {region}: {count} ({count/len(bbox_centers_x)*100:.1f}%)")

    print(f"\n垂直位置分布:")
    for region, count in y_distribution.items():
        print(f"  - {region}: {count} ({count/len(bbox_centers_y)*100:.1f}%)")

    # ========== 3. 检测置信度分布 ==========
    confidences = [ann.confidence for ann in annotations if ann.confidence is not None]

    print("\n" + "=" * 50)
    print("🎯 3. 检测置信度分布")
    print("=" * 50)
    print(f"平均置信度: {sum(confidences)/len(confidences):.4f}")
    print(f"最小置信度: {min(confidences):.4f}")
    print(f"最大置信度: {max(confidences):.4f}")
    print(f"置信度中位数: {sorted(confidences)[len(confidences)//2]:.4f}")

    # 置信度分位数
    sorted_conf = sorted(confidences)
    print(f"\n置信度分位数:")
    print(f"  - 25%: {sorted_conf[int(len(sorted_conf)*0.25)]:.4f}")
    print(f"  - 50%: {sorted_conf[int(len(sorted_conf)*0.5)]:.4f}")
    print(f"  - 75%: {sorted_conf[int(len(sorted_conf)*0.75)]:.4f}")
    print(f"  - 90%: {sorted_conf[int(len(sorted_conf)*0.9)]:.4f}")
    print(f"  - 95%: {sorted_conf[int(len(sorted_conf)*0.95)]:.4f}")

    # 置信度区间分布
    conf_distribution = {
        "低置信度 (<0.5)": sum(1 for c in confidences if c < 0.5),
        "中置信度 (0.5-0.7)": sum(1 for c in confidences if 0.5 <= c < 0.7),
        "高置信度 (0.7-0.85)": sum(1 for c in confidences if 0.7 <= c < 0.85),
        "极高置信度 (>=0.85)": sum(1 for c in confidences if c >= 0.85)
    }

    print(f"\n置信度区间分布:")
    for range_name, count in conf_distribution.items():
        print(f"  - {range_name}: {count} ({count/len(confidences)*100:.1f}%)")

    # ========== 4. 多角色图片比例 ==========
    # 统计每个样本的标注数量
    sample_annotation_count = defaultdict(int)
    for ann in annotations:
        sample_annotation_count[ann.sample_id] += 1

    multi_person_samples = sum(1 for count in sample_annotation_count.values() if count > 1)
    single_person_samples = sum(1 for count in sample_annotation_count.values() if count == 1)
    total_samples = len(sample_annotation_count)

    print("\n" + "=" * 50)
    print("👥 4. 多角色图片比例")
    print("=" * 50)
    print(f"总样本数: {total_samples}")
    print(f"单角色图片: {single_person_samples} ({single_person_samples/total_samples*100:.1f}%)")
    print(f"多角色图片: {multi_person_samples} ({multi_person_samples/total_samples*100:.1f}%)")

    # 多角色图片的详细统计
    person_count_distribution = defaultdict(int)
    for count in sample_annotation_count.values():
        person_count_distribution[count] += 1

    print(f"\n每张图片的角色数量分布:")
    for person_count in sorted(person_count_distribution.keys()):
        sample_count = person_count_distribution[person_count]
        print(f"  - {person_count}个角色: {sample_count} 张图片 ({sample_count/total_samples*100:.1f}%)")

    # 最多角色的图片
    max_person_count = max(person_count_distribution.keys())
    if max_person_count > 1:
        print(f"\n最多检测到 {max_person_count} 个角色的图片")

    # ========== 5. 标注来源统计 ==========
    annotator_distribution = defaultdict(int)
    for ann in annotations:
        annotator_distribution[ann.annotator] += 1

    print("\n" + "=" * 50)
    print("🏷️  5. 标注来源统计")
    print("=" * 50)
    for annotator, count in annotator_distribution.items():
        print(f"  - {annotator}: {count} ({count/len(annotations)*100:.1f}%)")

    # ========== 6. 验证状态统计 ==========
    verified_count = sum(1 for ann in annotations if ann.is_verified)
    unverified_count = len(annotations) - verified_count

    print("\n" + "=" * 50)
    print("✅ 6. 验证状态统计")
    print("=" * 50)
    print(f"已验证: {verified_count} ({verified_count/len(annotations)*100:.1f}%)")
    print(f"未验证: {unverified_count} ({unverified_count/len(annotations)*100:.1f}%)")

    # ========== 保存统计结果 ==========
    stats = {
        "total_annotations": len(annotations),
        "bbox_stats": {
            "avg_width": sum(bbox_widths)/len(bbox_widths),
            "avg_height": sum(bbox_heights)/len(bbox_heights),
            "avg_area": sum(bbox_areas)/len(bbox_areas),
            "min_area": min(bbox_areas),
            "max_area": max(bbox_areas),
            "median_area": sorted(bbox_areas)[len(bbox_areas)//2]
        },
        "position_distribution": {
            "avg_center_x": sum(bbox_centers_x)/len(bbox_centers_x),
            "avg_center_y": sum(bbox_centers_y)/len(bbox_centers_y),
            "avg_area_ratio": sum(bbox_areas_ratio)/len(bbox_areas_ratio),
            "x_distribution": x_distribution,
            "y_distribution": y_distribution
        },
        "confidence_stats": {
            "avg": sum(confidences)/len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "median": sorted(confidences)[len(confidences)//2],
            "percentiles": {
                "25": sorted_conf[int(len(sorted_conf)*0.25)],
                "50": sorted_conf[int(len(sorted_conf)*0.5)],
                "75": sorted_conf[int(len(sorted_conf)*0.75)],
                "90": sorted_conf[int(len(sorted_conf)*0.9)],
                "95": sorted_conf[int(len(sorted_conf)*0.95)]
            },
            "distribution": conf_distribution
        },
        "multi_person_stats": {
            "total_samples": total_samples,
            "single_person": single_person_samples,
            "multi_person": multi_person_samples,
            "person_count_distribution": dict(person_count_distribution)
        },
        "annotator_stats": dict(annotator_distribution),
        "verification_stats": {
            "verified": verified_count,
            "unverified": unverified_count
        }
    }

    # 保存到JSON文件
    output_path = "data/annotation_stats.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n💾 统计结果已保存到: {output_path}")

    session.close()
    engine.dispose()


if __name__ == "__main__":
    analyze_annotations()