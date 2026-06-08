#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline 质量检测报告
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.database.init_db import init_database, Sample, Annotation, Character
from sqlalchemy import func

engine, Session = init_database()
session = Session()

print("="*70)
print("📊 Pipeline 质量检测报告")
print("="*70)

# 1. 样本总数
total_samples = session.query(Sample).count()
print(f"\n📁 样本总数: {total_samples}")

# 2. 样本状态分布
print("\n📊 样本状态分布:")
status_counts = session.query(
    Sample.status,
    func.count(Sample.id)
).group_by(Sample.status).all()

for status, count in status_counts:
    pct = count / total_samples * 100 if total_samples > 0 else 0
    bar = "█" * int(pct / 5)
    print(f"  {status:20s}: {count:5d} ({pct:5.1f}%) {bar}")

# 3. 各角色样本数
print("\n📊 各角色样本数 (Top 15):")
char_counts = session.query(
    Sample.character_id,
    func.count(Sample.id)
).group_by(Sample.character_id).order_by(func.count(Sample.id).desc()).limit(15).all()

for char_id, count in char_counts:
    char = session.query(Character).get(char_id)
    if char:
        bar = "█" * min(int(count / 100), 20)
        print(f"  {char.name:20s}: {count:5d} {bar}")

# 4. 标注统计
total_annotations = session.query(Annotation).count()
print(f"\n📝 标注统计:")
print(f"  总标注数: {total_annotations}")

if total_annotations > 0:
    # 置信度统计
    confidences = [a.confidence for a in session.query(Annotation).all() if a.confidence]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)

        print(f"  平均置信度: {avg_conf:.3f}")
        print(f"  最小置信度: {min_conf:.3f}")
        print(f"  最大置信度: {max_conf:.3f}")

        # 置信度分布
        high_conf = len([c for c in confidences if c >= 0.8])
        mid_conf = len([c for c in confidences if 0.5 <= c < 0.8])
        low_conf = len([c for c in confidences if c < 0.5])

        print(f"\n  置信度分布:")
        print(f"    高 (≥0.8): {high_conf} ({high_conf/len(confidences)*100:.1f}%)")
        print(f"    中 (0.5-0.8): {mid_conf} ({mid_conf/len(confidences)*100:.1f}%)")
        print(f"    低 (<0.5): {low_conf} ({low_conf/len(confidences)*100:.1f}%)")

# 5. 角色统计
total_characters = session.query(Character).count()
print(f"\n👥 角色统计:")
print(f"  总角色数: {total_characters}")

# 6. 数据质量评估
print("\n✅ 数据质量评估:")

# 评估标准
quality_score = 0
issues = []

# 样本数量评估
if total_samples >= 1000:
    quality_score += 25
    print("  ✓ 样本数量充足 (≥1000)")
elif total_samples >= 500:
    quality_score += 15
    print("  ⚠ 样本数量一般 (500-1000)")
else:
    issues.append("样本数量不足")
    print("  ✗ 样本数量不足 (<500)")

# 标注覆盖率
if total_samples > 0:
    annotation_rate = total_annotations / total_samples
    if annotation_rate >= 0.5:
        quality_score += 25
        print(f"  ✓ 标注覆盖率高 ({annotation_rate*100:.1f}%)")
    elif annotation_rate >= 0.2:
        quality_score += 15
        print(f"  ⚠ 标注覆盖率一般 ({annotation_rate*100:.1f}%)")
    else:
        issues.append("标注覆盖率低")
        print(f"  ✗ 标注覆盖率低 ({annotation_rate*100:.1f}%)")

# 置信度评估
if confidences:
    if avg_conf >= 0.7:
        quality_score += 25
        print(f"  ✓ 标注质量高 (平均置信度 {avg_conf:.3f})")
    elif avg_conf >= 0.5:
        quality_score += 15
        print(f"  ⚠ 标注质量一般 (平均置信度 {avg_conf:.3f})")
    else:
        issues.append("标注质量低")
        print(f"  ✗ 标注质量低 (平均置信度 {avg_conf:.3f})")

# 角色分布
if char_counts:
    max_count = char_counts[0][1] if char_counts else 0
    min_count = char_counts[-1][1] if char_counts else 0
    if max_count > 0 and min_count > 0:
        ratio = max_count / min_count
        if ratio <= 5:
            quality_score += 25
            print(f"  ✓ 角色分布均衡 (最大/最小 = {ratio:.1f})")
        elif ratio <= 10:
            quality_score += 15
            print(f"  ⚠ 角色分布不均衡 (最大/最小 = {ratio:.1f})")
        else:
            issues.append("角色分布严重不均衡")
            print(f"  ✗ 角色分布严重不均衡 (最大/最小 = {ratio:.1f})")

print(f"\n📈 总体质量评分: {quality_score}/100")

if quality_score >= 80:
    print("  评级: 优秀 ⭐⭐⭐⭐⭐")
elif quality_score >= 60:
    print("  评级: 良好 ⭐⭐⭐⭐")
elif quality_score >= 40:
    print("  评级: 一般 ⭐⭐⭐")
else:
    print("  评级: 需改进 ⭐⭐")

if issues:
    print(f"\n⚠️ 需要改进的问题:")
    for issue in issues:
        print(f"  - {issue}")

print("\n" + "="*70)

session.close()
