#!/usr/bin/env python3
"""检查 training_dataset 样本分布"""
import json
from pathlib import Path

with open("models/efficientnet_b3_anime_20260616_132028/training_results.json") as f:
    r = json.load(f)
names = r["class_names"]

train_dir = Path("data/training_dataset")
final_dir = Path("data/final_dataset")

total = 0
counts = {}
missing = []
for cls in names:
    d = train_dir / cls
    if d.exists():
        n = len(list(d.glob("*")))
        counts[cls] = n
        total += n
    else:
        counts[cls] = 0
        missing.append(cls)

print(f"training_dataset 总样本数: {total}")
print(f"平均每类: {total // len(names)}")
print(f"缺失类别: {len(missing)} -> {missing[:10] if missing else '无'}")
print()

final_total = 0
for cls in names:
    d = final_dir / cls
    if d.exists():
        final_total += len(list(d.glob("*")))
print(f"final_dataset 总样本数: {final_total}")
print()

print("每个类别样本数:")
for cls in names:
    n = counts[cls]
    if n >= 500:
        status = "✅"
    elif n >= 100:
        status = "⚠️"
    elif n > 0:
        status = "🔸"
    else:
        status = "❌"
    bar = "█" * min(n // 50, 40)
    print(f"  [{n:5d}] {bar} {status} {cls}")

low = [(c, counts[c]) for c in names if counts[c] < 100]
print(f"\n< 100 张的类别: {len(low)}")
for c, n in low:
    print(f"  {c}: {n}")