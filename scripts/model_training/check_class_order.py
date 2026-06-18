#!/usr/bin/env python3
"""
训练数据诊断 — 检查 class_to_idx 一致性和数据分布
"""
import json
from pathlib import Path

MODEL_DIR = Path("models/efficientnet_b3_anime_20260616_132028")
DATA_DIR = Path("data/final_dataset")

with open(MODEL_DIR / "training_results.json") as f:
    results = json.load(f)

model_class_names = results["class_names"]

print("=" * 70)
print("1️⃣  ImageFolder 从 data_dir 解析的 order（默认按字母排序）")
print("=" * 70)
# ImageFolder sorts directories alphabetically
data_dirs = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and any(d.iterdir())])
data_class_names = [d for d in data_dirs if d in set(model_class_names)]

for i, name in enumerate(data_class_names):
    print(f"  [{i:3d}] -> {name}")
print(f"\n  模型类别数: {len(model_class_names)}")
print(f"  数据目录有图片的类别数: {len(data_class_names)}")
print()

# ==============================
# 对比
# ==============================
print("=" * 70)
print("2️⃣  对比结果（仅对比模型定义的99类）")
print("=" * 70)
mismatch_positions = []
for i, (m, d) in enumerate(zip(model_class_names, model_class_names)):
    pass  # same list

# Check if model_class_names is the same as ImageFolder would produce
# ImageFolder produces: sorted(os.listdir(root)) filtered by is_dir
# Then sorted again implicitly by the alphabetical order
sorted_nonempty_model_dirs = sorted([d for d in model_class_names if (DATA_DIR / d).exists() and any((DATA_DIR / d).iterdir())])

if model_class_names[:len(sorted_nonempty_model_dirs)] == sorted_nonempty_model_dirs:
    print("  ✅ 模型 class_names 与 ImageFolder 的字母顺序一致")
else:
    print("  ❌ 顺序不一致！")
    for i, (m, d) in enumerate(zip(model_class_names, data_class_names)):
        if m != d:
            print(f"  位置 [{i}]: 模型={m}, 数据目录={d}")

# Carful check - does the model include classes that DON'T exist in data dir?
model_cls_set = set(model_class_names)
data_nonempty_set = set(sorted_nonempty_model_dirs)
missing_in_data = model_cls_set - data_nonempty_set
if missing_in_data:
    print(f"\n  ⚠️ 模型有但本地目录无图片的类别 ({len(missing_in_data)}个):")
    for c in sorted(missing_in_data):
        print(f"    {c}")

print()
print("=" * 70)
print("3️⃣  每个类别的样本数 & 训练时是否为验证集 Top-1 命中")
print("=" * 70)
total = 0
counts = {}
for cls in model_class_names:
    cls_dir = DATA_DIR / cls
    if cls_dir.exists():
        n = len(list(cls_dir.glob("*")))
        counts[cls] = n
        total += n
    else:
        counts[cls] = 0

# Print in a structured table
print(f"  {'类别':<20} {'样本数':>6} {'状态':<10}")
print(f"  {'-'*20} {'-'*6} {'-'*10}")
low_count = []
for cls in model_class_names:
    n = counts[cls]
    status = "✅" if n >= 100 else "⚠️" if n > 0 else "❌"
    if n < 100:
        low_count.append((cls, n))
    print(f"  {cls:<20} {n:>6} {status}")

print(f"\n  总样本数: {total}")
print(f"  平均每类: {total // len(model_class_names)}")
print(f"  样本数 < 100 的类别 ({len(low_count)}个):")
for cls, n in sorted(low_count, key=lambda x: x[1]):
    print(f"    {cls}: {n}")

print()
print("=" * 70)
print("4️⃣  训练配置回顾")
print("=" * 70)
print(f"  model_name:      {results.get('model_name', 'N/A')}")
print(f"  num_classes:     {results.get('num_classes', 'N/A')}")
print(f"  best_accuracy:   {results.get('best_accuracy', 'N/A')}")
print(f"  image_size:      {results.get('image_size', 'N/A')}")
print(f"  augment_level:   {results.get('augment_level', 'N/A')}")
print(f"  cutmix:          {results.get('cutmix', 'N/A')}")
print(f"  label_smoothing: {results.get('label_smoothing', 'N/A')}")
print(f"  batch_size:      {results.get('batch_size', 'N/A')}")
print(f"  learning_rate:   {results.get('learning_rate', 'N/A')}")
print(f"  epochs:          {results.get('epochs', 'N/A')}")

print()
print("=" * 70)
print("5️⃣  验证集 vs 测试集结果对比 — 是否标签映射错位？")
print("=" * 70)
# Load eval report if exists
eval_path = MODEL_DIR / "eval_report" / "evaluation_report.json"
if eval_path.exists():
    with open(eval_path) as f:
        eval_report = json.load(f)
    eval_per_class = eval_report.get("per_class", {})
    zero_shot = [cls for cls, m in eval_per_class.items() if m.get("accuracy", 1) == 0 and m.get("total", 0) >= 10]
    print(f"  测试集 0% 且样本≥10 的类别 ({len(zero_shot)}个):")
    for cls in zero_shot[:15]:
        print(f"    {cls}: {eval_per_class[cls]['total']}张, Acc=0%")
    if len(zero_shot) > 15:
        print(f"    ... 还有 {len(zero_shot)-15} 个")

    # Check if specific classes that model should know well are all 0
    well_known = ["eula", "ganyu", "fischl", "xinyan", "shenhe", "beidou", "amber"]
    print(f"\n  验证集 Acc=12.46%，但以下熟知角色测试集表现:")
    for cls in well_known:
        if cls in eval_per_class:
            m = eval_per_class[cls]
            print(f"    {cls}: {m['accuracy']*100:.2f}% ({m['correct']}/{m['total']})")
else:
    print("  eval_report.json 不存在，请先运行 eval_report.py")