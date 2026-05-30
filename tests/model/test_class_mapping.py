#!/usr/bin/env python3
"""
测试类别映射修复
"""
import json
import os

# 加载类别映射
mapping_path = os.path.join("../models", "character_classifier_best_improved_class_mapping.json")
idx_to_class = None

if os.path.exists(mapping_path):
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
        idx_to_class = mapping["idx_to_class"]
    print(f"✅ 类别映射加载成功，包含 {len(idx_to_class)} 个类别")
else:
    print("❌ 类别映射文件不存在")
    exit(1)

# 测试不同索引
print("\n=== 测试类别映射 ===")
test_indices = [127, 0, 73, 130]

for idx in test_indices:
    # 测试修复前的方法（整数查找）
    old_method = None
    try:
        old_method = idx_to_class[idx]
        print(f"索引 {idx}（整数查找）: {old_method}")
    except KeyError:
        print(f"索引 {idx}（整数查找）: KeyError")

    # 测试修复后的方法（字符串查找）
    new_method = None
    try:
        new_method = idx_to_class[str(idx)]
        print(f"索引 {idx}（字符串查找）: {new_method}")
    except KeyError:
        print(f"索引 {idx}（字符串查找）: KeyError")

# 测试修复后的完整逻辑
print("\n=== 测试修复后的完整逻辑 ===")


def get_role_name(predicted_idx, idx_to_class):
    if idx_to_class:
        # 尝试将predicted_idx转换为字符串查找
        if str(predicted_idx) in idx_to_class:
            return idx_to_class[str(predicted_idx)]
        else:
            return f"类别_{predicted_idx}"
    else:
        return f"类别_{predicted_idx}"


for idx in test_indices:
    role = get_role_name(idx, idx_to_class)
    print(f"索引 {idx}: {role}")

print("\n🎉 测试完成！")
