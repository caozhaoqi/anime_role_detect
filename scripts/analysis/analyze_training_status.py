#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析数据集和训练状态
"""
from pathlib import Path
import json

# 检查数据集（支持多个可能的路径）
data_dirs = [
    Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'),
    Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/optimized_detection_v2/single_face'),
    Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/processed_dataset')
]

model_dir = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/models')

# 找到第一个存在的数据集目录
data_dir = None
total_images = 0
for d in data_dirs:
    if d.exists():
        data_dir = d
        break

print("=" * 80)
print("📊 数据集分析")
print("=" * 80)

if data_dir.exists():
    # 检查是否是split结构
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    if train_dir.exists() and val_dir.exists():
        # split结构：按类别子目录组织
        role_dirs = [d for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        total_images = 0
        role_stats = []
        
        for role_dir in role_dirs:
            train_count = len(list(role_dir.glob('*.jpg')))
            val_role_dir = val_dir / role_dir.name
            val_count = len(list(val_role_dir.glob('*.jpg'))) if val_role_dir.exists() else 0
            image_count = train_count + val_count
            if image_count > 0:
                total_images += image_count
                role_stats.append((role_dir.name, image_count))
        
        role_stats.sort(key=lambda x: x[1], reverse=True)
        
        print(f"数据集路径: {data_dir}")
        print(f"角色总数: {len(role_stats)}")
        print(f"图片总数: {total_images}")
        print(f"\n图片数量前10的角色:")
        for i, (role, count) in enumerate(role_stats[:10], 1):
            print(f"  {i}. {role}: {count}张")
        
        if len(role_stats) > 10:
            print(f"\n... 还有 {len(role_stats) - 10} 个角色")
        
        print(f"\n数据分布:")
        print(f"  - 平均每角色: {total_images / len(role_stats):.1f}张")
        print(f"  - 最少: {role_stats[-1][1]}张 ({role_stats[-1][0]})")
        print(f"  - 最多: {role_stats[0][1]}张 ({role_stats[0][0]})")
        
        # 统计训练/验证集
        train_total = sum(len(list(d.glob('*.jpg'))) for d in train_dir.iterdir() if d.is_dir())
        val_total = sum(len(list(d.glob('*.jpg'))) for d in val_dir.iterdir() if d.is_dir())
        print(f"\n数据集分割:")
        print(f"  - 训练集: {train_total}张")
        print(f"  - 验证集: {val_total}张")
    else:
        # 扁平结构：图片直接放在目录中
        all_images = list(data_dir.glob('*.jpg'))
        if all_images:
            # 尝试从文件名提取类别
            role_counts = {}
            for img_path in all_images:
                class_name = img_path.name.split('_')[0]
                if class_name not in role_counts:
                    role_counts[class_name] = 0
                role_counts[class_name] += 1
            
            role_stats = sorted(role_counts.items(), key=lambda x: x[1], reverse=True)
            
            print(f"数据集路径: {data_dir}")
            print(f"角色总数: {len(role_stats)}")
            print(f"图片总数: {len(all_images)}")
            print(f"\n图片数量前10的角色:")
            for i, (role, count) in enumerate(role_stats[:10], 1):
                print(f"  {i}. {role}: {count}张")
            
            if len(role_stats) > 10:
                print(f"\n... 还有 {len(role_stats) - 10} 个角色")
            
            print(f"\n数据分布:")
            print(f"  - 平均每角色: {len(all_images) / len(role_stats):.1f}张")
            print(f"  - 最少: {role_stats[-1][1]}张 ({role_stats[-1][0]})")
            print(f"  - 最多: {role_stats[0][1]}张 ({role_stats[0][0]})")
        else:
            print("❌ 数据集中没有找到图片")
else:
    print("❌ 数据集目录不存在")

print("\n" + "=" * 80)
print("🤖 模型训练状态")
print("=" * 80)

if model_dir.exists():
    model_subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
    
    if model_subdirs:
        print(f"找到 {len(model_subdirs)} 个模型目录:")
        for model_path in model_subdirs:
            print(f"\n📁 {model_path.name}")
            
            # 检查模型文件
            model_files = list(model_path.glob('*.pth'))
            if model_files:
                print(f"  ✅ 模型文件: {[f.name for f in model_files]}")
            else:
                print(f"  ❌ 未找到模型文件")
            
            # 检查训练结果
            result_file = model_path / 'training_results.json'
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"  📊 训练结果:")
                # 兼容新旧格式
                accuracy = results.get('best_accuracy', results.get('accuracy', 'N/A'))
                if accuracy != 'N/A' and isinstance(accuracy, float):
                    accuracy = f"{accuracy * 100:.2f}%"
                print(f"    - 准确率: {accuracy}")
                print(f"    - 训练样本: {results.get('train_samples', 'N/A')}")
                print(f"    - 验证样本: {results.get('val_samples', 'N/A')}")
                print(f"    - 类别数: {results.get('num_classes', 'N/A')}")
                print(f"    - 预训练: {results.get('use_pretrained', 'N/A')}")
                print(f"    - 训练时间: {results.get('timestamp', 'N/A')}")
            else:
                print(f"  ⚠️ 未找到训练结果文件")
    else:
        print("❌ 未找到任何训练好的模型")
else:
    print("❌ 模型目录不存在")

print("\n" + "=" * 80)
print("💡 训练建议")
print("=" * 80)

if data_dir.exists() and total_images > 0:
    print(f"✅ 数据集可用，共 {total_images} 张图片")
    print(f"✅ 支持 {len(role_stats)} 个角色分类")
    print(f"\n推荐训练配置:")
    print(f"  - 模型: efficientnet_b0 (平衡速度与精度)")
    print(f"  - Batch Size: 32")
    print(f"  - Epochs: 50")
    print(f"  - 学习率: 1e-4")
    print(f"\n训练命令:")
    print(f"  python3 scripts/training/train_all_models.py --model efficientnet_b0")
else:
    print("❌ 数据集不可用，请先准备数据")