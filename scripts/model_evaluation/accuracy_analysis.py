#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型准确率不稳定原因分析脚本
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from collections import defaultdict


def analyze_training_results(training_results_path):
    """分析训练结果"""
    with open(training_results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    print("=" * 70)
    print("📊 训练结果分析")
    print("=" * 70)

    # 提取训练和验证准确率
    train_accs = []
    val_accs = []
    for record in results["train_history"]:
        if record["phase"] == "train":
            train_accs.append(record["accuracy"])
        else:
            val_accs.append(record["accuracy"])

    # 统计信息
    best_train_acc = max(train_accs)
    best_val_acc = max(val_accs)
    final_train_acc = train_accs[-1]
    final_val_acc = val_accs[-1]

    print(f"📈 训练集样本数: {results['train_samples']}")
    print(f"📉 验证集样本数: {results['val_samples']}")
    print(f"🏆 最佳训练准确率: {best_train_acc * 100:.2f}%")
    print(f"🏆 最佳验证准确率: {best_val_acc * 100:.2f}%")
    print(f"🔚 最终训练准确率: {final_train_acc * 100:.2f}%")
    print(f"🔚 最终验证准确率: {final_val_acc * 100:.2f}%")
    print(f"📊 过拟合程度 (训练-验证差距): {(final_train_acc - final_val_acc) * 100:.2f}%")

    # 分析训练趋势
    print("\n📈 训练趋势分析:")
    if final_train_acc > 0.8 and final_val_acc < 0.6:
        print("   ⚠️ 严重过拟合: 训练准确率很高但验证准确率较低")
    elif final_train_acc - final_val_acc > 0.2:
        print("   ⚠️ 中度过拟合: 训练与验证差距较大")
    else:
        print("   ✅ 拟合正常: 训练与验证差距合理")

    return {
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_train": best_train_acc,
        "best_val": best_val_acc,
        "final_train": final_train_acc,
        "final_val": final_val_acc,
    }


def analyze_dataset_balance(data_dir, model_classes):
    """分析数据集类别平衡情况"""
    print("\n" + "=" * 70)
    print("📊 数据集类别平衡分析")
    print("=" * 70)

    class_counts = []
    missing_classes = []
    extra_classes = []

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        count = len([f for f in os.listdir(class_dir) if f.endswith(".jpg")])
        class_counts.append((class_name, count))

        if class_name not in model_classes:
            extra_classes.append(class_name)
        if class_name in model_classes and count < 10:
            print(f"   ⚠️ 样本稀少: {class_name} - {count} 张图片")

    # 检查缺失类别
    for model_class in model_classes:
        class_dir = os.path.join(data_dir, model_class)
        if not os.path.exists(class_dir):
            missing_classes.append(model_class)

    # 排序并输出统计
    class_counts.sort(key=lambda x: x[1])

    print(f"\n📁 数据集中类别总数: {len(class_counts)}")
    print(f"🧠 模型支持类别数: {len(model_classes)}")
    print(f"❓ 额外类别(模型不支持): {len(extra_classes)}")
    print(f"❌ 缺失类别(数据集中没有): {len(missing_classes)}")

    print("\n📊 样本数量分布:")
    counts = [c[1] for c in class_counts]
    print(f"   最小样本数: {min(counts)}")
    print(f"   最大样本数: {max(counts)}")
    print(f"   平均样本数: {sum(counts)/len(counts):.2f}")
    print(f"   样本总数: {sum(counts)}")

    # 分析类别不平衡
    if max(counts) / min(counts) > 10:
        print("   ⚠️ 严重类别不平衡: 最大类样本数是最小类的10倍以上")
    elif max(counts) / min(counts) > 5:
        print("   ⚠️ 中度类别不平衡: 最大类样本数是最小类的5倍以上")
    else:
        print("   ✅ 类别平衡较好")

    return {
        "class_counts": class_counts,
        "missing_classes": missing_classes,
        "extra_classes": extra_classes,
    }


def analyze_per_class_accuracy(benchmark_results):
    """分析各类别的准确率"""
    print("\n" + "=" * 70)
    print("📊 各类别准确率分析")
    print("=" * 70)

    per_class = benchmark_results.get("per_class_stats", {})

    # 计算各类别准确率
    class_accs = []
    for class_name, stats in per_class.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        class_accs.append((class_name, acc, stats["correct"], stats["total"]))

    # 排序
    class_accs.sort(key=lambda x: x[1])

    print("\n📉 准确率最低的10个类别:")
    for name, acc, correct, total in class_accs[:10]:
        print(f"   {name}: {correct}/{total} ({acc*100:.1f}%)")

    print("\n📈 准确率最高的10个类别:")
    for name, acc, correct, total in reversed(class_accs[-10:]):
        print(f"   {name}: {correct}/{total} ({acc*100:.1f}%)")

    # 计算整体统计
    all_accs = [a[1] for a in class_accs]
    print(f"\n📊 类别准确率分布:")
    print(f"   最低准确率: {min(all_accs)*100:.1f}%")
    print(f"   最高准确率: {max(all_accs)*100:.1f}%")
    print(f"   平均准确率: {sum(all_accs)/len(all_accs)*100:.1f}%")
    print(f"   标准差: {np.std(all_accs)*100:.1f}%")

    # 分析波动原因
    if np.std(all_accs) > 0.3:
        print("   ⚠️ 类别间准确率差异很大，可能存在类别不平衡或特征差异")
    else:
        print("   ✅ 类别间准确率分布较均匀")

    return class_accs


def run_consistency_test(model_path, data_dir, model_config, num_runs=5):
    """测试模型推理的一致性"""
    print("\n" + "=" * 70)
    print("🧪 模型推理一致性测试")
    print("=" * 70)

    # 加载模型
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # 图像预处理
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 随机选择测试图片
    test_images = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = [f for f in os.listdir(class_dir) if f.endswith(".jpg")]
        if files:
            test_images.append(os.path.join(class_dir, files[0]))
            if len(test_images) >= 10:
                break

    # 多次推理测试
    predictions = []
    with torch.no_grad():
        for run in range(num_runs):
            preds = []
            for img_path in test_images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = transform(img).unsqueeze(0).to(device)
                    output = model(img)
                    pred = torch.argmax(output, dim=1).item()
                    preds.append(pred)
                except:
                    preds.append(-1)
            predictions.append(preds)

    # 分析一致性
    consistent_count = 0
    for i in range(len(test_images)):
        preds_for_img = [p[i] for p in predictions]
        if len(set(preds_for_img)) == 1:
            consistent_count += 1

    consistency_rate = consistent_count / len(test_images)
    print(f"🧪 测试图片数: {len(test_images)}")
    print(f"🔄 重复推理次数: {num_runs}")
    print(f"✅ 预测一致率: {consistency_rate * 100:.1f}%")

    if consistency_rate == 1.0:
        print("   ✅ 模型推理完全一致")
    elif consistency_rate > 0.8:
        print("   ⚠️ 模型推理基本一致，但存在少量波动")
    else:
        print("   ❌ 模型推理存在明显波动")

    return consistency_rate


def main():
    print("🎯 模型准确率不稳定原因分析")
    print("=" * 70)

    # 路径设置
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(
        project_root, "models", "efficientnet_b3_loli_76_pretrained_20260520_162729"
    )
    data_dir = os.path.join(project_root, "data", "expanded_dataset")

    training_results_path = os.path.join(model_dir, "training_results.json")
    benchmark_results_path = os.path.join(model_dir, "benchmark_results_new.json")
    model_path = os.path.join(model_dir, "model_full.pth")

    # 1. 分析训练结果
    training_analysis = analyze_training_results(training_results_path)

    # 2. 分析数据集平衡
    with open(training_results_path, "r", encoding="utf-8") as f:
        training_results = json.load(f)
    dataset_analysis = analyze_dataset_balance(data_dir, training_results["class_names"])

    # 3. 分析各类别准确率
    with open(benchmark_results_path, "r", encoding="utf-8") as f:
        benchmark_results = json.load(f)
    class_analysis = analyze_per_class_accuracy(benchmark_results)

    # 4. 测试模型推理一致性
    consistency_rate = run_consistency_test(model_path, data_dir, training_results)

    # 5. 生成综合分析报告
    print("\n" + "=" * 70)
    print("📋 综合分析报告")
    print("=" * 70)

    issues = []
    suggestions = []

    # 过拟合问题
    if training_analysis["final_train"] - training_analysis["final_val"] > 0.2:
        issues.append("过拟合问题")
        suggestions.append("1. 增加训练数据量")
        suggestions.append("2. 添加正则化（Dropout、权重衰减）")
        suggestions.append("3. 使用数据增强")
        suggestions.append("4. 考虑早停策略")

    # 类别不平衡问题
    counts = [c[1] for c in dataset_analysis["class_counts"]]
    if max(counts) / min(counts) > 5:
        issues.append("类别不平衡")
        suggestions.append("5. 对样本少的类别进行数据扩充")
        suggestions.append("6. 使用加权损失函数")
        suggestions.append("7. 考虑过采样/欠采样")

    # 数据集不一致问题
    if dataset_analysis["missing_classes"] or dataset_analysis["extra_classes"]:
        issues.append("数据集与模型类别不一致")
        suggestions.append("8. 对齐训练集和测试集的类别")

    # 推理一致性问题
    if consistency_rate < 0.9:
        issues.append("模型推理一致性问题")
        suggestions.append("9. 检查模型是否有随机操作")
        suggestions.append("10. 确认模型处于eval模式")

    print("\n🔍 主要问题:")
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("   未发现明显问题")

    print("\n💡 优化建议:")
    for suggestion in suggestions:
        print(f"   {suggestion}")

    print("\n📊 总结:")
    print(f"   模型整体准确率: {benchmark_results['top_1_accuracy']*100:.1f}%")
    print(f"   训练集准确率: {training_analysis['final_train']*100:.1f}%")
    print(f"   验证集准确率: {training_analysis['final_val']*100:.1f}%")
    print(f"   主要挑战: {', '.join(issues) if issues else '无'}")


if __name__ == "__main__":
    main()
