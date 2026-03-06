#!/usr/bin/env python3
"""
模型基准测试脚本
对比 CLIP+FAISS (索引模式) 和 EfficientNet (模型模式) 的准确率和性能
自动采集测试数据并生成对比报告
"""
import os
import sys
import time
import json
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.general_classification import GeneralClassification
from tests.collect_test_data import collect_single_character_data

def prepare_benchmark_data(target_roles, num_images_per_role=5, output_dir="tests/benchmark_data"):
    """
    准备基准测试数据
    :param target_roles: 目标角色列表
    :param num_images_per_role: 每个角色的测试图片数量
    :param output_dir: 数据保存目录
    """
    print(f"=== 准备基准测试数据 ===")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    dataset_info = []
    
    for role in target_roles:
        role_dir = os.path.join(output_dir, role)
        if not os.path.exists(role_dir):
            os.makedirs(role_dir)
            
        # 检查现有图片数量
        existing_images = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if len(existing_images) < num_images_per_role:
            needed = num_images_per_role - len(existing_images)
            print(f"正在为角色 '{role}' 采集 {needed} 张图片...")
            # 调用采集脚本
            collect_single_character_data(role, needed, role_dir)
            
        # 重新获取图片列表
        images = [os.path.join(role_dir, f) for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        # 限制数量
        images = images[:num_images_per_role]
        
        for img_path in images:
            dataset_info.append({
                "image_path": img_path,
                "true_label": role
            })
            
    print(f"数据准备完成，共 {len(dataset_info)} 张测试图片")
    return dataset_info

def run_benchmark(dataset, classifier):
    """
    运行基准测试
    """
    results = {
        "clip": {"pred": [], "true": [], "time": [], "conf": []},
        "efficientnet": {"pred": [], "true": [], "time": [], "conf": []}
    }
    
    print("\n=== 开始基准测试 ===")
    print(f"测试样本数: {len(dataset)}")
    
    # 预热模型
    if len(dataset) > 0:
        dummy_img = dataset[0]["image_path"]
        classifier.classify_image(dummy_img, use_model=False)
        classifier.classify_image(dummy_img, use_model=True)
    
    for item in tqdm(dataset, desc="Testing"):
        img_path = item["image_path"]
        true_label = item["true_label"]
        
        # 1. 测试 CLIP + FAISS
        start_time = time.time()
        try:
            role, conf, _ = classifier.classify_image(img_path, use_model=False)
            elapsed = time.time() - start_time
            
            results["clip"]["pred"].append(role)
            results["clip"]["true"].append(true_label)
            results["clip"]["time"].append(elapsed)
            results["clip"]["conf"].append(conf)
        except Exception as e:
            print(f"CLIP Error on {img_path}: {e}")
            
        # 2. 测试 EfficientNet
        start_time = time.time()
        try:
            role, conf, _ = classifier.classify_image(img_path, use_model=True)
            elapsed = time.time() - start_time
            
            results["efficientnet"]["pred"].append(role)
            results["efficientnet"]["true"].append(true_label)
            results["efficientnet"]["time"].append(elapsed)
            results["efficientnet"]["conf"].append(conf)
        except Exception as e:
            print(f"EfficientNet Error on {img_path}: {e}")
            
    return results

def generate_report(results, output_dir="reports"):
    """生成测试报告"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    report_path = os.path.join(output_dir, "benchmark_report.md")
    
    # 计算指标
    metrics = {}
    for model in ["clip", "efficientnet"]:
        y_true = results[model]["true"]
        y_pred = results[model]["pred"]
        times = results[model]["time"]
        
        acc = accuracy_score(y_true, y_pred)
        avg_time = np.mean(times) * 1000 # ms
        fps = 1.0 / np.mean(times)
        
        metrics[model] = {
            "accuracy": acc,
            "avg_time": avg_time,
            "fps": fps
        }
        
    # 写入报告
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 模型基准测试报告\n\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. 核心指标对比\n\n")
        f.write("| 模型 | 准确率 (Accuracy) | 平均耗时 (ms) | FPS |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **CLIP + FAISS** | {metrics['clip']['accuracy']:.2%} | {metrics['clip']['avg_time']:.2f} | {metrics['clip']['fps']:.2f} |\n")
        f.write(f"| **EfficientNet** | {metrics['efficientnet']['accuracy']:.2%} | {metrics['efficientnet']['avg_time']:.2f} | {metrics['efficientnet']['fps']:.2f} |\n\n")
        
        f.write("## 2. 详细分类报告\n\n")
        for model in ["clip", "efficientnet"]:
            model_name = "CLIP (通用索引)" if model == "clip" else "EfficientNet (专用模型)"
            f.write(f"### {model_name}\n")
            f.write("```\n")
            f.write(classification_report(results[model]["true"], results[model]["pred"], zero_division=0))
            f.write("```\n\n")
            
    print(f"报告已生成: {report_path}")
    
    # 绘制图表
    plot_comparison(metrics, output_dir)

def plot_comparison(metrics, output_dir):
    """绘制对比图表"""
    models = ["CLIP", "EfficientNet"]
    accs = [metrics["clip"]["accuracy"], metrics["efficientnet"]["accuracy"]]
    times = [metrics["clip"]["avg_time"], metrics["efficientnet"]["avg_time"]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 准确率对比
    ax1.bar(models, accs, color=['#3498db', '#2ecc71'])
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.1)
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.02, f"{v:.2%}", ha='center')
        
    # 耗时对比
    ax2.bar(models, times, color=['#e74c3c', '#f1c40f'])
    ax2.set_title('Inference Time (ms)')
    ax2.set_ylabel('Time (ms)')
    for i, v in enumerate(times):
        ax2.text(i, v + 5, f"{v:.1f}", ha='center')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_chart.png"))
    print(f"图表已生成: {os.path.join(output_dir, 'benchmark_chart.png')}")

def main():
    parser = argparse.ArgumentParser(description="模型基准测试")
    parser.add_argument("--roles", nargs="+", default=["原神_胡桃", "原神_雷泽", "Re0_雷姆", "初音未来"], help="测试角色列表")
    parser.add_argument("--num", type=int, default=5, help="每个角色的测试图片数")
    args = parser.parse_args()
    
    # 1. 初始化系统
    print("正在初始化分类系统...")
    classifier = GeneralClassification()
    classifier.initialize()
    
    # 2. 准备数据
    dataset = prepare_benchmark_data(args.roles, args.num)
    
    if not dataset:
        print("没有测试数据，退出")
        return
        
    # 3. 运行测试
    results = run_benchmark(dataset, classifier)
    
    # 4. 生成报告
    generate_report(results)

if __name__ == "__main__":
    main()
