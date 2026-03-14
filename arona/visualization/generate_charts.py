#!/usr/bin/env python3
"""
生成模型性能图表
用于博客展示基准测试结果
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 模型数据
models = ['MobileNetV2', 'EfficientNet-B0', 'EfficientNet-B3', 'ResNet50']
inference_speed = [12.53, 12.87, 15.86, 18.24]  # ms/图像
model_size = [14.9, 29.2, 56.5, 295.2]  # MB
accuracy = [91.44, 93.16, 93.92, 90.68]  # %
memory_usage = ['低', '中低', '中', '高']

# 创建输出目录
output_dir = 'docs/charts'
os.makedirs(output_dir, exist_ok=True)

# 图表1：推理速度对比
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, inference_speed, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax.set_ylabel('推理速度 (ms/图像)', fontsize=12)
ax.set_title('模型推理速度对比', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 在柱状图上显示数值
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'inference_speed.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"已生成: {os.path.join(output_dir, 'inference_speed.png')}")

# 图表2：模型大小对比
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, model_size, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax.set_ylabel('模型大小 (MB)', fontsize=12)
ax.set_title('模型大小对比', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 在柱状图上显示数值
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_size.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"已生成: {os.path.join(output_dir, 'model_size.png')}")

# 图表3：准确率对比
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, accuracy, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax.set_ylabel('验证准确率 (%)', fontsize=12)
ax.set_ylim(85, 95)
ax.set_title('模型准确率对比', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 在柱状图上显示数值
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"已生成: {os.path.join(output_dir, 'accuracy.png')}")

# 图表4：综合性能雷达图
categories = ['速度', '大小', '准确率']
n_categories = len(categories)

# 将性能转换为评分（5分制）
speed_scores = [5 - (x - min(inference_speed)) / (max(inference_speed) - min(inference_speed)) * 4 for x in inference_speed]
size_scores = [5 - (x - min(model_size)) / (max(model_size) - min(model_size)) * 4 for x in model_size]
acc_scores = [1 + (x - min(accuracy)) / (max(accuracy) - min(accuracy)) * 4 for x in accuracy]

# 为每个模型创建雷达图
fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(projection='polar'))
axes = axes.flatten()

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
angles += angles[:1]

for i, model in enumerate(models):
    ax = axes[i]
    values = [speed_scores[i], size_scores[i], acc_scores[i]]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=8)
    ax.set_title(model, fontsize=12, fontweight='bold', pad=20)
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"已生成: {os.path.join(output_dir, 'radar_chart.png')}")

# 图表5：综合性能对比（多指标）
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# 推理速度
bars1 = ax1.bar(models, inference_speed, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax1.set_ylabel('推理速度 (ms/图像)', fontsize=10)
ax1.set_title('推理速度', fontsize=11, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# 模型大小
bars2 = ax2.bar(models, model_size, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax2.set_ylabel('模型大小 (MB)', fontsize=10)
ax2.set_title('模型大小', fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# 准确率
bars3 = ax3.bar(models, accuracy, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax3.set_ylabel('验证准确率 (%)', fontsize=10)
ax3.set_ylim(85, 95)
ax3.set_title('准确率', fontsize=11, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'combined_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"已生成: {os.path.join(output_dir, 'combined_comparison.png')}")

# 图表6：性能权衡分析
fig, ax = plt.subplots(figsize=(10, 8))

# 归一化数据
normalized_speed = [1 - (x - min(inference_speed)) / (max(inference_speed) - min(inference_speed)) for x in inference_speed]
normalized_size = [1 - (x - min(model_size)) / (max(model_size) - min(model_size)) for x in model_size]
normalized_acc = [(x - min(accuracy)) / (max(accuracy) - min(accuracy)) for x in accuracy]

# 绘制散点图
scatter = ax.scatter(normalized_speed, normalized_acc, 
                 s=[x * 100 for x in normalized_size],  # 模型大小作为点的大小
                 c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
                 alpha=0.7, edgecolors='black', linewidth=1.5)

# 添加标签
for i, model in enumerate(models):
    ax.annotate(model, 
                (normalized_speed[i], normalized_acc[i]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, fontweight='bold')

ax.set_xlabel('速度性能 (归一化)', fontsize=12)
ax.set_ylabel('准确率 (归一化)', fontsize=12)
ax.set_title('性能权衡分析 (点大小表示模型大小)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)

# 添加图例
legend_elements = [
    plt.scatter([], [], s=100, c='gray', alpha=0.5, edgecolors='black', label='小模型'),
    plt.scatter([], [], s=300, c='gray', alpha=0.5, edgecolors='black', label='大模型')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'performance_tradeoff.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"已生成: {os.path.join(output_dir, 'performance_tradeoff.png')}")

print("\n所有图表生成完成！")
print(f"图表保存在: {output_dir}")
