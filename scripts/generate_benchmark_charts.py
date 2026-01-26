#!/usr/bin/env python3
"""
生成模型Benchmark测试结果的图表
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# 测试结果数据
model_names = [
    'character_classifier_best.pth',
    'character_classifier_best_improved.pth',
    'character_classifier_best_v2.pth',
    'character_classifier_final.pth',
    'character_classifier_final_improved.pth',
    'character_classifier_final_v2.pth'
]

# 简化模型名称
short_names = [
    'best',
    'best_improved',
    'best_v2',
    'final',
    'final_improved',
    'final_v2'
]

accuracy = [0.0927, 0.1793, 0.1302, 0.0927, 0.1805, 0.1372]
f1_score = [0.0855, 0.1679, 0.1175, 0.0855, 0.1658, 0.1167]
fps = [210.24, 576.44, 272.93, 242.72, 181.10, 207.61]

# 创建输出目录
output_dir = 'benchmark_results'
os.makedirs(output_dir, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 准确率和F1分数对比图
plt.figure(figsize=(12, 6))
x = np.arange(len(short_names))
width = 0.35

plt.bar(x - width/2, accuracy, width, label='准确率', color='#4CAF50')
plt.bar(x + width/2, f1_score, width, label='F1分数', color='#2196F3')

plt.xlabel('模型版本')
plt.ylabel('分数')
plt.title('不同模型版本的准确率和F1分数对比')
plt.xticks(x, short_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_f1_comparison.png'), dpi=300)
plt.close()

# 2. 推理速度对比图
plt.figure(figsize=(12, 6))
plt.bar(short_names, fps, color='#FF9800')
plt.xlabel('模型版本')
plt.ylabel('推理速度 (FPS)')
plt.title('不同模型版本的推理速度对比')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fps_comparison.png'), dpi=300)
plt.close()

# 3. 综合性能雷达图
plt.figure(figsize=(10, 10))

# 标准化数据
max_accuracy = max(accuracy)
max_f1 = max(f1_score)
max_fps = max(fps)

normalized_accuracy = [acc / max_accuracy for acc in accuracy]
normalized_f1 = [f1 / max_f1 for f1 in f1_score]
normalized_fps = [fps_val / max_fps for fps_val in fps]

# 雷达图数据
metrics = ['准确率', 'F1分数', '推理速度']
n_angles = len(metrics)
angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False).tolist()
angles += angles[:1]  # 闭合

ax = plt.subplot(111, polar=True)

for i, model in enumerate(short_names):
    values = [normalized_accuracy[i], normalized_f1[i], normalized_fps[i]]
    values += values[:1]  # 闭合
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
plt.title('不同模型版本的综合性能对比', size=15, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300)
plt.close()

# 4. 性能趋势图
plt.figure(figsize=(12, 6))

# 按模型类型分组
model_types = ['best', 'final']
improved_accuracy = [accuracy[1], accuracy[4]]
original_accuracy = [accuracy[0], accuracy[3]]
v2_accuracy = [accuracy[2], accuracy[5]]

improved_f1 = [f1_score[1], f1_score[4]]
original_f1 = [f1_score[0], f1_score[3]]
v2_f1 = [f1_score[2], f1_score[5]]

x = np.arange(len(model_types))
width = 0.25

plt.bar(x - width, original_accuracy, width, label='原始版本', color='#9E9E9E')
plt.bar(x, improved_accuracy, width, label='改进版本', color='#4CAF50')
plt.bar(x + width, v2_accuracy, width, label='v2版本', color='#2196F3')

plt.xlabel('模型类型')
plt.ylabel('准确率')
plt.title('不同模型类型和版本的准确率对比')
plt.xticks(x, model_types)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'version_trend.png'), dpi=300)
plt.close()

print(f"图表生成完成！保存在 {output_dir} 目录中")
print("生成的图表：")
print("1. accuracy_f1_comparison.png - 准确率和F1分数对比图")
print("2. fps_comparison.png - 推理速度对比图")
print("3. radar_comparison.png - 综合性能雷达图")
print("4. version_trend.png - 性能趋势图")
