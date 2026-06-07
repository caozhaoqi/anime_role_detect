#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate benchmark report with charts
"""

import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import numpy as np
import os
from collections import Counter

# Set font for Chinese characters
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Load benchmark data
with open("models/efficientnet_b3_loli_optimized_v2_20260529_133654/benchmark_results.json") as f:
    benchmark = json.load(f)

with open("models/efficientnet_b3_loli_optimized_v2_20260529_133654/training_results.json") as f:
    training = json.load(f)

# Create reports directory
os.makedirs("reports", exist_ok=True)

# Chart 1: Accuracy Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ["Top-1", "Top-3", "Top-5"]
values = [
    benchmark["top_1_accuracy"] * 100,
    benchmark["top_3_accuracy"] * 100,
    benchmark["top_5_accuracy"] * 100,
]
colors = ["#4CAF50", "#2196F3", "#FF9800"]
bars = ax.bar(metrics, values, color=colors, edgecolor="black", linewidth=1.5)

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{val:.2f}%",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Model Accuracy Comparison", fontsize=16, fontweight="bold")
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("reports/accuracy_chart.png", dpi=150, bbox_inches="tight")
plt.close()

# Chart 2: Performance Radar Chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
categories = ["Top-1\nAccuracy", "Top-3\nAccuracy", "Top-5\nAccuracy", "Speed\n(FPS/100)"]
values_radar = [
    benchmark["top_1_accuracy"],
    benchmark["top_3_accuracy"],
    benchmark["top_5_accuracy"],
    min(benchmark["fps"] / 100, 1.0),
]
values_radar += values_radar[:1]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

ax.fill(angles, values_radar, color="#2196F3", alpha=0.25)
ax.plot(angles, values_radar, color="#2196F3", linewidth=2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title("Model Performance Radar", fontsize=16, fontweight="bold", y=1.08)
plt.tight_layout()
plt.savefig("reports/radar_chart.png", dpi=150, bbox_inches="tight")
plt.close()

# Chart 3: Role Distribution Pie Chart
with open("models/efficientnet_b3_loli_optimized_v2_20260529_133654_mapping.json") as f:
    mapping = json.load(f)

role_counts = Counter(mapping)
top_roles = role_counts.most_common(10)
labels = [r[0] for r in top_roles]
sizes = [r[1] for r in top_roles]

fig, ax = plt.subplots(figsize=(10, 8))
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    colors=colors_pie,
    startangle=90,
    textprops={"fontsize": 10},
)
ax.set_title("Top 10 Roles by Sample Count", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/role_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

print("Charts generated successfully!")
print("  - reports/accuracy_chart.png")
print("  - reports/radar_chart.png")
print("  - reports/role_distribution.png")
