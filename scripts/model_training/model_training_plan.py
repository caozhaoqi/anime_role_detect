#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据数据采集进度制定模型训练计划
"""
import os


def analyze_dataset():
    """分析数据集状态"""
    ROLE_LIST_FILE = (
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt"
    )
    DATASET_PATH = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset"

    # 读取角色列表
    roles = []
    with open(ROLE_LIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                roles.append({"cn": parts[0], "en": parts[2], "source": parts[1]})

    # 统计数据
    stats = []
    total_images = 0
    completed_count = 0  # >= 100
    low_count_roles = []  # < 100
    zero_count_roles = []  # == 0

    for role in roles:
        role_dir = os.path.join(DATASET_PATH, role["en"])
        if os.path.isdir(role_dir):
            count = len([f for f in os.listdir(role_dir) if f.endswith(".jpg")])
        else:
            count = 0

        stats.append({"cn": role["cn"], "en": role["en"], "source": role["source"], "count": count})
        total_images += count
        if count >= 100:
            completed_count += 1
        elif count == 0:
            zero_count_roles.append(role["cn"])
        else:
            low_count_roles.append((role["cn"], count))

    # 按数量排序
    stats.sort(key=lambda x: x["count"], reverse=True)
    low_count_roles.sort(key=lambda x: x[1])

    return {
        "total_roles": len(roles),
        "completed_count": completed_count,
        "low_count_roles": low_count_roles,
        "zero_count_roles": zero_count_roles,
        "total_images": total_images,
        "avg_images_per_role": total_images // len(roles) if roles else 0,
        "top_roles": stats[:10],
        "stats": stats,
    }


def generate_training_plan(dataset_info):
    """生成训练计划"""
    plan = []

    # 阶段1: 数据采集完成度评估
    plan.append("=" * 80)
    plan.append("📊 数据集状态评估")
    plan.append("=" * 80)
    plan.append(f"总角色数: {dataset_info['total_roles']} 个")
    plan.append(
        f"已满足100张: {dataset_info['completed_count']} 个 ({dataset_info['completed_count']/dataset_info['total_roles']*100:.1f}%)"
    )
    plan.append(f"未满足100张: {dataset_info['total_roles'] - dataset_info['completed_count']} 个")
    plan.append(f"总图片数: {dataset_info['total_images']:,} 张")
    plan.append(f"平均每角色: {dataset_info['avg_images_per_role']} 张")

    plan.append("\n📋 图片数量TOP10角色:")
    for r in dataset_info["top_roles"]:
        plan.append(f"  {r['cn']} ({r['source']}): {r['count']} 张")

    if dataset_info["zero_count_roles"]:
        plan.append("\n⚠️ 图片数量为0的角色:")
        for cn in dataset_info["zero_count_roles"]:
            plan.append(f"  • {cn}")

    if dataset_info["low_count_roles"]:
        plan.append("\n📉 图片不足100张的角色:")
        for cn, count in dataset_info["low_count_roles"]:
            plan.append(f"  • {cn}: {count} 张 (还差 {100-count} 张)")

    # 阶段2: 训练计划建议
    plan.append("\n" + "=" * 80)
    plan.append("🎯 模型训练计划")
    plan.append("=" * 80)

    # 评估当前数据是否适合训练
    completion_rate = dataset_info["completed_count"] / dataset_info["total_roles"]

    if completion_rate >= 0.9:
        plan.append("✅ 当前数据状态: 优秀")
        plan.append("   - 可以立即开始训练")
        plan.append("   - 建议使用完整数据集")
    elif completion_rate >= 0.7:
        plan.append("⚠️ 当前数据状态: 良好")
        plan.append("   - 可以开始预训练")
        plan.append("   - 同时继续补充剩余角色")
    elif completion_rate >= 0.5:
        plan.append("⚡ 当前数据状态: 中等")
        plan.append("   - 建议先完成主要角色的数据采集")
        plan.append("   - 可以开始小规模测试训练")
    else:
        plan.append("❌ 当前数据状态: 不足")
        plan.append("   - 建议继续数据采集")
        plan.append("   - 等待数据量达标后再开始训练")

    # 训练阶段划分
    plan.append("\n📅 训练阶段划分:")
    plan.append("┌─────────────────────────────────────────────────────────────┐")
    plan.append("│ 阶段1: 数据准备 (进行中)                                   │")
    plan.append("│   - 完成剩余14个角色的数据采集                              │")
    plan.append("│   - 数据清洗和标准化                                        │")
    plan.append("│   - 目标: 所有角色≥100张图片                               │")
    plan.append("├─────────────────────────────────────────────────────────────┤")
    plan.append("│ 阶段2: 模型预训练                                          │")
    plan.append("│   - 使用现有64个完整角色进行预训练                          │")
    plan.append("│   - 模型: ResNet50 / EfficientNet                          │")
    plan.append("│   - 目标: 达到初步分类能力                                  │")
    plan.append("├─────────────────────────────────────────────────────────────┤")
    plan.append("│ 阶段3: 全量训练                                            │")
    plan.append("│   - 使用完整78个角色数据集                                  │")
    plan.append("│   - 超参数调优                                             │")
    plan.append("│   - 目标: 最终模型训练                                      │")
    plan.append("├─────────────────────────────────────────────────────────────┤")
    plan.append("│ 阶段4: 模型评估与优化                                      │")
    plan.append("│   - 准确率评估                                              │")
    plan.append("│   - 混淆矩阵分析                                            │")
    plan.append("│   - 数据增强和模型优化                                      │")
    plan.append("└─────────────────────────────────────────────────────────────┘")

    # 建议的训练配置
    plan.append("\n⚙️ 建议训练配置:")
    plan.append(f"  • 训练集/验证集/测试集: 70%/20%/10%")
    plan.append(f"  • 批次大小: 32-64")
    plan.append(f"  • 学习率: 1e-4 ~ 1e-3")
    plan.append(f"  • 训练轮数: 50-100")
    plan.append(f"  • 数据增强: 随机裁剪、翻转、旋转")

    # 下一步行动建议
    plan.append("\n🚀 下一步行动:")
    if dataset_info["zero_count_roles"]:
        plan.append(f"  1. 优先完成 {len(dataset_info['zero_count_roles'])} 个零数据角色的采集")
    if dataset_info["low_count_roles"]:
        plan.append(f"  2. 补充 {len(dataset_info['low_count_roles'])} 个数据不足角色")
    plan.append("  3. 启动预训练（可选，使用现有64个角色）")
    plan.append("  4. 准备训练环境和代码框架")

    return "\n".join(plan)


if __name__ == "__main__":
    dataset_info = analyze_dataset()
    plan = generate_training_plan(dataset_info)
    print(plan)

    # 保存计划到文件
    output_file = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/training_plan.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(plan)
    print(f"\n📄 训练计划已保存到: {output_file}")
