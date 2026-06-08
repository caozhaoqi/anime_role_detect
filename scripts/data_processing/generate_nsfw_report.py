#!/usr/bin/env python3
"""NSFW检测分析报告生成"""

import json
from pathlib import Path

# 设置路径
RULES_RESULTS = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/nsfw_results/nsfw_detection_results.json')
RULES_SUMMARY = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/nsfw_results/nsfw_summary.txt')
OUTPUT_REPORT = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/nsfw_results/comprehensive_report.md')


def generate_report():
    """生成综合NSFW检测报告"""
    
    # 读取规则检测结果
    with open(RULES_RESULTS, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 读取规则检测摘要
    with open(RULES_SUMMARY, 'r', encoding='utf-8') as f:
        summary = f.read()
    
    # 统计分析
    total_images = len(results)
    nsfw_count = sum(1 for r in results if r['label'] == 'NSFW')
    suggestive_count = sum(1 for r in results if r['label'] == 'Suggestive')
    safe_count = sum(1 for r in results if r['label'] == 'Safe')
    
    # 按角色统计
    char_stats = {}
    for result in results:
        char = result['character']
        if char not in char_stats:
            char_stats[char] = {'total': 0, 'nsfw': 0, 'suggestive': 0, 'safe': 0, 'avg_score': 0, 'scores': []}
        char_stats[char]['total'] += 1
        char_stats[char]['scores'].append(result['score'])
        if result['label'] == 'NSFW':
            char_stats[char]['nsfw'] += 1
        elif result['label'] == 'Suggestive':
            char_stats[char]['suggestive'] += 1
        else:
            char_stats[char]['safe'] += 1
    
    # 计算平均分数
    for char in char_stats:
        char_stats[char]['avg_score'] = sum(char_stats[char]['scores']) / len(char_stats[char]['scores'])
    
    # 生成报告
    report = f"""# NSFW检测综合分析报告

## 📊 检测方法对比

### 基于规则的检测（已执行）
- **检测方法**: 皮肤颜色比例 + 边缘密度 + 红色区域检测
- **模型**: OpenCV + 自定义规则算法
- **状态**: ✅ 已完成
- **准确度**: 适合动漫图片，但可能存在误报

### 深度学习模型检测（待执行）
- **检测方法**: TensorFlow MobileNet V2
- **模型**: `mobilenet_v2_140_224` (来自 archived/nsfw_model_img)
- **状态**: ❌ TensorFlow未安装
- **建议**: 安装 `pip install tensorflow` 后执行

---

## 📈 整体检测结果（基于规则）

| 类别 | 数量 | 占比 |
|------|------|------|
| **NSFW** | {nsfw_count} | {nsfw_count/total_images*100:.1f}% |
| **Suggestive** | {suggestive_count} | {suggestive_count/total_images*100:.1f}% |
| **Safe** | {safe_count} | {safe_count/total_images*100:.1f}% |
| **总计** | {total_images} | 100% |

---

## 🎭 各角色NSFW分布

| 角色 | 总数 | NSFW | Suggestive | Safe | NSFW率 | 平均分数 |
|------|------|------|------------|------|--------|----------|
"""
    
    # 按NSFW率排序
    sorted_chars = sorted(char_stats.items(), key=lambda x: x[1]['nsfw']/x[1]['total'] if x[1]['total'] > 0 else 0, reverse=True)
    
    for char, stats in sorted_chars[:20]:  # 显示前20个
        nsfw_rate = stats['nsfw']/stats['total']*100 if stats['total'] > 0 else 0
        report += f"| {char} | {stats['total']} | {stats['nsfw']} | {stats['suggestive']} | {stats['safe']} | {nsfw_rate:.1f}% | {stats['avg_score']:.1f} |\n"
    
    report += f"""

---

## 🔍 高风险角色（NSFW率 > 50%）

"""
    
    high_risk_chars = [(char, stats) for char, stats in sorted_chars if stats['nsfw']/stats['total']*100 > 50]
    
    for char, stats in high_risk_chars:
        nsfw_rate = stats['nsfw']/stats['total']*100
        report += f"- **{char}**: {nsfw_rate:.1f}% NSFW ({stats['nsfw']}/{stats['total']})\n"
    
    report += f"""

---

## 💡 数据质量建议

### 1. NSFW内容过滤
- 当前NSFW占比: **{nsfw_count/total_images*100:.1f}%**
- 建议过滤掉NSFW和Suggestive类别，保留Safe内容用于训练
- 过滤后剩余: **{safe_count}** 张图片 ({safe_count/total_images*100:.1f}%)

### 2. 角色数据平衡
- 部分角色NSFW率过高，建议重新采集或清理
- 优先保留Safe内容较多的角色用于训练

### 3. 深度学习模型建议
- 安装TensorFlow: `pip install tensorflow`
- 使用MobileNet V2模型进行更准确的检测
- 对比规则检测结果，优化过滤策略

---

## 📋 后续步骤

1. **安装依赖**: `pip install tensorflow`
2. **运行深度学习检测**: `python3 scripts/data_processing/run_nsfw_tensorflow.py`
3. **对比结果**: 分析两种检测方法的差异
4. **数据过滤**: 根据检测结果过滤NSFW内容
5. **重新训练**: 使用过滤后的数据进行模型训练

---

*报告生成时间: 2026-06-08*
*数据来源: final_dataset (634张图片)*
"""
    
    # 保存报告
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 综合报告已生成: {OUTPUT_REPORT}")
    print(f"\n📊 检测摘要:")
    print(f"   总图片数: {total_images}")
    print(f"   NSFW: {nsfw_count} ({nsfw_count/total_images*100:.1f}%)")
    print(f"   Suggestive: {suggestive_count} ({suggestive_count/total_images*100:.1f}%)")
    print(f"   Safe: {safe_count} ({safe_count/total_images*100:.1f}%)")
    print(f"   高风险角色: {len(high_risk_chars)} 个")


if __name__ == "__main__":
    generate_report()