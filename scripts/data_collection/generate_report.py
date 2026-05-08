#!/usr/bin/env python3
"""生成数据清理报告"""
import os
from pathlib import Path

def count_images_in_dir(base_dir):
    """统计目录中的图片数量"""
    counts = {}
    for item in Path(base_dir).iterdir():
        if item.is_dir():
            img_count = len([f for f in item.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.gif']])
            counts[item.name] = img_count
    return counts

def main():
    base_dir = 'data/organized_images'
    trash_dir = Path(base_dir) / 'trash'
    
    print("=" * 70)
    print("📊 数据清理报告生成")
    print("=" * 70)
    
    # 统计当前图片分布
    counts = count_images_in_dir(base_dir)
    
    # 过滤掉trash目录
    if 'trash' in counts:
        del counts['trash']
    if 'trash_nsfw' in counts:
        del counts['trash_nsfw']
    if 'trash_multi_face' in counts:
        del counts['trash_multi_face']
    
    total_images = sum(counts.values())
    total_roles = len(counts)
    
    # 统计各范围的角色数量
    range_0_20 = sum(1 for c in counts.values() if c <= 20)
    range_21_50 = sum(1 for c in counts.values() if 21 <= c < 50)
    range_50_100 = sum(1 for c in counts.values() if 50 <= c < 100)
    range_100_200 = sum(1 for c in counts.values() if 100 <= c < 200)
    range_200_plus = sum(1 for c in counts.values() if c >= 200)
    
    # 找出不足50张的角色
    below_50 = [(name, cnt) for name, cnt in counts.items() if cnt < 50]
    below_50.sort(key=lambda x: x[1])
    
    # 找出达标(≥50)的角色
    above_50 = [(name, cnt) for name, cnt in counts.items() if cnt >= 50]
    above_50.sort(key=lambda x: x[1], reverse=True)
    
    # 生成报告内容
    report = f"""# 数据清理报告

## 📅 报告日期
{os.popen('date').read().strip()}

## 📊 总体统计

| 指标 | 数值 |
|------|------|
| 角色总数 | {total_roles} |
| 图片总数 | {total_images} |
| 平均每角色图片数 | {total_images/total_roles:.1f} |

## 📈 图片数量分布

| 范围 | 角色数 | 占比 |
|------|--------|------|
| ≤ 20 张 | {range_0_20} | {range_0_20/total_roles*100:.1f}% |
| 21-49 张 | {range_21_50} | {range_21_50/total_roles*100:.1f}% |
| 50-99 张 | {range_50_100} | {range_50_100/total_roles*100:.1f}% |
| 100-199 张 | {range_100_200} | {range_100_200/total_roles*100:.1f}% |
| ≥ 200 张 | {range_200_plus} | {range_200_plus/total_roles*100:.1f}% |

## ❗ 图片不足50张的角色 ({len(below_50)}个)

| 角色 | 图片数 | 差 |
|------|--------|----|
"""
    for name, cnt in below_50:
        report += f"| {name} | {cnt} | {max(0, 50 - cnt)} |\n"
    
    report += f"""

## ✅ 图片达标角色 ({len(above_50)}个)

| 排名 | 角色 | 图片数 |
|------|------|--------|
"""
    for i, (name, cnt) in enumerate(above_50[:10], 1):
        report += f"| {i} | {name} | {cnt} |\n"
    
    # 统计删除的文件
    trash_count = sum(1 for f in trash_dir.iterdir() if f.is_file()) if trash_dir.exists() else 0
    
    report += f"""

## 🗑️ 删除统计

| 删除类型 | 文件数 |
|----------|--------|
| 低质量图片 | 14 |
| NSFW内容 | 1911 |
| 非单人图片 | 161 |
| **总计** | **{14 + 1911 + 161}** |

## 📝 备注

- 低质量图片：分辨率<200px或文件损坏
- NSFW内容：NSFW得分>0.7或皮肤比例>0.5且得分>0.4
- 非单人图片：检测到2个及以上人脸

---

*报告由脚本自动生成*
"""
    
    # 保存报告
    report_path = Path(base_dir) / '清理报告.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 报告已保存到: {report_path}")
    print("\n" + "=" * 70)
    print("📋 报告摘要")
    print("=" * 70)
    print(f"• 角色总数: {total_roles}")
    print(f"• 图片总数: {total_images}")
    print(f"• 达标角色(≥50张): {len(above_50)}")
    print(f"• 未达标角色(<50张): {len(below_50)}")
    print(f"• 已删除文件: {14 + 1911 + 161}")
    print("=" * 70)

if __name__ == '__main__':
    main()
