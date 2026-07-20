#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗流水线集成测试 - 从数据采集到清洗标注
"""

import sys
import shutil
from pathlib import Path


from src.data_pipeline.cleaning_pipeline import CleaningPipeline, CleaningConfig


def main():
    print("="*60)
    print("数据清洗流水线集成测试")
    print("="*60)
    
    # 准备测试数据
    src_dir = Path("data/final_dataset")
    test_input = Path("data/test_cleaning_input")
    test_output = Path("data/test_cleaning_output")
    
    # 清理旧测试数据
    for d in [test_input, test_output]:
        if d.exists():
            shutil.rmtree(d)
    
    # 复制角色进行测试（每角色取前30张）
    test_chars = ["Hina", "Arona"]
    for char in test_chars:
        src = src_dir / char
        dst = test_input / char
        if src.exists():
            shutil.copytree(src, dst)
            # 限制图片数量以加速测试
            images = list(dst.glob("*.jpg"))[10:30]  # 取10-30张
            for img in list(dst.glob("*.jpg")):
                if img not in images:
                    img.unlink()
            print(f"复制角色: {char} ({len(list(dst.glob('*.jpg')))} 张)")
    
    if not any(test_input.iterdir()):
        print("没有测试数据")
        return
    
    print(f"\n输入目录: {test_input}")
    print(f"输出目录: {test_output}")
    
    # 配置（使用干运行避免真正删除文件）
    config = CleaningConfig(
        enable_deduplication=True,
        enable_consistency_filter=True,
        enable_cluster_filter=True,
        enable_mislabeled_detector=True,
        enable_danbooru_enrichment=False,
        similarity_threshold=0.95,
        consistency_threshold=0.25,
        dedup_dry_run=True,      # 干运行 - 不删除
        consistency_dry_run=True,
        cluster_dry_run=True,
        min_images_per_character=5,
    )
    
    # 运行流水线
    pipeline = CleaningPipeline(
        input_dir=str(test_input),
        output_dir=str(test_output),
        config=config,
    )
    
    print("\n🚀 启动清洗流水线...")
    report = pipeline.run()
    
    # 打印结果
    print("\n" + "="*60)
    print("📊 清洗流水线报告")
    print("="*60)
    
    if report:
        print(f"\n⏱️  运行时间: {report.duration_seconds:.1f} 秒")
        print(f"👥 处理角色: {report.total_characters}")
        print(f"📷 原始图片: {report.total_original_images}")
        print(f"📷 清洗后图片: {report.total_cleaned_images}")
        print(f"📈 总体保留率: {report.overall_keep_rate:.1%}")
        
        print(f"\n🔍 各阶段移除统计(干运行):")
        print(f"   CLIP去重: {report.dedup_removed} 对")
        print(f"   一致性过滤: {report.consistency_removed}")
        print(f"   聚类过滤: {report.cluster_removed}")
        print(f"   错误标签检测: {report.mislabeled_removed}")
        
        print(f"\n👤 角色详细结果:")
        for name, result in sorted(report.character_results.items()):
            keep_rate = result["after_mislabeled"] / result["original_count"] if result["original_count"] > 0 else 0
            print(f"   ✅ {name}: {result['original_count']} -> {result['after_mislabeled']} ({keep_rate:.0%})")
            print(f"      - 重复对: {result['duplicate_pairs']}")
            print(f"      - 低一致性: {result['low_consistency_count']}")
            print(f"      - 聚类异常: {result['cluster_outliers']}")
            print(f"      - 可疑标注: {result['mislabeled_count']}")
    
    # 清理测试数据
    shutil.rmtree(test_input)
    shutil.rmtree(test_output)
    print("\n✅ 测试数据已清理")
    
    print("\n" + "="*60)
    print("✅ 集成测试完成")
    print("="*60)


if __name__ == "__main__":
    main()
