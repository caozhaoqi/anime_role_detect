#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整流水线测试 - 使用少量角色快速测试
"""

import sys
import shutil
from pathlib import Path


from src.data_pipeline.cleaning_pipeline import CleaningPipeline, CleaningConfig


def main():
    print("="*60)
    print("数据清洗流水线完整测试")
    print("="*60)
    
    # 准备测试数据（复制少量角色）
    src_dir = Path("data/final_dataset")
    test_input = Path("data/test_cleaning_input")
    test_output = Path("data/test_cleaning_output")
    
    # 清理旧测试数据
    if test_input.exists():
        shutil.rmtree(test_input)
    if test_output.exists():
        shutil.rmtree(test_output)
    
    # 复制3个角色进行测试
    test_chars = ["Hina", "Arona", "Kikyo"]
    for char in test_chars:
        src = src_dir / char
        dst = test_input / char
        if src.exists():
            shutil.copytree(src, dst)
            print(f"复制角色: {char}")
    
    if not any(test_input.iterdir()):
        print("没有测试数据")
        return
    
    print(f"\n测试角色数: {len(list(test_input.iterdir()))}")
    
    # 配置
    config = CleaningConfig(
        enable_deduplication=True,
        enable_consistency_filter=True,
        enable_cluster_filter=True,
        enable_mislabeled_detector=True,
        enable_danbooru_enrichment=False,
        similarity_threshold=0.95,
        consistency_threshold=0.25,
        dedup_dry_run=False,
        consistency_dry_run=False,
        cluster_dry_run=False,
        min_images_per_character=5,
    )
    
    # 运行流水线
    pipeline = CleaningPipeline(
        input_dir=str(test_input),
        output_dir=str(test_output),
        config=config,
    )
    
    report = pipeline.run()
    
    # 打印结果
    print("\n" + "="*60)
    print("📊 最终报告")
    print("="*60)
    
    if report:
        print(f"\n⏱️  运行时间: {report.duration_seconds:.1f} 秒")
        print(f"👥 处理角色: {report.total_characters}")
        print(f"📷 原始图片: {report.total_original_images}")
        print(f"📷 清洗后图片: {report.total_cleaned_images}")
        print(f"📈 总体保留率: {report.overall_keep_rate:.1%}")
        
        print(f"\n🔍 各阶段移除:")
        print(f"   CLIP去重: {report.dedup_removed}")
        print(f"   一致性过滤: {report.consistency_removed}")
        print(f"   聚类过滤: {report.cluster_removed}")
        print(f"   错误标签检测: {report.mislabeled_removed}")
    
    # 清理测试数据
    shutil.rmtree(test_input)
    shutil.rmtree(test_output)
    print("\n✅ 测试数据已清理")


if __name__ == "__main__":
    main()
