#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗流水线集成测试
从数据采集后到数据清洗标注的完整流程测试
"""

import sys
import os
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)

# 导入清洗流水线
from src.data_pipeline.cleaning_pipeline import CleaningPipeline, CleaningConfig


def test_cleaning_pipeline():
    """测试数据清洗流水线"""
    print("="*60)
    print("数据清洗流水线集成测试")
    print("="*60)
    
    # 使用实际数据
    input_dir = project_root / "data" / "final_dataset"
    output_dir = project_root / "data" / "cleaned_output"
    
    if not input_dir.exists():
        print(f"❌ 测试数据目录不存在: {input_dir}")
        print("   使用 data/final_dataset 作为输入数据进行测试")
        return False
    
    # 列出可用角色
    characters = [d.name for d in input_dir.iterdir() if d.is_dir()]
    print(f"\n📂 输入目录: {input_dir}")
    print(f"📂 输出目录: {output_dir}")
    print(f"👥 可用角色: {len(characters)} 个")
    print(f"   {', '.join(characters[:10])}{'...' if len(characters) > 10 else ''}")
    
    # 配置（使用干运行模式避免删除文件）
    config = CleaningConfig(
        enable_deduplication=True,
        enable_consistency_filter=True,
        enable_cluster_filter=True,
        enable_mislabeled_detector=True,
        enable_danbooru_enrichment=False,  # 跳过网络请求加速测试
        similarity_threshold=0.95,
        consistency_threshold=0.25,
        dedup_dry_run=True,  # 干运行
        consistency_dry_run=True,
        cluster_dry_run=True,
        min_images_per_character=5,
    )
    
    # 创建流水线
    pipeline = CleaningPipeline(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        config=config,
    )
    
    # 运行
    print("\n🚀 启动清洗流水线...")
    report = pipeline.run()
    
    # 验证结果
    print("\n" + "="*60)
    print("📋 测试结果验证")
    print("="*60)
    
    success = True
    
    # 检查报告
    if report:
        print(f"✅ 报告生成成功")
        print(f"   - 处理角色: {report.total_characters}")
        print(f"   - 原始图片: {report.total_original_images}")
        print(f"   - 清洗后图片: {report.total_cleaned_images}")
        print(f"   - 总体保留率: {report.overall_keep_rate:.1%}")
    else:
        print(f"❌ 报告生成失败")
        success = False
    
    # 检查输出目录
    if output_dir.exists():
        print(f"✅ 输出目录已创建: {output_dir}")
        
        # 检查报告文件
        report_file = output_dir / "cleaning_report.json"
        if report_file.exists():
            print(f"✅ 报告文件已生成: {report_file}")
        else:
            print(f"⚠️ 报告文件未找到: {report_file}")
    else:
        print(f"⚠️ 输出目录未创建（可能所有角色图片数不足）")
    
    # 检查各阶段统计
    if report:
        total_removed = (report.dedup_removed + report.consistency_removed + 
                        report.cluster_removed + report.mislabeled_removed)
        
        print(f"\n📊 各阶段移除统计:")
        print(f"   CLIP去重: {report.dedup_removed}")
        print(f"   一致性过滤: {report.consistency_removed}")
        print(f"   聚类过滤: {report.cluster_removed}")
        print(f"   错误标签检测: {report.mislabeled_removed}")
        print(f"   总计: {total_removed}")
    
    return success and report is not None


def test_single_character():
    """测试单个角色的完整清洗流程"""
    print("\n" + "="*60)
    print("单角色完整流程测试")
    print("="*60)
    
    from src.data_pipeline.cleaners import (
        CLIPDeduplicator,
        CharacterConsistencyFilter,
        HDBSCANClusterFilter,
        MislabeledDetector,
    )
    from src.core.recognition.clip_embedder_cached import CLIPEmbedderCached
    
    # 使用 Hina 角色测试
    input_dir = project_root / "data" / "final_dataset" / "Hina"
    
    if not input_dir.exists():
        print(f"⚠️ 测试角色不存在: {input_dir}")
        return False
    
    images = list(input_dir.glob("*.jpg"))[:10]
    if len(images) < 5:
        print(f"⚠️ 测试角色图片不足: {len(images)}")
        return False
    
    print(f"测试角色: Hina, 图片数: {len(images)}")
    
    try:
        # 初始化
        embedder = CLIPEmbedderCached(model_name="ViT-B/32")
        
        # 阶段1: CLIP去重
        print("\n[阶段1] CLIP去重...")
        dedup = CLIPDeduplicator(similarity_threshold=0.95, embedder=embedder)
        feats = dedup.extract_features([str(p) for p in images])
        dups = dedup.find_duplicates(feats)
        print(f"  发现 {len(dups)} 对重复图片")
        
        # 阶段2: 角色一致性
        print("\n[阶段2] 角色一致性过滤...")
        filter = CharacterConsistencyFilter(consistency_threshold=0.25, embedder=embedder)
        scores = []
        for img in images:
            score = filter.compute_text_similarity(str(img), "Hina")
            scores.append((str(img), score))
            print(f"  {img.name[:30]:30s} -> {score:.4f}")
        
        low_consistency = [p for p, s in scores if s < 0.25]
        print(f"  低一致性图片: {len(low_consistency)}")
        
        # 阶段3: HDBSCAN聚类
        print("\n[阶段3] HDBSCAN聚类...")
        cluster = HDBSCANClusterFilter(min_cluster_size=5, embedder=embedder)
        analysis = cluster.analyze_clusters([str(p) for p in images])
        print(f"  簇数: {analysis.get('num_clusters', 0)}")
        print(f"  噪声点: {analysis.get('num_noise', 0)}")
        
        # 阶段4: 错误标签检测
        print("\n[阶段4] 错误标签检测...")
        detector = MislabeledDetector(embedder=embedder)
        quality_results = []
        for img in images[:3]:
            result = detector.detect_image_quality_issues(str(img))
            quality_results.append(result)
            status = "✓" if not result[0] else "✗"
            print(f"  {img.name[:30]:30s} -> {status} {result[1] if result[0] else ''}")
        
        print("\n✅ 单角色测试完成")
        return True
        
    except Exception as e:
        print(f"\n❌ 单角色测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🚀 数据清洗流水线集成测试")
    print()
    
    # 测试1: 单角色完整流程
    test_single_character()
    
    print()
    
    # 测试2: 完整流水线
    test_cleaning_pipeline()
    
    print("\n" + "="*60)
    print("✅ 集成测试完成")
    print("="*60)


if __name__ == "__main__":
    main()
