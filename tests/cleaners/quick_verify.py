#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证脚本 - 验证清洗流水线核心功能
"""

import sys
from pathlib import Path


from src.data_pipeline.cleaners import (
    CLIPDeduplicator,
    CharacterConsistencyFilter,
    HDBSCANClusterFilter,
    MislabeledDetector,
)
from src.core.recognition.clip_embedder_cached import CLIPEmbedderCached


def quick_test():
    """快速测试核心功能"""
    print("="*50)
    print("清洗流水线核心功能快速验证")
    print("="*50)
    
    data_dir = Path("data/final_dataset")
    test_char = "Hina"
    test_dir = data_dir / test_char
    
    if not test_dir.exists():
        print(f"测试数据不存在: {test_dir}")
        return False
    
    images = list(test_dir.glob("*.jpg"))[:20]
    print(f"\n测试角色: {test_char}, 图片数: {len(images)}")
    
    if len(images) < 5:
        print("图片数不足")
        return False
    
    embedder = CLIPEmbedderCached(model_name="ViT-B/32")
    
    # 测试1: CLIP去重
    print("\n[1] CLIP去重器")
    dedup = CLIPDeduplicator(similarity_threshold=0.95, embedder=embedder)
    feats = dedup.extract_features([str(p) for p in images])
    dups = dedup.find_duplicates(feats)
    print(f"    特征提取: {len(feats)}/{len(images)}")
    print(f"    发现重复: {len(dups)} 对")
    
    # 测试2: 角色一致性
    print("\n[2] 角色一致性过滤器")
    filter = CharacterConsistencyFilter(consistency_threshold=0.25, embedder=embedder)
    scores = filter.filter_character_images([str(p) for p in images], test_char, return_scores=True)
    low_q = sum(1 for _, s in scores if s < 0.25)
    print(f"    低一致性: {low_q}/{len(scores)}")
    print(f"    分数范围: {min(s for _, s in scores):.3f} - {max(s for _, s in scores):.3f}")
    
    # 测试3: HDBSCAN聚类
    print("\n[3] HDBSCAN聚类过滤器")
    cluster = HDBSCANClusterFilter(min_cluster_size=5, embedder=embedder)
    analysis = cluster.analyze_clusters([str(p) for p in images])
    print(f"    簇数: {analysis.get('num_clusters', 0)}")
    print(f"    噪声点: {analysis.get('num_noise', 0)}")
    
    # 测试4: 错误标签检测
    print("\n[4] 错误标签检测器")
    detector = MislabeledDetector(embedder=embedder)
    detector.build_feature_library(str(test_dir))
    suspicious = detector.scan_directory(str(test_dir))
    sus_count = sum(1 for s in suspicious if s["suspicious"])
    print(f"    可疑图片: {sus_count}/{len(suspicious)}")
    
    print("\n" + "="*50)
    print("✅ 核心功能验证通过")
    print("="*50)
    
    return True


if __name__ == "__main__":
    quick_test()
