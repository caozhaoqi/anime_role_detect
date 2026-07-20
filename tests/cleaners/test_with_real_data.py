#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗流水线 - 实际数据测试
"""

import sys
from pathlib import Path


from src.data_pipeline.cleaners import (
    CLIPDeduplicator,
    CharacterConsistencyFilter,
    HDBSCANClusterFilter,
    MislabeledDetector,
)


def main():
    data_dir = Path('data/final_dataset')
    
    if not data_dir.exists():
        print("测试数据目录不存在")
        return
    
    characters = list(data_dir.iterdir())[:2]
    print(f"测试角色: {[c.name for c in characters]}")
    
    for char_dir in characters:
        images = list(char_dir.glob('*.jpg'))[:10]
        print(f"\n角色: {char_dir.name}, 图片数: {len(images)}")
        
        if not images:
            continue
        
        # 测试1: CLIP去重
        print("  [1] CLIP去重器...")
        dedup = CLIPDeduplicator(similarity_threshold=0.95)
        feats = dedup.extract_features([str(p) for p in images])
        print(f"      特征提取: {len(feats)}/{len(images)} 成功")
        
        # 测试2: 角色一致性
        print("  [2] 角色一致性过滤器...")
        filter = CharacterConsistencyFilter(consistency_threshold=0.25)
        for img in images[:3]:
            score = filter.compute_text_similarity(str(img), char_dir.name)
            print(f"      {img.name[:25]:25s} -> {score:.4f}")
        
        # 测试3: HDBSCAN聚类
        print("  [3] HDBSCAN聚类过滤器...")
        cluster = HDBSCANClusterFilter(min_cluster_size=5)
        if len(images) >= 5:
            analysis = cluster.analyze_clusters([str(p) for p in images])
            print(f"      簇数: {analysis.get('num_clusters', 0)}, 噪声: {analysis.get('num_noise', 0)}")
        
        # 测试4: 错误标签检测
        print("  [4] 错误标签检测器...")
        detector = MislabeledDetector()
        for img in images[:3]:
            result = detector.detect_image_quality_issues(str(img))
            status = "✓" if not result[0] else "✗"
            print(f"      {img.name[:25]:25s} -> {status}")
        
        break
    
    print("\n✅ 实际数据测试完成")


if __name__ == "__main__":
    main()
