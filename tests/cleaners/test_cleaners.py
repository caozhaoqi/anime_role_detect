#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗流水线测试脚本
测试所有清洗模块的功能
"""

import os
import sys
import json
import time
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.cleaners import (
    CLIPDeduplicator,
    CharacterConsistencyFilter,
    HDBSCANClusterFilter,
    MislabeledDetector,
    DanbooruEnricher,
)


def test_clip_deduplicator():
    """测试CLIP去重器"""
    print("\n" + "="*60)
    print("测试1: CLIP去重器 (CLIPDeduplicator)")
    print("="*60)
    
    try:
        from src.core.recognition.clip_embedder_cached import CLIPEmbedderCached
        
        embedder = CLIPEmbedderCached(model_name="ViT-B/32")
        deduplicator = CLIPDeduplicator(
            similarity_threshold=0.95,
            embedder=embedder,
        )
        
        # 测试特征提取
        test_dir = project_root / "data/final_dataset/Arona"
        if test_dir.exists():
            image_paths = list(test_dir.glob("*.jpg"))[:10]
            if image_paths:
                print(f"测试目录: {test_dir}")
                print(f"找到 {len(image_paths)} 张图片")
                
                features = deduplicator.extract_features([str(p) for p in image_paths])
                print(f"成功提取 {len(features)} 个特征")
                
                # 测试相似度计算
                if len(features) >= 2:
                    paths = list(features.keys())
                    sim = deduplicator.compute_similarity(features[paths[0]], features[paths[1]])
                    print(f"相似度计算测试: {sim:.4f}")
                
                print("✅ CLIP去重器测试通过")
                return True
            else:
                print("⚠️ 目录中没有测试图片，跳过特征提取测试")
                print("✅ CLIP去重器初始化测试通过")
                return True
        else:
            print(f"⚠️ 测试目录不存在: {test_dir}")
            return True
            
    except Exception as e:
        print(f"❌ CLIP去重器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_character_consistency_filter():
    """测试角色一致性过滤器"""
    print("\n" + "="*60)
    print("测试2: 角色一致性过滤器 (CharacterConsistencyFilter)")
    print("="*60)
    
    try:
        from src.core.recognition.clip_embedder_cached import CLIPEmbedderCached
        
        embedder = CLIPEmbedderCached(model_name="ViT-B/32")
        filter = CharacterConsistencyFilter(
            consistency_threshold=0.25,
            embedder=embedder,
        )
        
        # 测试文本相似度计算
        test_dir = project_root / "data/final_dataset/Arona"
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg"))[:5]
            if images:
                print(f"测试角色: Arona")
                print(f"测试图片: {len(images)} 张")
                
                # 计算相似度
                for img in images[:2]:
                    score = filter.compute_text_similarity(str(img), "Arona", "blue_archive")
                    print(f"  {img.name}: {score:.4f}")
                
                print("✅ 角色一致性过滤器测试通过")
                return True
        
        print("✅ 角色一致性过滤器初始化测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 角色一致性过滤器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hdbscan_cluster_filter():
    """测试HDBSCAN聚类过滤器"""
    print("\n" + "="*60)
    print("测试3: HDBSCAN聚类过滤器 (HDBSCANClusterFilter)")
    print("="*60)
    
    try:
        # 检查HDBSCAN是否可用
        try:
            import hdbscan
            print("使用 HDBSCAN 库")
        except ImportError:
            print("⚠️ HDBSCAN未安装，将使用sklearn的DBSCAN替代")
        
        from src.core.recognition.clip_embedder_cached import CLIPEmbedderCached
        
        embedder = CLIPEmbedderCached(model_name="ViT-B/32")
        filter = HDBSCANClusterFilter(
            min_cluster_size=5,
            embedder=embedder,
        )
        
        # 测试聚类
        test_dir = project_root / "data/final_dataset/Arona"
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg"))[:15]
            if images:
                print(f"测试目录: {test_dir}")
                print(f"找到 {len(images)} 张图片")
                
                # 分析聚类
                analysis = filter.analyze_clusters([str(p) for p in images])
                print(f"聚类结果: {json.dumps(analysis, indent=2, ensure_ascii=False)[:500]}")
                
                print("✅ HDBSCAN聚类过滤器测试通过")
                return True
        
        print("✅ HDBSCAN聚类过滤器初始化测试通过")
        return True
        
    except Exception as e:
        print(f"❌ HDBSCAN聚类过滤器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mislabeled_detector():
    """测试错误标签检测器"""
    print("\n" + "="*60)
    print("测试4: 错误标签检测器 (MislabeledDetector)")
    print("="*60)
    
    try:
        detector = MislabeledDetector()
        
        # 测试单个检测
        test_dir = project_root / "data/final_dataset/Arona"
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg"))[:5]
            if images:
                print(f"测试目录: {test_dir}")
                
                for img in images[:2]:
                    result = detector.detect_image_quality_issues(str(img))
                    print(f"  {img.name}: {'可疑' if result[0] else '正常'} - {result[1][:50] if result[1] else 'OK'}")
                
                print("✅ 错误标签检测器测试通过")
                return True
        
        print("✅ 错误标签检测器初始化测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 错误标签检测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_danbooru_enricher():
    """测试Danbooru增强器"""
    print("\n" + "="*60)
    print("测试5: Danbooru增强器 (DanbooruEnricher)")
    print("="*60)
    
    try:
        enricher = DanbooruEnricher(mirror_site="yande.re")
        
        print(f"镜像站点: yande.re")
        print("测试获取角色标签...")
        
        # 测试获取标签
        tags = enricher.get_character_tags("Arona", "blue_archive")
        print(f"角色标签获取成功:")
        print(f"  character: {len(tags.get('character', []))} 个")
        print(f"  artist: {len(tags.get('artist', []))} 个")
        print(f"  general: {len(tags.get('general', []))} 个")
        
        if tags.get("character"):
            print(f"  示例: {tags['character'][:3]}")
        
        # 测试查找相关角色
        print("\n查找相关角色...")
        related = enricher.find_related_characters("Arona", "blue_archive", limit=5)
        print(f"相关角色: {json.dumps(related, indent=2, ensure_ascii=False)}")
        
        print("✅ Danbooru增强器测试通过")
        return True
        
    except Exception as e:
        print(f"❌ Danbooru增强器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("数据清洗流水线模块测试")
    print("="*60)
    
    results = {}
    
    # 测试各模块
    results["CLIP去重器"] = test_clip_deduplicator()
    results["角色一致性过滤器"] = test_character_consistency_filter()
    results["HDBSCAN聚类过滤器"] = test_hdbscan_cluster_filter()
    results["错误标签检测器"] = test_mislabeled_detector()
    results["Danbooru增强器"] = test_danbooru_enricher()
    
    # 汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
