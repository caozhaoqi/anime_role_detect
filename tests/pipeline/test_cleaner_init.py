#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗模块配置测试
仅测试类定义和属性，不导入任何torch模型
"""

import sys
import os
import platform

# 在导入任何项目模块之前禁用CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 延迟导入torch，仅用于检查
import torch
# 确保torch不尝试使用CUDA
torch.cuda.is_available = lambda: False
torch.backends.mps.is_available = lambda: False

from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("🧪 开始运行数据清洗模块配置测试")
print("=" * 60)
print(f"平台: {platform.system()}")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print("=" * 60)


def test_quality_filter():
    """测试QualityFilter"""
    print("\n📋 测试 QualityFilter...")
    from src.data_pipeline.cleaner import QualityFilter
    
    filter = QualityFilter()
    
    assert hasattr(filter, 'min_resolution'), "缺少min_resolution属性"
    assert hasattr(filter, 'min_aspect_ratio'), "缺少min_aspect_ratio属性"
    assert hasattr(filter, 'max_aspect_ratio'), "缺少max_aspect_ratio属性"
    assert filter.min_resolution == 256, "min_resolution默认值错误"
    
    print(f"  ✅ min_resolution: {filter.min_resolution}")
    print(f"  ✅ min_aspect_ratio: {filter.min_aspect_ratio}")
    print(f"  ✅ max_aspect_ratio: {filter.max_aspect_ratio}")
    return True


def test_character_cropper():
    """测试CharacterCropper"""
    print("\n📋 测试 CharacterCropper...")
    from src.data_pipeline.cleaner import CharacterCropper
    
    cropper = CharacterCropper()
    
    assert cropper is not None, "CharacterCropper初始化失败"
    assert hasattr(cropper, 'default_expand_ratio'), "缺少default_expand_ratio属性"
    
    print(f"  ✅ default_expand_ratio: {cropper.default_expand_ratio}")
    return True


def test_anime_classifier():
    """测试AnimeClassifier"""
    print("\n📋 测试 AnimeClassifier...")
    from src.data_pipeline.cleaner import AnimeClassifier
    
    classifier = AnimeClassifier()
    
    assert classifier is not None, "AnimeClassifier初始化失败"
    assert hasattr(classifier, 'device'), "缺少device属性"
    assert hasattr(classifier, 'prompts'), "缺少prompts属性"
    assert 'anime' in classifier.prompts, "缺少anime提示词"
    assert 'non_anime' in classifier.prompts, "缺少non_anime提示词"
    assert not classifier._initialized, "初始状态错误"
    
    print(f"  ✅ device: {classifier.device}")
    print(f"  ✅ prompts keys: {list(classifier.prompts.keys())}")
    print(f"  ✅ anime prompts数量: {len(classifier.prompts['anime'])}")
    print(f"  ✅ non_anime prompts数量: {len(classifier.prompts['non_anime'])}")
    return True


def test_ai_detector():
    """测试AIDetector"""
    print("\n📋 测试 AIDetector...")
    from src.data_pipeline.cleaner import AIDetector
    
    detector = AIDetector()
    
    assert detector is not None, "AIDetector初始化失败"
    assert hasattr(detector, 'device'), "缺少device属性"
    assert not detector._initialized, "初始状态错误"
    
    print(f"  ✅ device: {detector.device}")
    return True


def test_clip_tagger():
    """测试CLIPTagger"""
    print("\n📋 测试 CLIPTagger...")
    from src.data_pipeline.cleaner import CLIPTagger
    
    tagger = CLIPTagger()
    
    assert tagger is not None, "CLIPTagger初始化失败"
    assert hasattr(tagger, 'tag_categories'), "缺少tag_categories属性"
    assert hasattr(tagger, 'top_k'), "缺少top_k属性"
    
    # 检查所有必需的类别
    required_categories = ['style', 'genre', 'character', 'hair', 'eyes']
    for cat in required_categories:
        assert cat in tagger.tag_categories, f"缺少{cat}类别"
        assert isinstance(tagger.tag_categories[cat], list), f"{cat}标签类型错误"
        assert len(tagger.tag_categories[cat]) > 0, f"{cat}标签为空"
    
    print(f"  ✅ tag_categories数量: {len(tagger.tag_categories)}")
    print(f"  ✅ top_k: {tagger.top_k}")
    for cat in required_categories:
        print(f"  ✅ {cat}: {len(tagger.tag_categories[cat])}个标签")
    return True


def test_multi_tagger():
    """测试MultiTagger"""
    print("\n📋 测试 MultiTagger...")
    from src.data_pipeline.cleaner import MultiTagger
    
    tagger = MultiTagger()
    
    assert tagger is not None, "MultiTagger初始化失败"
    assert hasattr(tagger, 'taggers'), "缺少taggers属性"
    
    print(f"  ✅ taggers数量: {len(tagger.taggers)}")
    return True


if __name__ == '__main__':
    tests = [
        ("QualityFilter", test_quality_filter),
        ("CharacterCropper", test_character_cropper),
        ("AnimeClassifier", test_anime_classifier),
        ("AIDetector", test_ai_detector),
        ("CLIPTagger", test_clip_tagger),
        ("MultiTagger", test_multi_tagger),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {name} 测试通过!")
        except Exception as e:
            failed += 1
            print(f"\n❌ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}通过, {failed}失败")
    print("=" * 60)
    
    sys.exit(0 if failed == 0 else 1)
