#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据清洗模块
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.cleaner import AnimeClassifier, QualityFilter, AIDetector, MultiTagger


def test_cleaner_modules():
    """测试所有数据清洗模块"""
    # 找一个测试图片
    test_image = None
    data_dir = Path("data/final_dataset")
    for char_dir in data_dir.iterdir():
        if char_dir.is_dir():
            for img_file in char_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.png']:
                    test_image = str(img_file)
                    break
            if test_image:
                break
    
    if not test_image:
        print("❌ 未找到测试图片")
        return
    
    print(f"📷 测试图片: {test_image}")
    
    # 测试AnimeClassifier
    print('\n🎯 测试 AnimeClassifier...')
    classifier = AnimeClassifier()
    classifier.initialize()
    prob, result = classifier.classify(test_image)
    print(f'  分类结果: {result} (置信度: {prob:.4f})')
    
    # 测试QualityFilter
    print('\n🎯 测试 QualityFilter...')
    filter = QualityFilter()
    quality_ok, info = filter.filter(test_image)
    print(f'  质量过滤: {quality_ok}')
    print(f'  详细信息: {info}')
    
    # 测试AIDetector
    print('\n🎯 测试 AIDetector...')
    detector = AIDetector()
    detector.initialize()
    prob, result = detector.detect(test_image)
    print(f'  AI检测: {result} (置信度: {prob:.4f})')
    
    # 测试MultiTagger
    print('\n🎯 测试 MultiTagger...')
    tagger = MultiTagger()
    tagger.initialize()
    tags = tagger.generate_comprehensive_tags(test_image)
    print(f'  标签生成完成')
    for category, tag_list in tags.get('by_category', {}).items():
        tag_names = [t['tag'] for t in tag_list[:3]]
        print(f'    {category}: {tag_names}')
    
    print('\n✅ 所有模块测试通过！')


if __name__ == "__main__":
    test_cleaner_modules()
