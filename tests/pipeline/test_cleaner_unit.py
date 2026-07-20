#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗模块单元测试（无模型版本）
仅测试初始化和配置，不加载CLIP模型
"""

import sys
import os

# 在导入torch之前设置环境变量，禁用CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用CUDA

from pathlib import Path

project_root = Path(__file__).parent.parent.parent

import unittest
import platform


class TestQualityFilter(unittest.TestCase):
    """测试质量过滤器"""

    def test_filter_initialization(self):
        """测试过滤器初始化"""
        from src.data_pipeline.cleaner import QualityFilter
        filter = QualityFilter()
        self.assertIsNotNone(filter)
        self.assertEqual(filter.min_width, 256)
        self.assertEqual(filter.min_height, 256)
        self.assertEqual(filter.min_ratio, 0.1)
        self.assertEqual(filter.max_ratio, 10.0)
        print("✅ QualityFilter初始化测试通过")

    def test_filter_with_invalid_path(self):
        """测试无效路径"""
        from src.data_pipeline.cleaner import QualityFilter
        filter = QualityFilter()
        ok, info = filter.filter("invalid/path/to/image.jpg")
        self.assertFalse(ok)
        self.assertIn('reason', info)
        self.assertIn(info['check'], ('resolution', 'format'))
        print("✅ QualityFilter无效路径测试通过")


class TestCharacterCropper(unittest.TestCase):
    """测试角色裁剪器"""

    def test_cropper_initialization(self):
        """测试裁剪器初始化"""
        from src.data_pipeline.cleaner import CharacterCropper
        cropper = CharacterCropper()
        self.assertIsNotNone(cropper)
        print("✅ CharacterCropper初始化测试通过")


class TestAnimeClassifier(unittest.TestCase):
    """测试动漫分类器"""

    def test_classifier_initialization(self):
        """测试分类器初始化"""
        from src.data_pipeline.cleaner import AnimeClassifier
        classifier = AnimeClassifier()
        self.assertIsNotNone(classifier)
        self.assertFalse(classifier._initialized)
        # 检查设备选择
        self.assertIn(classifier.device, ['cpu', 'mps', 'cuda'])
        print(f"✅ AnimeClassifier初始化测试通过，设备: {classifier.device}")

    def test_prompts_defined(self):
        """测试提示词已定义"""
        from src.data_pipeline.cleaner import AnimeClassifier
        classifier = AnimeClassifier()
        
        self.assertIsNotNone(classifier.anime_prompts)
        self.assertIsNotNone(classifier.non_anime_prompts)
        self.assertIsInstance(classifier.anime_prompts, list)
        self.assertIsInstance(classifier.non_anime_prompts, list)
        self.assertGreater(len(classifier.anime_prompts), 0)
        self.assertGreater(len(classifier.non_anime_prompts), 0)
        print(f"✅ AnimeClassifier提示词定义测试通过")


class TestAIDetector(unittest.TestCase):
    """测试AI检测器"""

    def test_detector_initialization(self):
        """测试检测器初始化"""
        from src.data_pipeline.cleaner import AIDetector
        detector = AIDetector()
        self.assertIsNotNone(detector)
        self.assertFalse(detector._initialized)
        # 检查设备选择
        self.assertIn(detector.device, ['cpu', 'mps', 'cuda'])
        print(f"✅ AIDetector初始化测试通过，设备: {detector.device}")


class TestCLIPTagger(unittest.TestCase):
    """测试CLIP标签生成器"""

    def test_tagger_initialization(self):
        """测试标签器初始化"""
        from src.data_pipeline.cleaner import CLIPTagger
        tagger = CLIPTagger()
        self.assertIsNotNone(tagger)

    def test_tag_categories_defined(self):
        """测试标签类别已定义"""
        from src.data_pipeline.cleaner import CLIPTagger
        tagger = CLIPTagger()
        
        self.assertIn('style', tagger.tag_categories)
        self.assertIn('genre', tagger.tag_categories)
        self.assertIn('character', tagger.tag_categories)
        self.assertIn('hair', tagger.tag_categories)
        self.assertIn('eyes', tagger.tag_categories)
        print(f"✅ CLIPTagger标签类别定义测试通过")

    def test_all_categories_have_tags(self):
        """测试所有类别都有标签"""
        from src.data_pipeline.cleaner import CLIPTagger
        tagger = CLIPTagger()
        
        for category, tags in tagger.tag_categories.items():
            self.assertIsInstance(tags, list)
            self.assertGreater(len(tags), 0)
        print(f"✅ CLIPTagger所有类别标签测试通过")


class TestDeviceSelection(unittest.TestCase):
    """测试设备选择"""

    def test_platform_detection(self):
        """测试平台检测"""
        system = platform.system()
        self.assertIn(system, ['Darwin', 'Linux', 'Windows'])
        print(f"✅ 平台检测测试通过，当前平台: {system}")

    def test_classifier_device_on_mac(self):
        """测试Mac上的设备选择"""
        from src.data_pipeline.cleaner import AnimeClassifier
        
        if platform.system() == "Darwin":
            classifier = AnimeClassifier()
            # Mac上不应该选择cuda
            self.assertNotEqual(classifier.device, "cuda")
            print(f"✅ Mac设备选择测试通过，设备: {classifier.device}")


if __name__ == '__main__':
    print("=" * 60)
    print("🧪 开始运行数据清洗模块单元测试（无模型版本）")
    print("=" * 60)
    
    # 运行测试
    unittest.main(verbosity=2)
