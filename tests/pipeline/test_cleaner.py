#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗模块单元测试
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

import unittest
from unittest.mock import MagicMock, patch
import os

# 测试用的模拟图片路径
TEST_IMAGE = "data/final_dataset/Bocchi the Rock!/00001.jpg"


class TestQualityFilter(unittest.TestCase):
    """测试质量过滤器"""

    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        from src.data_pipeline.cleaner import QualityFilter
        cls.filter = QualityFilter()
        
        # 查找一个测试图片
        data_dir = Path("data/final_dataset")
        if data_dir.exists():
            for char_dir in data_dir.iterdir():
                if char_dir.is_dir():
                    for img_file in char_dir.iterdir():
                        if img_file.suffix.lower() in ['.jpg', '.png']:
                            cls.test_image = str(img_file)
                            return
        cls.test_image = None

    def test_filter_initialization(self):
        """测试过滤器初始化"""
        filter = self.filter
        self.assertIsNotNone(filter)
        self.assertEqual(filter.min_width, 256)
        self.assertEqual(filter.min_height, 256)
        self.assertEqual(filter.min_ratio, 0.1)
        self.assertEqual(filter.max_ratio, 10.0)

    def test_filter_with_valid_image(self):
        """测试有效图片"""
        if not self.test_image:
            self.skipTest("测试图片不存在")
        
        ok, info = self.filter.filter(self.test_image)
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(info, dict)
        self.assertIn('check', info)

    def test_filter_with_invalid_path(self):
        """测试无效路径"""
        ok, info = self.filter.filter("invalid/path/to/image.jpg")
        self.assertFalse(ok)
        self.assertIn('reason', info)


class TestCharacterCropper(unittest.TestCase):
    """测试角色裁剪器"""

    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        try:
            from src.data_pipeline.cleaner import CharacterCropper
            cls.cropper = CharacterCropper()
        except ImportError:
            cls.cropper = None
        
        # 查找一个测试图片
        data_dir = Path("data/final_dataset")
        if data_dir.exists():
            for char_dir in data_dir.iterdir():
                if char_dir.is_dir():
                    for img_file in char_dir.iterdir():
                        if img_file.suffix.lower() in ['.jpg', '.png']:
                            cls.test_image = str(img_file)
                            return
        cls.test_image = None

    def test_cropper_initialization(self):
        """测试裁剪器初始化"""
        if self.cropper is None:
            self.skipTest("CharacterCropper 未导出")
        self.assertIsNotNone(self.cropper)

    def test_crop_with_bbox(self):
        """测试边界框裁剪"""
        if not self.test_image or self.cropper is None:
            self.skipTest("测试图片不存在或 CharacterCropper 未导出")
        
        bbox = (100, 100, 300, 300)
        result = self.cropper.crop_character(self.test_image, bbox)
        # 返回裁剪后的图片路径
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_calculate_character_ratio(self):
        """测试角色占比计算"""
        if not self.test_image or self.cropper is None:
            self.skipTest("测试图片不存在或 CharacterCropper 未导出")
        
        bbox = (100, 100, 300, 300)
        ratio = self.cropper.calculate_character_ratio(self.test_image, bbox)
        self.assertIsInstance(ratio, float)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)


class TestAnimeClassifier(unittest.TestCase):
    """测试动漫分类器"""

    def test_classifier_initialization(self):
        """测试分类器初始化"""
        from src.data_pipeline.cleaner import AnimeClassifier
        classifier = AnimeClassifier()
        self.assertIsNotNone(classifier)
        self.assertFalse(classifier._initialized)

    def test_prompts_defined(self):
        """测试提示词已定义"""
        from src.data_pipeline.cleaner import AnimeClassifier
        classifier = AnimeClassifier()
        
        self.assertIsNotNone(classifier.anime_prompts)
        self.assertIsNotNone(classifier.non_anime_prompts)
        self.assertIsInstance(classifier.anime_prompts, list)
        self.assertIsInstance(classifier.non_anime_prompts, list)
        self.assertGreater(len(classifier.anime_prompts), 0)


class TestAIDetector(unittest.TestCase):
    """测试AI检测器"""

    def test_detector_initialization(self):
        """测试检测器初始化"""
        from src.data_pipeline.cleaner import AIDetector
        detector = AIDetector()
        self.assertIsNotNone(detector)
        self.assertFalse(detector._initialized)


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

    def test_all_categories_have_tags(self):
        """测试所有类别都有标签"""
        from src.data_pipeline.cleaner import CLIPTagger
        tagger = CLIPTagger()
        
        for category, tags in tagger.tag_categories.items():
            self.assertIsInstance(tags, list)
            self.assertGreater(len(tags), 0)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
