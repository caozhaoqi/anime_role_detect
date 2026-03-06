#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块测试
"""

import unittest
import os
from pathlib import Path

from src.utils.config_utils import (
    get_config,
    get_data_dir,
    get_train_dir,
    get_val_dir,
    get_test_dir,
    get_model_dir,
    get_checkpoint_dir,
    get_onnx_dir,
    get_log_dir,
    get_docs_dir,
    get_scripts_dir,
    get_src_dir,
    get_config_dir,
    get_tests_dir,
    get_characters_dir,
    get_anime_set_file,
    get_character_file,
    get_batch_size,
    get_epochs,
    get_learning_rate,
    get_confidence_threshold,
    get_max_images_per_character,
    get_min_images_per_character,
    get_min_image_size,
    get_max_image_size,
    get_min_aspect_ratio,
    get_max_aspect_ratio
)


class TestConfig(unittest.TestCase):
    """配置模块测试类"""
    
    def test_config_initialization(self):
        """测试配置初始化"""
        config = get_config()
        self.assertIsNotNone(config)
    
    def test_path_configs(self):
        """测试路径配置"""
        base_dir = Path(__file__).parent.parent
        
        self.assertEqual(get_data_dir(), str(base_dir / "data"))
        self.assertEqual(get_train_dir(), str(base_dir / "data" / "train"))
        self.assertEqual(get_val_dir(), str(base_dir / "data" / "val"))
        self.assertEqual(get_test_dir(), str(base_dir / "data" / "test"))
        self.assertEqual(get_model_dir(), str(base_dir / "models"))
        self.assertEqual(get_checkpoint_dir(), str(base_dir / "models" / "checkpoints"))
        self.assertEqual(get_onnx_dir(), str(base_dir / "models" / "onnx"))
        self.assertEqual(get_log_dir(), str(base_dir / "logs"))
        self.assertEqual(get_docs_dir(), str(base_dir / "docs"))
        self.assertEqual(get_scripts_dir(), str(base_dir / "scripts"))
        self.assertEqual(get_src_dir(), str(base_dir / "src"))
        self.assertEqual(get_config_dir(), str(base_dir / "config"))
        self.assertEqual(get_tests_dir(), str(base_dir / "tests"))
        self.assertEqual(get_characters_dir(), str(base_dir / "auto_spider_img" / "characters"))
        self.assertEqual(get_anime_set_file(), str(base_dir / "auto_spider_img" / "anime_set.txt"))
    
    def test_character_file(self):
        """测试角色文件路径"""
        base_dir = Path(__file__).parent.parent
        expected_path = str(base_dir / "auto_spider_img" / "characters" / "test.txt")
        self.assertEqual(get_character_file("test"), expected_path)
    
    def test_training_configs(self):
        """测试训练配置"""
        self.assertEqual(get_batch_size(), 32)
        self.assertEqual(get_epochs(), 100)
        self.assertEqual(get_learning_rate(), 1e-4)
    
    def test_inference_configs(self):
        """测试推理配置"""
        self.assertEqual(get_confidence_threshold(), 0.7)
    
    def test_data_collection_configs(self):
        """测试数据采集配置"""
        self.assertEqual(get_max_images_per_character(), 100)
        self.assertEqual(get_min_images_per_character(), 50)
    
    def test_image_quality_configs(self):
        """测试图像质量配置"""
        self.assertEqual(get_min_image_size(), 200)
        self.assertEqual(get_max_image_size(), 2048)
        self.assertEqual(get_min_aspect_ratio(), 0.3)
        self.assertEqual(get_max_aspect_ratio(), 3.0)


if __name__ == '__main__':
    unittest.main()
