#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置管理
"""

import os
import sys
from pathlib import Path


class Config:
    """主配置类"""
    
    # 项目根目录
    BASE_DIR = Path(__file__).parent.parent
    
    # 数据目录
    DATA_DIR = BASE_DIR / "data"
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"
    
    # 模型目录
    MODEL_DIR = BASE_DIR / "models"
    CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
    ONNX_DIR = MODEL_DIR / "onnx"
    
    # 日志目录
    LOG_DIR = BASE_DIR / "logs"
    
    # 文档目录
    DOCS_DIR = BASE_DIR / "docs"
    
    # 脚本目录
    SCRIPTS_DIR = BASE_DIR / "scripts"
    
    # 源代码目录
    SRC_DIR = BASE_DIR / "src"
    
    # 配置目录
    CONFIG_DIR = BASE_DIR / "config"
    
    # 测试目录
    TESTS_DIR = BASE_DIR / "tests"
    
    # 角色列表目录
    CHARACTERS_DIR = BASE_DIR / "auto_spider_img" / "characters"
    
    # 系列配置文件
    ANIME_SET_FILE = BASE_DIR / "auto_spider_img" / "anime_set.txt"
    
    # 图像质量阈值
    MIN_IMAGE_SIZE = 200  # 最小图像尺寸
    MAX_IMAGE_SIZE = 2048  # 最大图像尺寸
    MIN_ASPECT_RATIO = 0.3  # 最小宽高比
    MAX_ASPECT_RATIO = 3.0  # 最大宽高比
    
    # 数据采集配置
    MAX_IMAGES_PER_CHARACTER = 100  # 每个角色最大图像数
    MIN_IMAGES_PER_CHARACTER = 50  # 每个角色最小图像数
    
    # 模型训练配置
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    
    # 推理配置
    CONFIDENCE_THRESHOLD = 0.7
    
    def __init__(self):
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.DATA_DIR,
            self.TRAIN_DIR,
            self.VAL_DIR,
            self.TEST_DIR,
            self.MODEL_DIR,
            self.CHECKPOINT_DIR,
            self.ONNX_DIR,
            self.LOG_DIR,
            self.DOCS_DIR,
            self.SCRIPTS_DIR,
            self.SRC_DIR,
            self.CONFIG_DIR,
            self.TESTS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)
    
    def get_path(self, path):
        """获取路径"""
        return str(path)
    
    def get_character_file(self, series_name):
        """获取角色文件路径"""
        return str(self.CHARACTERS_DIR / f"{series_name}.txt")


# 创建全局配置实例
config = Config()
