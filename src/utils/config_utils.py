#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置工具模块
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config


def get_config():
    """获取配置实例"""
    return config


def get_data_dir():
    """获取数据目录"""
    return config.get_path(config.DATA_DIR)


def get_train_dir():
    """获取训练数据目录"""
    return config.get_path(config.TRAIN_DIR)


def get_val_dir():
    """获取验证数据目录"""
    return config.get_path(config.VAL_DIR)


def get_test_dir():
    """获取测试数据目录"""
    return config.get_path(config.TEST_DIR)


def get_model_dir():
    """获取模型目录"""
    return config.get_path(config.MODEL_DIR)


def get_checkpoint_dir():
    """获取检查点目录"""
    return config.get_path(config.CHECKPOINT_DIR)


def get_onnx_dir():
    """获取ONNX模型目录"""
    return config.get_path(config.ONNX_DIR)


def get_log_dir():
    """获取日志目录"""
    return config.get_path(config.LOG_DIR)


def get_docs_dir():
    """获取文档目录"""
    return config.get_path(config.DOCS_DIR)


def get_scripts_dir():
    """获取脚本目录"""
    return config.get_path(config.SCRIPTS_DIR)


def get_src_dir():
    """获取源代码目录"""
    return config.get_path(config.SRC_DIR)


def get_config_dir():
    """获取配置目录"""
    return config.get_path(config.CONFIG_DIR)


def get_tests_dir():
    """获取测试目录"""
    return config.get_path(config.TESTS_DIR)


def get_characters_dir():
    """获取角色列表目录"""
    return config.get_path(config.CHARACTERS_DIR)


def get_anime_set_file():
    """获取系列配置文件"""
    return config.get_path(config.ANIME_SET_FILE)


def get_character_file(series_name):
    """获取角色文件路径"""
    return config.get_character_file(series_name)


def get_batch_size():
    """获取批次大小"""
    return config.BATCH_SIZE


def get_epochs():
    """获取训练轮数"""
    return config.EPOCHS


def get_learning_rate():
    """获取学习率"""
    return config.LEARNING_RATE


def get_confidence_threshold():
    """获取置信度阈值"""
    return config.CONFIDENCE_THRESHOLD


def get_max_images_per_character():
    """获取每个角色最大图像数"""
    return config.MAX_IMAGES_PER_CHARACTER


def get_min_images_per_character():
    """获取每个角色最小图像数"""
    return config.MIN_IMAGES_PER_CHARACTER


def get_min_image_size():
    """获取最小图像尺寸"""
    return config.MIN_IMAGE_SIZE


def get_max_image_size():
    """获取最大图像尺寸"""
    return config.MAX_IMAGE_SIZE


def get_min_aspect_ratio():
    """获取最小宽高比"""
    return config.MIN_ASPECT_RATIO


def get_max_aspect_ratio():
    """获取最大宽高比"""
    return config.MAX_ASPECT_RATIO
