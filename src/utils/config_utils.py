#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向后兼容模块 — 已合并到 src.config

请直接从统一配置模块导入:
    from src.config import project_config
    from src.config import TrainingConfig, InferenceConfig, ...

此文件保留仅为兼容旧代码，将在后续版本移除。
"""

from src.config import (
    ProjectPaths,
    TrainingConfig,
    InferenceConfig,
    ImageConfig,
    project_config as _project_config,
)

# --- 旧式便利函数（兼容 from src.utils.config_utils import get_*） ---

config = _project_config

get_config         = lambda: _project_config
get_data_dir       = lambda: _project_config.get_path(ProjectPaths.DATA_DIR)
get_train_dir      = lambda: _project_config.get_path(ProjectPaths.TRAIN_DIR)
get_val_dir        = lambda: _project_config.get_path(ProjectPaths.VAL_DIR)
get_test_dir       = lambda: _project_config.get_path(ProjectPaths.TEST_DIR)
get_model_dir      = lambda: _project_config.get_path(ProjectPaths.MODEL_DIR)
get_checkpoint_dir = lambda: _project_config.get_path(ProjectPaths.CHECKPOINT_DIR)
get_onnx_dir       = lambda: _project_config.get_path(ProjectPaths.ONNX_DIR)
get_log_dir        = lambda: _project_config.get_path(ProjectPaths.LOG_DIR)
get_docs_dir       = lambda: _project_config.get_path(ProjectPaths.DOCS_DIR)
get_scripts_dir    = lambda: _project_config.get_path(ProjectPaths.SCRIPTS_DIR)
get_src_dir        = lambda: _project_config.get_path(ProjectPaths.SRC_DIR)
get_config_dir     = lambda: _project_config.get_path(ProjectPaths.CONFIG_DIR)
get_tests_dir      = lambda: _project_config.get_path(ProjectPaths.TESTS_DIR)
get_characters_dir = lambda: _project_config.get_path(ProjectPaths.CHARACTERS_DIR)
get_anime_set_file = lambda: _project_config.get_path(ProjectPaths.ANIME_SET_FILE)
get_character_file = lambda name: _project_config.get_character_file(name)

get_batch_size               = lambda: TrainingConfig.BATCH_SIZE
get_epochs                   = lambda: TrainingConfig.EPOCHS
get_learning_rate            = lambda: TrainingConfig.LEARNING_RATE
get_confidence_threshold     = lambda: InferenceConfig.CONFIDENCE_THRESHOLD
get_max_images_per_character = lambda: ImageConfig.MAX_IMAGES_PER_CHARACTER
get_min_images_per_character = lambda: ImageConfig.MIN_IMAGES_PER_CHARACTER
get_min_image_size           = lambda: ImageConfig.MIN_IMAGE_SIZE
get_max_image_size           = lambda: ImageConfig.MAX_IMAGE_SIZE
get_min_aspect_ratio         = lambda: ImageConfig.MIN_ASPECT_RATIO
get_max_aspect_ratio         = lambda: ImageConfig.MAX_ASPECT_RATIO

__all__ = [
    "config", "get_config",
    "get_data_dir", "get_train_dir", "get_val_dir", "get_test_dir",
    "get_model_dir", "get_checkpoint_dir", "get_onnx_dir",
    "get_log_dir", "get_docs_dir", "get_scripts_dir", "get_src_dir",
    "get_config_dir", "get_tests_dir", "get_characters_dir",
    "get_anime_set_file", "get_character_file",
    "get_batch_size", "get_epochs", "get_learning_rate",
    "get_confidence_threshold", "get_max_images_per_character",
    "get_min_images_per_character", "get_min_image_size",
    "get_max_image_size", "get_min_aspect_ratio", "get_max_aspect_ratio",
]
