#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
加载和管理配置文件
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: str = None):
        """初始化配置管理器"""
        if config_path is None:
            # 默认配置文件路径
            self.config_path = Path(__file__).parent / "config.json"
        else:
            self.config_path = Path(config_path)

        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> bool:
        """加载配置文件"""
        try:
            if not self.config_path.exists():
                # 创建默认配置
                self._create_default_config()

            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

            return True
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            # 加载失败时使用默认配置
            self._load_default_config()
            return False

    def _create_default_config(self):
        """创建默认配置文件"""
        default_config = {
            "backend": {
                "api": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "timeout": 30,
                    "max_request_size": "100MB",
                },
                "auth": {
                    "jwt_secret": "your-secret-key-here",
                    "jwt_expiry": 3600,
                    "refresh_expiry": 86400,
                    "enable_rate_limit": True,
                    "max_requests_per_minute": 100,
                },
                "model": {
                    "default_model": "clip",
                    "coreml_enabled": True,
                    "max_batch_size": 32,
                    "confidence_threshold": 0.7,
                    "enable_multi_role": True,
                    "max_roles_per_image": 5,
                },
                "ocr": {
                    "enabled": True,
                    "language": "ch_sim",
                    "min_confidence": 0.5,
                    "filter_special_chars": True,
                },
                "storage": {
                    "recognition_records_dir": "./recognition_records",
                    "max_records_per_user": 1000,
                    "enable_backup": True,
                    "backup_interval": 86400,
                },
                "logging": {
                    "level": "info",
                    "file": "./logs/app.log",
                    "rotation": "daily",
                    "retention": 7,
                },
                "performance": {
                    "enable_caching": True,
                    "cache_ttl": 3600,
                    "max_cache_size": "1GB",
                    "enable_gpu": False,
                    "num_workers": 4,
                },
            },
            "frontend": {
                "ui": {
                    "theme": "light",
                    "enable_dark_mode": True,
                    "animate_transitions": True,
                    "show_platform_info": True,
                },
                "features": {
                    "enable_model_selection": True,
                    "enable_coreml_switch": True,
                    "enable_attributes_switch": True,
                    "enable_multi_role_switch": True,
                    "enable_history_panel": True,
                    "enable_drag_drop": True,
                    "enable_copy_download": True,
                },
                "api": {
                    "base_url": "/api",
                    "timeout": 30000,
                    "retry_count": 3,
                    "retry_delay": 1000,
                },
                "messages": {
                    "welcome_message": "你好！我是动漫角色识别助手。请上传一张动漫角色图片，我将尝试识别出这个角色。",
                    "processing_message": "正在识别...",
                    "error_message": "识别过程中出现错误，请重试。",
                },
                "validation": {
                    "max_image_size": 10485760,
                    "allowed_formats": ["image/jpeg", "image/png", "image/gif", "image/webp"],
                    "min_image_dimension": 100,
                },
            },
            "data_collection": {
                "enabled": True,
                "max_images_per_role": 100,
                "min_image_resolution": [800, 800],
                "timeout": 15,
                "max_retries": 3,
                "delay": 0.5,
                "batch_size": 5,
            },
            "model_training": {
                "enabled": True,
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.0001,
                "enable_incremental": True,
                "validation_split": 0.2,
                "save_best_only": True,
            },
        }

        # 确保目录存在
        self.config_path.parent.mkdir(exist_ok=True)

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)

    def _load_default_config(self):
        """加载默认配置"""
        self.config = {
            "backend": {
                "api": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "timeout": 30,
                    "max_request_size": "100MB",
                },
                "auth": {
                    "jwt_secret": "your-secret-key-here",
                    "jwt_expiry": 3600,
                    "refresh_expiry": 86400,
                    "enable_rate_limit": True,
                    "max_requests_per_minute": 100,
                },
                "model": {
                    "default_model": "clip",
                    "coreml_enabled": True,
                    "max_batch_size": 32,
                    "confidence_threshold": 0.7,
                    "enable_multi_role": True,
                    "max_roles_per_image": 5,
                },
                "ocr": {
                    "enabled": True,
                    "language": "ch_sim",
                    "min_confidence": 0.5,
                    "filter_special_chars": True,
                },
                "storage": {
                    "recognition_records_dir": "./recognition_records",
                    "max_records_per_user": 1000,
                    "enable_backup": True,
                    "backup_interval": 86400,
                },
                "logging": {
                    "level": "info",
                    "file": "./logs/app.log",
                    "rotation": "daily",
                    "retention": 7,
                },
                "performance": {
                    "enable_caching": True,
                    "cache_ttl": 3600,
                    "max_cache_size": "1GB",
                    "enable_gpu": False,
                    "num_workers": 4,
                },
            },
            "frontend": {
                "ui": {
                    "theme": "light",
                    "enable_dark_mode": True,
                    "animate_transitions": True,
                    "show_platform_info": True,
                },
                "features": {
                    "enable_model_selection": True,
                    "enable_coreml_switch": True,
                    "enable_attributes_switch": True,
                    "enable_multi_role_switch": True,
                    "enable_history_panel": True,
                    "enable_drag_drop": True,
                    "enable_copy_download": True,
                },
                "api": {
                    "base_url": "/api",
                    "timeout": 30000,
                    "retry_count": 3,
                    "retry_delay": 1000,
                },
                "messages": {
                    "welcome_message": "你好！我是动漫角色识别助手。请上传一张动漫角色图片，我将尝试识别出这个角色。",
                    "processing_message": "正在识别...",
                    "error_message": "识别过程中出现错误，请重试。",
                },
                "validation": {
                    "max_image_size": 10485760,
                    "allowed_formats": ["image/jpeg", "image/png", "image/gif", "image/webp"],
                    "min_image_dimension": 100,
                },
            },
            "data_collection": {
                "enabled": True,
                "max_images_per_role": 100,
                "min_image_resolution": [800, 800],
                "timeout": 15,
                "max_retries": 3,
                "delay": 0.5,
                "batch_size": 5,
            },
            "model_training": {
                "enabled": True,
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.0001,
                "enable_incremental": True,
                "validation_split": 0.2,
                "save_best_only": True,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""
        keys = key.split(".")
        config = self.config

        # 遍历到倒数第二个键
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # 设置值
        config[keys[-1]] = value

        # 保存配置
        return self.save_config()

    def save_config(self) -> bool:
        """保存配置文件"""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            return False

    def get_backend_config(self) -> Dict[str, Any]:
        """获取后端配置"""
        return self.config.get("backend", {})

    def get_frontend_config(self) -> Dict[str, Any]:
        """获取前端配置"""
        return self.config.get("frontend", {})

    def get_data_collection_config(self) -> Dict[str, Any]:
        """获取数据采集配置"""
        return self.config.get("data_collection", {})

    def get_model_training_config(self) -> Dict[str, Any]:
        """获取模型训练配置"""
        return self.config.get("model_training", {})

    def reload(self) -> bool:
        """重新加载配置"""
        return self.load_config()


# 创建全局配置实例
config_manager = ConfigManager()
