#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心配置管理模块 
"""

import os
import sys
from typing import Dict, Any

class ConfigManager:
    """统一配置管理器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = {}
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """加载配置文件"""
        # 基础配置
        self._config = {
            "server": {
                "host": os.getenv("HOST", "0.0.0.0"),
                "port": int(os.getenv("PORT", 8000)),
                "workers": int(os.getenv("WORKERS", 1))
            },
            "service": {
                "multimedia": {
                    "name": "多媒体服务",
                    "port": 8002,
                    "script": "services/multimedia_service/multimedia_service_app.py",
                    "description": "整合图像搜索和视频识别功能",
                    "enabled": True
                },
                "model": {
                    "name": "模型服务",
                    "port": 8000,
                    "script": "services/model_service/app_simple.py",
                    "description": "AI模型推理服务",
                    "enabled": True
                },
                "api": {
                    "name": "主API服务",
                    "port": 8001,
                    "script": "api/run_api.py",
                    "description": "主API服务，提供角色识别接口",
                    "enabled": True
                },
                "gateway": {
                    "name": "API网关",
                    "port": 8080,
                    "script": "services/api_gateway/app.py",
                    "description": "统一API网关",
                    "enabled": True
                }
            },
            "data": {
                "dataset_path": os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "merged_english_dataset"
                ),
                "index_path": os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "faiss_index"
                )
            },
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "format": "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
            },
            "security": {
                "api_key": os.getenv("API_KEY", "your-secret-key"),
                "allowed_origins": ["*"]
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值，支持点路径访问"""
        keys = key_path.split('.')
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """设置配置值"""
        keys = key_path.split('.')
        config = self._config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """获取服务配置"""
        return self.get(f"service.{service_name}", {})
    
    def get_all_services(self) -> Dict[str, Any]:
        """获取所有服务配置"""
        return self.get("service", {})

# 全局配置实例
config = ConfigManager()

def get_config():
    """获取配置实例"""
    return config