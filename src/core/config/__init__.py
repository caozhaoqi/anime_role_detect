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
                "workers": int(os.getenv("WORKERS", 1)),
            },
            "service": {
                "model": {
                    "name": "模型服务",
                    "port": 8000,
                    "script": "src/services/model_service/app.py",
                    "description": "AI模型推理服务",
                    "enabled": True,
                    "args": ["--host", "0.0.0.0", "--port", "8000"],
                    "env": {
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "PYTORCH_MPS_DISABLE": "1",
                        "KMP_DUPLICATE_LIB_OK": "TRUE"
                    }
                },
                "api": {
                    "name": "主API服务",
                    "port": 8001,
                    "script": "src/api/app.py",
                    "description": "主API服务，提供角色识别接口",
                    "enabled": True,
                    "env": {
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "KMP_DUPLICATE_LIB_OK": "TRUE",
                        "MKL_THREADING_LAYER": "GNU"
                    }
                },
                "gateway": {
                    "name": "API网关",
                    "port": 8080,
                    "script": "src/services/api_gateway/app.py",
                    "description": "统一API网关",
                    "enabled": True,
                },
                "multimedia": {
                    "name": "多媒体服务",
                    "port": 8002,
                    "script": "src/services/multimedia_service/multimedia_service_app.py",
                    "description": "整合图像搜索和视频识别功能",
                    "enabled": True,
                },
                "search": {
                    "name": "搜索服务",
                    "port": 8003,
                    "script": "src/services/search_service/app_queue.py",
                    "description": "图像搜索服务",
                    "enabled": True,
                },
                "search-worker": {
                    "name": "搜索工作进程",
                    "script": "src/run/sh/start_search_worker.sh",
                    "description": "搜索队列工作进程",
                    "enabled": True,
                },
                "inference-worker": {
                    "name": "推理工作进程",
                    "script": "src/services/inference_worker/worker.py",
                    "description": "模型推理工作进程",
                    "enabled": True,
                    "args": ["--num-workers", "2", "--model", "ViT-B/32"],
                    "env": {
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "KMP_DUPLICATE_LIB_OK": "TRUE",
                        "MKL_THREADING_LAYER": "GNU"
                    }
                },
                "monitor-dashboard": {
                    "name": "监控面板",
                    "script": "src/run/monitor/monitor_dashboard.py",
                    "description": "系统监控面板",
                    "enabled": True,
                },
                "frontend": {
                    "name": "前端服务",
                    "port": 3000,
                    "script": "npm run dev",
                    "description": "Next.js前端应用",
                    "enabled": True,
                    "directory": "src/frontend",
                    "env": {
                        "PORT": "3000",
                        "NODE_ENV": "development"
                    }
                },
                "log-viewer": {
                    "name": "日志查看器",
                    "script": "-m scripts.tools.log_viewer",
                    "description": "实时日志查看服务",
                    "enabled": True,
                },
                "health-check": {
                    "name": "健康检查",
                    "script": "scripts/monitoring/health_check.py",
                    "description": "服务健康检查",
                    "enabled": True,
                    "args": ["--daemon", "--interval", "60"],
                },
                "log-monitor": {
                    "name": "日志监控",
                    "script": "scripts/monitoring/log_monitor.py",
                    "description": "日志文件监控",
                    "enabled": True,
                    "args": ["--daemon", "--interval", "10"],
                },
                "resource-monitor": {
                    "name": "资源监控",
                    "script": "scripts/monitoring/resource_monitor.py",
                    "description": "系统资源监控",
                    "enabled": True,
                    "args": ["--daemon", "--interval", "30"],
                },
            },
            "data": {
                "dataset_path": os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data",
                    "merged_english_dataset",
                ),
                "index_path": os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data",
                    "faiss_index",
                ),
            },
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "format": "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            },
            "security": {
                "api_key": os.getenv("API_KEY", "your-secret-key"),
                "allowed_origins": ["*"],
            },
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值，支持点路径访问"""
        keys = key_path.split(".")
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """设置配置值"""
        keys = key_path.split(".")
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
