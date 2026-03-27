#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块

统一管理项目的所有配置项，支持从配置文件加载配置，并提供默认配置作为 fallback。

配置文件格式为 YAML，默认路径为项目根目录下的 config.yaml。
"""

import os
import yaml
from pathlib import Path

class ConfigManager:
    """配置管理器
    
    负责加载、管理和提供项目的所有配置项。
    支持从 YAML 配置文件加载配置，并提供合理的默认值。
    """
    
    def __init__(self, config_file=None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，如果为 None，则使用默认路径
        """
        # 默认配置文件路径
        if config_file is None:
            project_root = self.get_project_root()
            config_file = os.path.join(project_root, "config.yaml")
        
        self.config_file = config_file
        self.config = self.load_config()
        
        # 记录配置文件路径
        print(f"使用配置文件: {self.config_file}")
        print(f"配置文件存在: {os.path.exists(self.config_file)}")
        
    def get_project_root(self):
        """
        获取项目根目录
        
        Returns:
            project_root: 项目根目录路径
        """
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        return project_root
    
    def load_config(self):
        """
        加载配置文件
        
        首先加载默认配置，然后如果配置文件存在，则加载并合并用户配置。
        
        Returns:
            config: 合并后的配置字典
        """
        # 默认配置
        default_config = {
            "project": {
                "root": self.get_project_root()
            },
            "api": {
                "host": "127.0.0.1",  # API服务绑定的主机
                "port": 8000,  # API服务绑定的端口
                "reload": False  # 是否开启热重载
            },
            "model": {
                "default_index": "role_index.faiss",  # 默认模型索引文件
                "models_dir": "models",  # 模型目录
                "model_mappings": {
                    "default": "role_index.faiss",
                    "augmented_training": "models/augmented_training/role_index.faiss",
                    "arona_plana": "models/arona_plana/role_index.faiss",
                    "arona_plana_efficientnet": "models/arona_plana_efficientnet/role_index.faiss",
                    "arona_plana_resnet18": "models/arona_plana_resnet18/role_index.faiss",
                    "optimized": "models/optimized/role_index.faiss"
                }
            },
            "cache": {
                "classifiers_max_size": 10,  # 分类器缓存的最大大小
                "feature_extraction": {
                    "quantize": True  # 是否量化特征提取器
                }
            },
            "file": {
                "max_size": 10 * 1024 * 1024,  # 10MB，最大文件大小
                "allowed_types": ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/svg+xml"]  # 允许的文件类型
            },
            "batch": {
                "max_files": 10  # 批量处理的最大文件数量
            }
        }
        
        # 如果配置文件存在，加载配置文件
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                # 合并配置
                self._merge_config(default_config, user_config)
                print("配置文件加载成功")
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                print("使用默认配置")
        else:
            print("配置文件不存在，使用默认配置")
        
        return default_config
    
    def _merge_config(self, default, user):
        """
        合并配置
        
        递归合并用户配置到默认配置中，用户配置会覆盖默认配置。
        
        Args:
            default: 默认配置字典
            user: 用户配置字典
        """
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                # 如果是嵌套字典，递归合并
                self._merge_config(default[key], value)
            else:
                # 否则，直接覆盖
                default[key] = value
    
    def get(self, key, default=None):
        """
        获取配置项
        
        支持使用点号分隔的路径来获取嵌套配置项。
        
        Args:
            key: 配置键，支持点号分隔的路径，如 "api.host"
            default: 如果配置项不存在，返回的默认值
        
        Returns:
            value: 配置值，如果不存在则返回默认值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_project_root(self):
        """
        获取项目根目录
        
        Returns:
            project_root: 项目根目录路径
        """
        # 如果config已初始化，从配置中获取
        if hasattr(self, 'config') and self.config:
            return self.get("project.root")
        # 否则，直接计算项目根目录
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        return project_root
    
    def get_model_path(self, model_name):
        """
        获取模型路径
        
        Args:
            model_name: 模型名称
        
        Returns:
            model_path: 模型路径
        """
        project_root = self.get_project_root()
        model_mappings = self.get("model.model_mappings", {})
        model_path = model_mappings.get(model_name, self.get("model.default_index"))
        
        # 如果是相对路径，拼接项目根目录
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)
        
        return model_path
    
    def get_classifiers_max_size(self):
        """
        获取分类器缓存最大大小
        
        Returns:
            max_size: 最大大小
        """
        return self.get("cache.classifiers_max_size", 10)
    
    def get_max_file_size(self):
        """
        获取最大文件大小
        
        Returns:
            max_size: 最大大小（字节）
        """
        return self.get("file.max_size", 10 * 1024 * 1024)
    
    def get_allowed_file_types(self):
        """
        获取允许的文件类型
        
        Returns:
            allowed_types: 允许的文件类型列表
        """
        return self.get("file.allowed_types", ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/svg+xml"])
    
    def get_max_batch_files(self):
        """
        获取批量处理的最大文件数量
        
        Returns:
            max_files: 最大文件数量
        """
        return self.get("batch.max_files", 10)
    
    def get_api_config(self):
        """
        获取API配置
        
        Returns:
            api_config: API配置字典
        """
        return self.get("api", {})

# 创建全局配置实例
config_manager = ConfigManager()