#!/usr/bin/env python3
"""
配置管理类，集中管理所有配置项
"""
import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理类"""
    
    def __init__(self, config_file=None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        default_config = {
            # 网络配置
            'network': {
                'max_retries': 3,
                'backoff_factor': 0.5,
                'timeout': 15,
                'download_timeout': 30,
                'pool_connections': 10,
                'pool_maxsize': 10
            },
            
            # 存储配置
            'storage': {
                'output_dir': 'data/sdv50_train',
                'test_dir': 'data/test_sdv50',
                'temp_dir': 'data/temp',
                'file_extension': 'jpg',
                'quality': 95
            },
            
            # 采集配置
            'collection': {
                'max_workers': 5,
                'max_images_per_character': 500,
                'min_image_size': 300,
                'ranking_modes': ['daily', 'weekly', 'monthly'],
                'search_urls': [
                    'https://sd.vv50.de/search.php?word={}',
                    'https://sd.vv50.de/illustration?word={}',
                    'https://sd.vv50.de/ranking.php?word={}',
                    'https://sd.vv50.de/bookmark_new_illust.php?word={}'
                ]
            },
            
            # 数据源配置
            'data_sources': {
                'sdv50': {
                    'base_url': 'https://sd.vv50.de',
                    'ranking_url': 'https://sd.vv50.de/ranking.php?mode={}&content={}',
                    'enabled': True
                },
                'bing': {
                    'base_url': 'https://www.bing.com',
                    'search_url': 'https://www.bing.com/images/search?q={}&count=50',
                    'enabled': True
                }
            },
            
            # 日志配置
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'log_dir': 'logs'
            },
            
            # 并发配置
            'concurrency': {
                'max_workers': 5,
                'dynamic_adjustment': True,
                'min_workers': 2,
                'max_workers_limit': 10
            }
        }
        
        # 如果指定了配置文件，尝试加载
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # 合并配置
                self._merge_config(default_config, user_config)
                logger.info(f"配置文件加载成功: {self.config_file}")
            except Exception as e:
                logger.error(f"配置文件加载失败: {e}, 使用默认配置")
        
        return default_config
    
    def _merge_config(self, default, user):
        """合并配置"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def get(self, key, default=None):
        """
        获取配置项
        
        Args:
            key: 配置键，支持点号分隔的路径，如 'network.timeout'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key, value):
        """
        设置配置项
        
        Args:
            key: 配置键，支持点号分隔的路径
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.info(f"配置项更新: {key} = {value}")
    
    def save(self, output_file=None):
        """
        保存配置到文件
        
        Args:
            output_file: 输出文件路径，默认使用初始化时的配置文件
        """
        save_file = output_file or self.config_file
        if not save_file:
            logger.warning("未指定配置文件路径，无法保存")
            return
        
        try:
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            logger.info(f"配置保存成功: {save_file}")
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
    
    def get_all(self):
        """
        获取所有配置
        
        Returns:
            dict: 所有配置
        """
        return self.config


# 全局配置管理器实例
config_manager = ConfigManager()
