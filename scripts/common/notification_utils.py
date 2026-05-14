#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共通知工具模块
提供统一的通知接口，支持飞书、Telegram等多种通知渠道
"""

import os
import sys
import json
import time
import logging
import requests
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class NotificationConfig:
    """
    通知配置类
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.app_id: Optional[str] = None
        self.app_secret: Optional[str] = None
        self.receive_id: Optional[str] = None
        self.telegram_token: Optional[str] = None
        self.telegram_chat_id: Optional[str] = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """
        从配置文件加载通知配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                feishu_config = config.get('feishu', {})
                self.app_id = feishu_config.get('app_id')
                self.app_secret = feishu_config.get('app_secret')
                self.receive_id = feishu_config.get('receive_id')
                
                telegram_config = config.get('telegram', {})
                self.telegram_token = telegram_config.get('token')
                self.telegram_chat_id = telegram_config.get('chat_id')
                
                logger.info("通知配置加载成功")
            else:
                logger.warning(f"配置文件不存在: {config_path}")
        except Exception as e:
            logger.error(f"加载通知配置失败: {e}")


class FeishuNotifier:
    """
    飞书通知器
    """
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.app_id: Optional[str] = config.app_id if config else None
        self.app_secret: Optional[str] = config.app_secret if config else None
        self.receive_id: Optional[str] = config.receive_id if config else None
        self.access_token: Optional[str] = None
        self.token_expires: float = 0
    
    def _get_access_token(self) -> Optional[str]:
        """
        获取飞书Access Token
        
        Returns:
            Access Token字符串，失败返回None
        """
        # 检查token是否有效
        if self.access_token and time.time() < self.token_expires:
            return self.access_token
        
        # 检查配置是否完整
        if not self.app_id or not self.app_secret:
            logger.warning("飞书配置不完整，无法获取Access Token")
            return None
        
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"app_id": self.app_id, "app_secret": self.app_secret},
                timeout=10
            )
            result = response.json()
            
            if result.get("code") == 0:
                self.access_token = result.get("tenant_access_token")
                self.token_expires = time.time() + result.get("expire", 7200) - 300
                return self.access_token
            else:
                logger.error(f"获取飞书Access Token失败: {result.get('msg', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"获取飞书Access Token失败: {e}")
        
        return None
    
    def send_message(self, text: str) -> bool:
        """
        发送飞书消息
        
        Args:
            text: 消息内容
        
        Returns:
            True表示成功，False表示失败
        """
        if not self.receive_id:
            logger.warning("飞书接收ID未配置")
            return False
        
        access_token = self._get_access_token()
        if not access_token:
            return False
        
        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        params = {"receive_id_type": "chat_id"}
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        data = {
            "receive_id": self.receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": text})
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json=data, timeout=10)
            result = response.json()
            success = result.get("code") == 0
            
            if success:
                logger.debug("飞书消息发送成功")
            else:
                logger.error(f"飞书消息发送失败: {result.get('msg', 'Unknown error')}")
            
            return success
        
        except Exception as e:
            logger.error(f"发送飞书消息失败: {e}")
            return False


class TelegramNotifier:
    """
    Telegram通知器
    """
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.token: Optional[str] = config.telegram_token if config else None
        self.chat_id: Optional[str] = config.telegram_chat_id if config else None
    
    def send_message(self, text: str) -> bool:
        """
        发送Telegram消息
        
        Args:
            text: 消息内容
        
        Returns:
            True表示成功，False表示失败
        """
        if not self.token or not self.chat_id:
            logger.warning("Telegram配置不完整")
            return False
        
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        params = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            result = response.json()
            
            if result.get("ok"):
                logger.debug("Telegram消息发送成功")
                return True
            else:
                logger.error(f"Telegram消息发送失败: {result.get('description', 'Unknown error')}")
                return False
        
        except Exception as e:
            logger.error(f"发送Telegram消息失败: {e}")
            return False


class CompositeNotifier:
    """
    组合通知器
    支持同时发送到多个通知渠道
    """
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        self.notifiers = []
        
        if config:
            if config.app_id and config.app_secret and config.receive_id:
                self.notifiers.append(FeishuNotifier(config))
            
            if config.telegram_token and config.telegram_chat_id:
                self.notifiers.append(TelegramNotifier(config))
        
        # 如果没有配置任何通知渠道，添加一个空通知器
        if not self.notifiers:
            self.notifiers.append(NullNotifier())
    
    def send_message(self, text: str, level: str = "info") -> bool:
        """
        发送消息到所有配置的通知渠道
        
        Args:
            text: 消息内容
            level: 消息级别 (info, success, warning, error)
        
        Returns:
            True表示至少一个渠道发送成功
        """
        results = [notifier.send_message(text) for notifier in self.notifiers]
        return any(results)


class NullNotifier:
    """
    空通知器（默认）
    不发送任何消息，用于占位
    """
    
    def send_message(self, text: str) -> bool:
        """不发送消息，直接返回True"""
        return True


class ProgressNotifier:
    """
    进度通知器
    支持定时发送下载进度通知
    """
    
    def __init__(self, interval: int = 300):
        """
        Args:
            interval: 通知间隔（秒），默认5分钟
        """
        self.interval = interval
        self.last_send_time = time.time()
        self.role_stats: Dict[str, Dict[str, int]] = {}
        self.total_stats: Dict[str, int] = {"total_success": 0, "total_fail": 0}
        self.notifier = CompositeNotifier()
    
    def update_stats(self, role_stats: Dict[str, Dict[str, int]], total_stats: Dict[str, int]):
        """
        更新统计数据
        
        Args:
            role_stats: 各角色统计数据
            total_stats: 总统计数据
        """
        self.role_stats = role_stats.copy()
        self.total_stats = total_stats.copy()
    
    def check_and_send(self) -> bool:
        """
        检查是否需要发送通知
        
        Returns:
            True表示已发送，False表示未发送
        """
        current_time = time.time()
        
        if current_time - self.last_send_time >= self.interval:
            self.last_send_time = current_time
            return self.send_progress()
        
        return False
    
    def send_progress(self) -> bool:
        """
        发送进度通知
        
        Returns:
            True表示发送成功
        """
        current_time_str = datetime.now().strftime("%H:%M:%S")
        message = f"📥 数据采集中...\n时间: {current_time_str}\n\n"
        
        if self.role_stats:
            message += "📊 当前进度:\n"
            top_roles = list(self.role_stats.items())[:5]
            for role_name, stats in top_roles:
                message += f"  • {role_name}: {stats.get('success', 0)} 张\n"
            
            if len(self.role_stats) > 5:
                message += f"  ... 还有 {len(self.role_stats) - 5} 个角色\n"
            
            total_success = self.total_stats.get('total_success', 0)
            total_fail = self.total_stats.get('total_fail', 0)
            message += f"\n总计: 成功 {total_success} 张, 失败 {total_fail} 张"
        
        logger.info("发送定时进度通知...")
        success = self.notifier.send_message(message, level="info")
        
        if success:
            logger.info("定时进度通知发送成功")
        else:
            logger.warning("定时进度通知发送失败")
        
        return success
    
    def send_start(self, batch_id: str, batch_name: str, role_count: int) -> bool:
        """
        发送任务开始通知
        
        Args:
            batch_id: 批次ID
            batch_name: 批次名称
            role_count: 角色数量
        
        Returns:
            True表示发送成功
        """
        message = f"📥 开始数据采集\n批次: {batch_id} - {batch_name}\n角色数: {role_count}"
        return self.notifier.send_message(message, level="info")
    
    def send_complete(self, batch_id: str, batch_name: str, elapsed_str: str, 
                     success_count: int, fail_count: int, save_dir: str) -> bool:
        """
        发送任务完成通知
        
        Args:
            batch_id: 批次ID
            batch_name: 批次名称
            elapsed_str: 耗时字符串
            success_count: 成功数量
            fail_count: 失败数量
            save_dir: 保存目录
        
        Returns:
            True表示发送成功
        """
        message = (
            f"✅ 数据采集完成\n"
            f"批次: {batch_id} - {batch_name}\n"
            f"耗时: {elapsed_str}\n\n"
            f"📊 结果:\n"
            f"  成功: {success_count} 张\n"
            f"  失败: {fail_count} 张\n\n"
            f"📁 保存目录: {save_dir}"
        )
        return self.notifier.send_message(message, level="success")
    
    def send_error(self, stage: str, error_message: str) -> bool:
        """
        发送错误通知
        
        Args:
            stage: 出错阶段
            error_message: 错误消息
        
        Returns:
            True表示发送成功
        """
        message = f"❌ 数据采集失败\n阶段: {stage}\n错误: {error_message}"
        return self.notifier.send_message(message, level="error")


__all__ = [
    'NotificationConfig',
    'FeishuNotifier',
    'TelegramNotifier',
    'CompositeNotifier',
    'NullNotifier',
    'ProgressNotifier'
]
