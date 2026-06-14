#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞书通知服务
"""
import json
import requests
import os
from pathlib import Path


class FeishuNotification:
    """飞书通知类"""

    def __init__(self, config_path=None):
        """初始化"""
        if config_path is None:
            # 从脚本目录加载配置
            config_path = Path(__file__).parent.parent.parent / "scripts" / "notification_config.json"
        
        self.config_path = config_path
        self.config = self._load_config()
        self.token = None

    def _load_config(self):
        """加载配置"""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _get_token(self):
        """获取飞书token"""
        if self.token:
            return self.token

        feishu_config = self.config.get("feishu", {})
        app_id = feishu_config.get("app_id")
        app_secret = feishu_config.get("app_secret")

        if not app_id or not app_secret:
            print("⚠️ 飞书配置不完整")
            return None

        try:
            url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
            data = {"app_id": app_id, "app_secret": app_secret}
            response = requests.post(url, json=data)
            response.raise_for_status()
            self.token = response.json().get("tenant_access_token")
            return self.token
        except Exception as e:
            print(f"❌ 获取飞书token失败: {e}")
            return None

    def send_message(self, title, message):
        """发送飞书消息"""
        token = self._get_token()
        if not token:
            return False

        feishu_config = self.config.get("feishu", {})
        receive_id = feishu_config.get("receive_id")
        receive_id_type = feishu_config.get("receive_id_type", "chat_id")

        if not receive_id:
            print("⚠️ 缺少receive_id配置")
            return False

        try:
            # 使用正确的飞书消息API
            url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={receive_id_type}"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            data = {
                "receive_id": receive_id,
                "content": json.dumps({"text": f"**{title}**\n\n{message}"}),
                "msg_type": "text"
            }
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            print(f"✅ 飞书通知发送成功")
            return True
        except Exception as e:
            print(f"❌ 飞书通知发送失败: {e}")
            return False


# 单例实例
_notifier = None


def get_notifier():
    """获取通知器实例"""
    global _notifier
    if _notifier is None:
        _notifier = FeishuNotification()
    return _notifier


def send_notification(title, message):
    """发送通知"""
    notifier = get_notifier()
    return notifier.send_message(title, message)


if __name__ == "__main__":
    # 测试通知
    send_notification("测试通知", "这是一条测试消息")