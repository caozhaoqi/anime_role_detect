#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试飞书通知功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.notification_service import NotificationManager

def test_feishu_notification():
    """测试飞书通知"""
    print("=== 测试飞书通知 ===")
    
    # 检查环境变量
    print("环境变量配置:")
    print(f"NOTIFICATION_ENABLED: {os.environ.get('NOTIFICATION_ENABLED')}")
    print(f"NOTIFICATION_PLATFORM: {os.environ.get('NOTIFICATION_PLATFORM')}")
    print(f"FEISHU_APP_ID: {os.environ.get('FEISHU_APP_ID')}")
    print(f"FEISHU_APP_SECRET: {os.environ.get('FEISHU_APP_SECRET')}")
    print(f"FEISHU_RECEIVE_ID: {os.environ.get('FEISHU_RECEIVE_ID')}")
    print(f"FEISHU_RECEIVE_ID_TYPE: {os.environ.get('FEISHU_RECEIVE_ID_TYPE')}")
    
    # 初始化通知管理器
    notification_manager = NotificationManager()
    
    # 测试发送消息
    test_message = "🚀 测试消息：飞书通知服务正常运行！"
    print(f"\n发送测试消息: {test_message}")
    
    # 先测试获取token
    print("\n测试获取飞书Access Token...")
    token = notification_manager._get_feishu_access_token()
    if token:
        print(f"✅ 获取token成功: {token[:20]}...")
    else:
        print("❌ 获取token失败")
    
    # 测试发送消息
    print("\n测试发送飞书消息...")
    result = notification_manager.send_feishu_message(test_message)
    print(f"发送结果: {result}")
    
    if result:
        print("✅ 飞书通知发送成功！")
    else:
        print("❌ 飞书通知发送失败，请检查配置")

if __name__ == "__main__":
    test_feishu_notification()
