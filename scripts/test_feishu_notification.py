#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试飞书通知功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from src.services.notification_service import get_notification_manager, send_notification

def test_feishu_notification():
    """测试飞书通知"""
    print("=== 测试飞书通知 ===")
    
    # 检查环境变量
    print("环境变量配置:")
    print(f"NOTIFICATION_ENABLED: {os.environ.get('NOTIFICATION_ENABLED')}")
    print(f"NOTIFICATION_PLATFORM: {os.environ.get('NOTIFICATION_PLATFORM')}")
    print(f"FEISHU_APP_ID: {os.environ.get('FEISHU_APP_ID')}")
    print(f"FEISHU_APP_SECRET: {os.environ.get('FEISHU_APP_SECRET')}")
    print(f"FEISHU_WEBHOOK_URL: {os.environ.get('FEISHU_WEBHOOK_URL')}")
    print(f"FEISHU_RECEIVE_ID: {os.environ.get('FEISHU_RECEIVE_ID')}")
    
    # 初始化通知管理器
    notification_manager = get_notification_manager()
    
    # 打印通知管理器状态
    print(f"\n通知管理器状态:")
    print(f"  enabled: {notification_manager.enabled}")
    print(f"  platform: {notification_manager.platform}")
    print(f"  feishu_webhook: {notification_manager.feishu_webhook}")
    print(f"  feishu_app_id: {notification_manager.feishu_app_id}")
    print(f"  feishu_receive_id: {notification_manager.feishu_receive_id}")
    
    # 测试发送消息
    test_message = "🚀 测试消息：飞书通知服务正常运行！"
    print(f"\n发送测试消息: {test_message}")
    
    # 测试发送通知
    print("\n测试发送飞书消息...")
    result = send_notification(test_message, "飞书通知测试", level="success")
    print(f"发送结果: {result}")
    
    if result:
        print("✅ 飞书通知发送成功！")
    else:
        print("❌ 飞书通知发送失败，请检查配置")

if __name__ == "__main__":
    test_feishu_notification()