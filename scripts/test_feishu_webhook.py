#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试飞书机器人webhook
"""

import requests
import json

def test_feishu_webhook():
    """测试飞书机器人webhook"""
    print("=== 测试飞书机器人webhook ===")
    
    # 这里需要替换为实际的飞书机器人webhook URL
    # 获取方式：飞书群聊 -> 群设置 -> 群机器人 -> 添加机器人 -> 自定义机器人 -> 复制webhook URL
    webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
    
    if "xxx" in webhook_url:
        print("❌ 请先设置飞书机器人webhook URL")
        print("获取方式：飞书群聊 -> 群设置 -> 群机器人 -> 添加机器人 -> 自定义机器人 -> 复制webhook URL")
        return
    
    # 测试消息
    content = "🚀 测试消息：飞书机器人webhook正常运行！"
    
    # 构建请求数据
    data = {
        "msg_type": "text",
        "content": {
            "text": content
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"发送测试消息到webhook: {webhook_url}")
    print(f"消息内容: {content}")
    
    try:
        response = requests.post(webhook_url, headers=headers, json=data, timeout=10)
        result = response.json()
        print(f"响应状态码: {response.status_code}")
        print(f"响应结果: {result}")
        
        if result.get("code") == 0:
            print("✅ 飞书机器人webhook消息发送成功！")
        else:
            print("❌ 飞书机器人webhook消息发送失败")
            
    except Exception as e:
        print(f"❌ 发送异常: {e}")

if __name__ == "__main__":
    test_feishu_webhook()
