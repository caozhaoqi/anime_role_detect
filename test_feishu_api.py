#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试飞书消息推送

根据飞书官方文档: https://open.feishu.cn/document/server-docs/im-v1/message/create?appId=cli_a966711bc1785cd3

示例请求格式:
{
  "receive_id": "oc_b376c0f5a01eef8f6240b1f3f7b249d2",
  "msg_type": "text",
  "content": "{\"text\":\"test content\"}",
  "uuid": "选填，每次调用前请更换"
}
"""

import os
import sys
import json
import requests
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_tenant_access_token(app_id, app_secret):
    """获取飞书 tenant_access_token"""
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    headers = {"Content-Type": "application/json"}
    data = {
        "app_id": app_id,
        "app_secret": app_secret
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        result = response.json()
        
        if result.get("code") == 0:
            return result.get("tenant_access_token")
        else:
            print(f"获取 access_token 失败: {result}")
            return None
    except Exception as e:
        print(f"请求异常: {e}")
        return None

def send_message(access_token, receive_id, receive_id_type, msg_type, content):
    """发送消息"""
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "receive_id": receive_id,
        "msg_type": msg_type,
        "content": content,
        "uuid": str(uuid.uuid4())
    }
    
    # 添加查询参数
    params = {"receive_id_type": receive_id_type}
    
    try:
        response = requests.post(url, headers=headers, params=params, json=data, timeout=10)
        result = response.json()
        
        print(f"\n发送消息响应:")
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        if result.get("code") == 0:
            print("\n✅ 消息发送成功!")
            return True
        else:
            print("\n❌ 消息发送失败")
            return False
    except Exception as e:
        print(f"\n❌ 请求异常: {e}")
        return False

def main():
    print("=" * 70)
    print("🔔 飞书消息推送测试")
    print("=" * 70)
    print()
    
    # 从环境变量获取配置
    app_id = os.environ.get('FEISHU_APP_ID', '')
    app_secret = os.environ.get('FEISHU_APP_SECRET', '')
    
    # 如果环境变量未设置，从 .env 文件读取
    if not app_id or not app_secret:
        try:
            with open('.env', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('app_id ='):
                        app_id = line.split('=')[1].strip().strip("'\"")
                    elif line.startswith('app_secret ='):
                        app_secret = line.split('=')[1].strip().strip("'\"")
                    elif line.strip() == 'cli_a966711bc1785cd3':
                        app_id = 'cli_a966711bc1785cd3'
                    elif line.strip() == 'ob6KCeeVB9KrbzQo5WA2BEX4byfmLAHg':
                        app_secret = 'ob6KCeeVB9KrbzQo5WA2BEX4byfmLAHg'
        except Exception as e:
            print(f"读取 .env 文件失败: {e}")
    
    # 检查配置
    if not app_id:
        app_id = input("请输入飞书 App ID: ").strip()
    if not app_secret:
        app_secret = input("请输入飞书 App Secret: ").strip()
    
    receive_id = "oc_b376c0f5a01eef8f6240b1f3f7b249d2"
    receive_id_type = "chat_id"  # 因为 receive_id 是 oc_ 开头，说明是群聊ID
    msg_type = "text"
    content = "{\"text\":\"测试消息：这是一条来自飞书开放平台 API 的测试通知\"}"
    
    print()
    print("📋 配置信息:")
    print("-" * 70)
    print(f"App ID: {app_id}")
    print(f"App Secret: {'*' * len(app_secret)}")
    print(f"Receive ID: {receive_id}")
    print(f"Receive ID Type: {receive_id_type}")
    print(f"Message Type: {msg_type}")
    print(f"Content: {content}")
    print()
    
    # 获取 access_token
    print("🚀 获取 access_token...")
    access_token = get_tenant_access_token(app_id, app_secret)
    
    if not access_token:
        print("❌ 获取 access_token 失败")
        return
    
    print(f"✅ 获得 access_token: {access_token[:20]}...")
    
    # 发送消息
    print("\n🚀 发送测试消息...")
    send_message(access_token, receive_id, receive_id_type, msg_type, content)
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    print()
    print("📖 注意事项:")
    print("  1. 应用需要开启机器人能力并发布")
    print("  2. 应用需要添加 im:message:send 权限")
    print("  3. 机器人需要在该群聊中")
    print("  4. 群聊 ID (receive_id) 格式为 oc_ 开头")
    print()
    print("🌐 官方文档:")
    print("  https://open.feishu.cn/document/server-docs/im-v1/message/create?appId=cli_a966711bc1785cd3")

if __name__ == "__main__":
    main()