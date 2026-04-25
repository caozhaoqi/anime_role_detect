#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞书消息推送测试脚本

飞书机器人文档: https://open.feishu.cn/document/client-docs/bot-v3/bot-overview

机器人类型:
1. 应用机器人 - 需要在开放平台创建应用，配置权限，发布应用
2. 自定义机器人 - 在群聊中添加，配置简单，通过Webhook发送消息

推荐使用「自定义机器人」，配置最简单:
1. 在飞书群中添加「自定义机器人」
2. 获取Webhook地址
3. 配置环境变量即可使用
"""

import os
import sys
import json
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    print("=" * 70)
    print("🔔 飞书消息推送测试")
    print("=" * 70)
    print()

def test_webhook_direct(webhook_url):
    """直接测试Webhook (不通过notification_service)"""
    print("\n" + "-" * 70)
    print("直接测试飞书自定义机器人 Webhook")
    print("-" * 70)

    if not webhook_url:
        print("❌ 未提供 Webhook URL")
        return False

    print(f"Webhook URL: {webhook_url[:60]}...")

    # 飞书自定义机器人消息格式
    payload = {
        "msg_type": "text",
        "content": {
            "text": "🔔 测试消息\n这是一条来自自定义机器人的测试通知"
        }
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        result = response.json()

        print(f"\n响应状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}")

        if response.status_code == 200 and result.get("code") == 0:
            print("\n✅ Webhook消息发送成功!")
            return True
        else:
            print(f"\n❌ Webhook消息发送失败")
            return False
    except Exception as e:
        print(f"\n❌ 请求异常: {e}")
        return False

def test_feishu_notification_service():
    """通过通知服务测试"""
    print("\n" + "-" * 70)
    print("通过 notification_service 测试")
    print("-" * 70)

    from src.services.notification_service import send_notification, NotificationManager

    # 检查配置
    manager = NotificationManager()
    print(f"通知服务配置:")
    print(f"  启用状态: {manager.enabled}")
    print(f"  平台: {manager.platform}")
    print(f"  飞书Webhook: {'已配置' if manager.feishu_webhook else '未配置'}")
    print(f"  飞书App ID: {'已配置' if manager.feishu_app_id else '未配置'}")

    if not manager.enabled:
        print("\n⚠️ 通知服务未启用，请设置 NOTIFICATION_ENABLED=true")

    print("\n发送测试通知...")
    result = send_notification("🔔 通知服务测试\n通过notification_service发送的消息", level="info")
    print(f"发送结果: {result}")

    return result

def main():
    print_banner()

    # 检查是否提供了Webhook URL作为命令行参数
    webhook_url = ""
    if len(sys.argv) > 1:
        webhook_url = sys.argv[1]
    else:
        webhook_url = os.environ.get('FEISHU_WEBHOOK_URL', '')

    print("\n📋 配置说明:")
    print("-" * 70)
    print("推荐方式 - 自定义机器人 Webhook:")
    print("  1. 在飞书群设置中添加「自定义机器人」")
    print("  2. 设置机器人名称（建议：训练通知）")
    print("  3. 复制 Webhook 地址")
    print("  4. 设置环境变量并测试:")
    print()
    print("     export NOTIFICATION_ENABLED=true")
    print("     export NOTIFICATION_PLATFORM=feishu")
    print("     export FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxx")
    print()
    print("     python3 test_feishu_notification.py")
    print()
    print("-" * 70)

    # 直接测试Webhook
    if webhook_url:
        print("\n🚀 直接测试Webhook...")
        test_webhook_direct(webhook_url)

    # 通过通知服务测试
    print("\n🚀 通过通知服务测试...")
    test_feishu_notification_service()

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    print()
    print("📖 详细文档:")
    print("  飞书机器人概述: https://open.feishu.cn/document/client-docs/bot-v3/bot-overview")
    print("  自定义机器人: https://open.feishu.cn/document/client-docs/bot-v3/bot/create")
    print("  发送消息: https://open.feishu.cn/document/server-docs/im-v1/message-content-description/create_json")

if __name__ == "__main__":
    main()