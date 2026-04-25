#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

# 测试通知服务
try:
    from src.services.notification_service import send_notification
    print("通知服务导入成功")
    send_notification("测试消息", level="info")
    print("测试通知发送成功")
except ImportError as e:
    print(f"通知服务导入失败: {e}")
except Exception as e:
    print(f"通知发送失败: {e}")