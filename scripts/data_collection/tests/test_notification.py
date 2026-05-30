# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 测试数据采集通知功能
# """

# import os
# import sys
# import time
# import json

# # 正确设置项目根目录
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # 测试通知服务
# try:
#     from src.services.notification_service import (
#         send_notification,
#         send_training_progress_notification,
#         send_training_complete_notification,
#         send_training_error_notification
#     )
#     print("通知服务导入成功")
# except ImportError as e:
#     print(f"通知服务导入失败: {e}")
#     sys.exit(1)

# # 测试数据采集通知
# print("=" * 60)
# print("测试数据采集通知功能")
# print("=" * 60)

# # 测试1: 开始采集通知
# print("\n1. 测试开始采集通知")
# send_notification(
#     "📥 开始数据采集\n批次: 1 - 原神核心角色\n角色数: 10",
#     level="info"
# )

# # 测试2: 进度通知
# print("\n2. 测试进度通知")
# send_training_progress_notification(
#     stage="数据采集",
#     progress=30.5,
#     message="正在下载: 派蒙",
#     metrics={'total_processed': 150, 'success': 120, 'fail': 30}
# )

# # 测试3: 完成通知
# print("\n3. 测试完成通知")
# send_training_complete_notification(
#     model_name="数据采集批次1",
#     metrics={'total_images': 500, 'success': 450, 'fail': 50},
#     model_path="/path/to/images",
#     training_time=1200  # 20分钟
# )

# # 测试4: 错误通知
# print("\n4. 测试错误通知")
# send_training_error_notification(
#     stage="数据采集",
#     error_message="网络连接超时，请检查网络设置"
# )

# print("\n" + "=" * 60)
# print("测试完成")
# print("=" * 60)
# print("请查看企业微信/飞书通知是否收到")
# print("注意: 通知功能需要配置相应的环境变量")
# print("\n配置示例:")
# print("  export NOTIFICATION_ENABLED=true")
# print("  export NOTIFICATION_PLATFORM=wecom  # wecom, feishu, wxpusher, dingtalk")
# print("  export WECOM_CORP_ID=your_corp_id")
# print("  export WECOM_AGENT_ID=your_agent_id")
# print("  export WECOM_SECRET=your_secret")
# print("  # 或飞书配置")
# print("  export FEISHU_APP_ID=your_app_id")
# print("  export FEISHU_APP_SECRET=your_app_secret")
