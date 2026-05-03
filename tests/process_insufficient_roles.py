#!/usr/bin/env python3
"""处理下载不足的角色"""
import os
import sys
import time
import json
import requests
import sqlite3
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "data" / "role_images.db"
CONFIG_PATH = PROJECT_ROOT / "scripts" / "notification_config.json"

class FeishuNotifier:
    """飞书通知器"""
    
    def __init__(self):
        self.app_id = None
        self.app_secret = None
        self.receive_id = None
        self.access_token = None
        self.token_expires = 0
        self._load_config()
    
    def _load_config(self):
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.app_id = config.get('feishu', {}).get('app_id')
                    self.app_secret = config.get('feishu', {}).get('app_secret')
                    self.receive_id = config.get('feishu', {}).get('receive_id')
        except Exception as e:
            print(f"加载飞书配置失败: {e}")
    
    def _get_access_token(self):
        if self.access_token and time.time() < self.token_expires:
            return self.access_token
        if not self.app_id or not self.app_secret:
            return None
        try:
            response = requests.post(
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                json={"app_id": self.app_id, "app_secret": self.app_secret},
                timeout=10
            )
            result = response.json()
            if result.get("code") == 0:
                self.access_token = result.get("tenant_access_token")
                self.token_expires = time.time() + result.get("expire", 7200) - 300
                return self.access_token
        except Exception as e:
            print(f"获取飞书Token失败: {e}")
        return None
    
    def send_message(self, text):
        if not self.receive_id:
            return False
        access_token = self._get_access_token()
        if not access_token:
            return False
        try:
            response = requests.post(
                "https://open.feishu.cn/open-apis/im/v1/messages",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"},
                params={"receive_id_type": "chat_id"},
                json={"receive_id": self.receive_id, "msg_type": "text", "content": json.dumps({"text": text})},
                timeout=10
            )
            return response.json().get("code") == 0
        except Exception as e:
            print(f"发送飞书消息失败: {e}")
            return False

def get_insufficient_roles(threshold=200, progress_threshold=50):
    """获取下载不足的角色"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # 获取待下载数量
        cursor.execute('SELECT role_name, COUNT(1) FROM raw_urls WHERE status = "pending" GROUP BY role_name')
        pending = {row[0]: row[1] for row in cursor.fetchall()}

        # 获取已下载数量
        cursor.execute('SELECT role_name, COUNT(1) FROM downloaded_images WHERE status = "success" GROUP BY role_name')
        downloaded = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        result = []
        all_roles = set(pending.keys()) | set(downloaded.keys())
        
        for role in all_roles:
            p = pending.get(role, 0)
            d = downloaded.get(role, 0)
            total = p + d
            if total == 0:
                continue
            progress = (d / total) * 100
            
            if total < threshold or progress < progress_threshold:
                result.append({
                    'role': role,
                    'pending': p,
                    'downloaded': d,
                    'total': total,
                    'progress': progress
                })

        return sorted(result, key=lambda x: (x['progress'], x['total']))
    except Exception as e:
        print(f"获取不足角色失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def trigger_spider(role_name):
    """触发角色采集"""
    try:
        response = requests.post(
            "http://localhost:5000/sis/spider_start",
            json={"keyword": role_name, "limit": 300},
            timeout=30
        )
        return response.status_code == 200
    except Exception as e:
        print(f"触发采集失败 {role_name}: {e}")
        return False

def main():
    print("=" * 80)
    print(" 🚀 处理下载不足角色")
    print(f" 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    notifier = FeishuNotifier()

    # 获取下载不足的角色
    insufficient = get_insufficient_roles(threshold=200, progress_threshold=50)
    
    if not insufficient:
        print("✅ 没有发现下载不足的角色")
        notifier.send_message("✅ 没有发现下载不足的角色")
        return

    print(f"\n📊 发现 {len(insufficient)} 个下载不足的角色")

    # 发送通知
    msg = f"""🚀 **开始处理下载不足的角色**

发现 {len(insufficient)} 个需要处理的角色:

"""
    for i, role_info in enumerate(insufficient[:10], 1):
        msg += f"{i}. {role_info['role']}: {role_info['downloaded']}/{role_info['total']} ({role_info['progress']:.1f}%)\n"
    
    if len(insufficient) > 10:
        msg += f"... 还有 {len(insufficient) - 10} 个角色"
    
    notifier.send_message(msg)

    # 处理逻辑
    for role_info in insufficient:
        role = role_info['role']
        total = role_info['total']
        downloaded = role_info['downloaded']
        progress = role_info['progress']
        
        print(f"\n🔄 处理角色: {role}")
        print(f"   当前状态: {downloaded}/{total} ({progress:.1f}%)")

        # 策略1: 如果URL总数太少，先重新采集
        if total < 100:
            print(f"   ⚠️ URL数量不足，触发重新采集...")
            if trigger_spider(role):
                print(f"   ✅ 采集任务已触发")
                notifier.send_message(f"🔄 正在重新采集角色: {role}")
                time.sleep(5)  # 等待采集启动
        else:
            print(f"   ℹ️ URL数量充足，等待下载任务处理")

    print("\n" + "=" * 80)
    print(" 📋 处理完成")
    print("=" * 80)

    notifier.send_message(f"✅ 下载不足角色处理完成，共 {len(insufficient)} 个角色")

if __name__ == '__main__':
    main()
