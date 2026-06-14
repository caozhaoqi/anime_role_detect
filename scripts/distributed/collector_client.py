#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式采集系统 - 客户端
定时轮询服务端，下载并解压采集数据
"""

import os
import sys
import json
import time
import uuid
import zipfile
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin

# 配置
DEFAULT_SERVER = "http://localhost:5000"
LOCAL_DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/downloaded")
CHECK_INTERVAL = 300  # 5分钟检查一次


class CollectorClient:
    """采集客户端"""

    def __init__(self, server_url=DEFAULT_SERVER, client_id=None):
        self.server_url = server_url.rstrip('/')
        self.local_dir = LOCAL_DATA_DIR
        self.local_dir.mkdir(parents=True, exist_ok=True)

        # 客户端ID（用于服务端识别避免重复下载）
        self.client_id_file = self.local_dir / "client_id.txt"
        self.client_id = client_id or self._load_or_create_client_id()

        # 记录已下载的包（本地备份）
        self.downloaded_file = self.local_dir / "downloaded_packages.json"
        self.downloaded = self._load_downloaded()

        # 飞书通知配置
        self.feishu_config = None

    def _load_or_create_client_id(self):
        """加载或创建客户端ID"""
        if self.client_id_file.exists():
            with open(self.client_id_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            # 创建新的客户端ID
            client_id = f"client_{uuid.uuid4().hex[:12]}"
            with open(self.client_id_file, 'w', encoding='utf-8') as f:
                f.write(client_id)
            return client_id

    def _load_downloaded(self):
        """加载已下载记录"""
        if self.downloaded_file.exists():
            with open(self.downloaded_file, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        return set()

    def _save_downloaded(self):
        """保存已下载记录"""
        with open(self.downloaded_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.downloaded), f, ensure_ascii=False, indent=2)

    def _load_feishu_config(self):
        """加载飞书配置"""
        config_path = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/notification_config.json")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.feishu_config = json.load(f).get("feishu")

    def _get_feishu_token(self):
        """获取飞书访问令牌"""
        if not self.feishu_config:
            return None

        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        headers = {"Content-Type": "application/json; charset=utf-8"}
        data = {
            "app_id": self.feishu_config["app_id"],
            "app_secret": self.feishu_config["app_secret"]
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            result = response.json()
            if result.get("code") == 0:
                return result.get("tenant_access_token")
        except Exception as e:
            print(f"获取飞书令牌失败: {e}")
        return None

    def _send_feishu(self, message):
        """发送飞书消息"""
        if not self.feishu_config:
            return False

        token = self._get_feishu_token()
        if not token:
            return False

        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        params = {"receive_id_type": self.feishu_config["receive_id_type"]}
        data = {
            "receive_id": self.feishu_config["receive_id"],
            "msg_type": "text",
            "content": json.dumps({"text": message})
        }

        try:
            response = requests.post(url, headers=headers, params=params, json=data, timeout=10)
            return response.json().get("code") == 0
        except Exception as e:
            print(f"发送飞书消息失败: {e}")
        return False

    def check_health(self):
        """检查服务端健康状态"""
        try:
            url = f"{self.server_url}/api/health"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True, response.json()
        except Exception as e:
            return False, str(e)
        return False, None

    def get_status(self):
        """获取服务端状态"""
        try:
            url = f"{self.server_url}/api/status"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"获取状态失败: {e}")
        return None

    def get_latest_package(self):
        """获取最新数据包信息"""
        try:
            url = f"{self.server_url}/api/package/latest"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json().get("package")
        except Exception as e:
            print(f"获取最新包失败: {e}")
        return None

    def get_package_list(self):
        """获取数据包列表"""
        try:
            url = f"{self.server_url}/api/package/list"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json().get("packages", [])
        except Exception as e:
            print(f"获取包列表失败: {e}")
        return []

    def download_package(self, package_name):
        """下载数据包"""
        if package_name in self.downloaded:
            print(f"已下载过 {package_name}，跳过")
            return False

        try:
            url = f"{self.server_url}/api/package/{package_name}"
            print(f"正在下载: {package_name} (client_id: {self.client_id})")

            # 添加 client_id 参数避免重复下载
            headers = {"X-Client-ID": self.client_id}
            params = {"client_id": self.client_id}

            response = requests.get(url, timeout=300, stream=True, headers=headers, params=params)

            # 检查是否已下载过（服务端返回409）
            if response.status_code == 409:
                print(f"服务端记录显示已下载过 {package_name}")
                # 同步本地记录
                self.downloaded.add(package_name)
                self._save_downloaded()
                return False

            if response.status_code != 200:
                print(f"下载失败: HTTP {response.status_code}")
                return False

            # 保存到临时文件
            temp_file = self.local_dir / f"temp_{package_name}"
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # 解压
            print(f"正在解压: {package_name}")
            self._extract_package(temp_file)

            # 删除临时文件
            temp_file.unlink()

            # 记录已下载
            self.downloaded.add(package_name)
            self._save_downloaded()

            print(f"完成: {package_name}")
            return True

        except Exception as e:
            print(f"下载失败: {e}")
            return False

    def _extract_package(self, zip_path):
        """解压数据包"""
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.local_dir)

    def sync_data(self):
        """同步数据"""
        print("=" * 60)
        print(f"开始同步 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # 检查服务端状态
        healthy, result = self.check_health()
        if not healthy:
            print(f"服务端不可用: {result}")
            return False

        # 获取最新包
        latest = self.get_latest_package()
        if not latest:
            print("暂无新数据包")
            return False

        print(f"最新包: {latest['name']}")
        print(f"创建时间: {latest['created_at']}")
        print(f"大小: {latest['size'] / 1024 / 1024:.2f} MB")

        stats = latest.get("stats", {})
        print(f"角色数: {stats.get('total_chars', 0)}")
        print(f"图片数: {stats.get('total_images', 0)}")

        # 下载
        if latest["name"] not in self.downloaded:
            success = self.download_package(latest["name"])
            if success:
                # 发送通知
                message = f"""✅ 数据同步完成
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📦 包名: {latest['name']}
📊 角色: {stats.get('total_chars', 0)}
🖼️ 图片: {stats.get('total_images', 0)}
💾 大小: {latest['size'] / 1024 / 1024:.2f} MB
📁 保存至: {self.local_dir}"""
                self._send_feishu(message)
                return True
        else:
            print("已是最新数据，无需下载")

        return False

    def run(self, interval=CHECK_INTERVAL, once=False):
        """运行客户端"""
        print("=" * 60)
        print("分布式采集系统 - 客户端")
        print("=" * 60)
        print(f"服务端: {self.server_url}")
        print(f"本地目录: {self.local_dir}")
        print(f"客户端ID: {self.client_id}")
        print(f"检查间隔: {interval}秒")
        print(f"已下载: {len(self.downloaded)} 个包")
        print("=" * 60)

        # 加载飞书配置
        self._load_feishu_config()

        if once:
            # 单次执行
            self.sync_data()
        else:
            # 循环执行
            print("\n开始轮询...")
            while True:
                try:
                    self.sync_data()
                    print(f"\n等待 {interval} 秒后再次检查...")
                    time.sleep(interval)
                except KeyboardInterrupt:
                    print("\n客户端已停止")
                    break
                except Exception as e:
                    print(f"错误: {e}")
                    time.sleep(60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='分布式采集客户端')
    parser.add_argument('--server', type=str, default=DEFAULT_SERVER,
                        help='服务端地址')
    parser.add_argument('--interval', type=int, default=CHECK_INTERVAL,
                        help=f'检查间隔(秒)，默认{CHECK_INTERVAL}')
    parser.add_argument('--once', action='store_true',
                        help='只执行一次')
    parser.add_argument('--local-dir', type=str,
                        default=str(LOCAL_DATA_DIR),
                        help='本地数据目录')
    parser.add_argument('--client-id', type=str,
                        help='客户端ID（可选，默认自动生成）')

    args = parser.parse_args()

    client = CollectorClient(server_url=args.server, client_id=args.client_id)
    client.local_dir = Path(args.local_dir)
    client.local_dir.mkdir(parents=True, exist_ok=True)

    client.run(interval=args.interval, once=args.once)


if __name__ == "__main__":
    main()
