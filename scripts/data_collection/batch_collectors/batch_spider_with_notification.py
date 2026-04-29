#!/usr/bin/env python3
"""
批量爬取缺失角色的二次元图片（带飞书通知）
"""
import requests
import time
import urllib.parse
import os
import sys
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# 加载飞书配置
config_path = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/notification_config.json'
with open(config_path, 'r', encoding='utf-8') as f:
    notification_config = json.load(f)

# 设置飞书通知环境变量
os.environ['NOTIFICATION_ENABLED'] = 'true'
os.environ['NOTIFICATION_PLATFORM'] = notification_config['platform']
os.environ['FEISHU_APP_ID'] = notification_config['feishu']['app_id']
os.environ['FEISHU_APP_SECRET'] = notification_config['feishu']['app_secret']
os.environ['FEISHU_RECEIVE_ID'] = notification_config['feishu']['receive_id']
os.environ['FEISHU_RECEIVE_ID_TYPE'] = notification_config['feishu']['receive_id_type']

from src.services.notification_service import get_notification_manager

# 缺失角色列表
missing_roles = [
    '纳西妲',
    '可莉',
    '蕾贝',
    '迪奥娜',
    '阿洛娜',
    '普拉娜',
    '希格雯',
    '瑶瑶'
]

# API基础URL
BASE_URL = 'http://localhost:33333/api/v1.2.5.260305/sis'

class SpiderNotifier:
    """爬取通知器"""
    
    def __init__(self):
        self.notification_manager = get_notification_manager()
        
    def send_start_notification(self, total_roles):
        """发送爬取开始通知"""
        message = f"""🚀 开始批量爬取二次元角色图片
        
**任务信息:**
- 待爬取角色数: {total_roles}
- 目标角色: {', '.join(missing_roles)}
- 模式: 无头模式(Headless)"""
        self.notification_manager.send(message, title="图片采集任务开始", level="info")
    
    def send_progress_notification(self, current, total, role_name, success, failed):
        """发送爬取进度通知"""
        progress = (current / total) * 100
        message = f"""📸 采集进度: {current}/{total} ({progress:.1f}%)
        
**当前角色:** {role_name}
- 成功: {success}
- 失败: {failed}"""
        self.notification_manager.send(message, title="采集进度更新", level="info")
    
    def send_complete_notification(self, results):
        """发送爬取完成通知"""
        total_success = sum(r['success'] for r in results)
        total_failed = sum(r['failed'] for r in results)
        success_roles = [r['role'] for r in results if r['success'] > 0]
        failed_roles = [r['role'] for r in results if r['success'] == 0]
        
        message = f"""✅ 批量爬取完成
        
**统计信息:**
- 总角色数: {len(results)}
- 成功角色: {len(success_roles)}
- 失败角色: {len(failed_roles)}
- 下载图片: {total_success} 张

**成功角色:** {', '.join(success_roles) if success_roles else '无'}
**失败角色:** {', '.join(failed_roles) if failed_roles else '无'}"""
        self.notification_manager.send(message, title="图片采集任务完成", level="success")
    
    def send_error_notification(self, role_name, error_message):
        """发送错误通知"""
        message = f"""❌ 爬取失败
        
角色: {role_name}
错误: {error_message}"""
        self.notification_manager.send(message, title="采集任务出错", level="error")

def spider_single_role(keyword):
    """爬取单个角色"""
    encoded_keyword = urllib.parse.quote(keyword)
    url = f'{BASE_URL}/spider_start/single?key_word={encoded_keyword}'
    
    try:
        response = requests.post(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                print(f"✅ 开始爬取角色: {keyword}")
                return True, None
            else:
                error_msg = data.get('msg', '未知错误')
                print(f"❌ 爬取失败 [{keyword}]: {error_msg}")
                return False, error_msg
        else:
            error_msg = f"HTTP错误 {response.status_code}"
            print(f"❌ 请求失败 [{keyword}]: {error_msg}")
            return False, error_msg
    except Exception as e:
        error_msg = str(e)
        print(f"❌ 爬取异常 [{keyword}]: {error_msg}")
        return False, error_msg

def download_images():
    """下载已爬取的图片"""
    url = f'{BASE_URL}/download_all_image/start/'
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                print("✅ 开始下载图片")
                return True
            else:
                print(f"❌ 下载失败: {data.get('msg')}")
                return False
        else:
            print(f"❌ 请求失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 下载异常: {e}")
        return False

def count_downloaded_images(role_name):
    """统计下载的图片数量"""
    import glob
    download_path = f"/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/src/run/data/downloaded_images/{role_name}"
    jpg_files = glob.glob(f"{download_path}/*.jpg")
    return len(jpg_files)

def main():
    print("=== 开始批量爬取缺失角色 (带飞书通知) ===")
    print(f"待爬取角色数: {len(missing_roles)}")
    
    # 初始化通知器
    notifier = SpiderNotifier()
    
    # 发送开始通知
    notifier.send_start_notification(len(missing_roles))
    
    results = []
    
    for i, role in enumerate(missing_roles):
        print(f"\n[{i+1}/{len(missing_roles)}] 处理角色: {role}")
        
        # 爬取角色图片URL
        success, error_msg = spider_single_role(role)
        
        if success:
            # 等待爬取完成
            time.sleep(20)
            
            # 下载图片
            download_success = download_images()
            
            if download_success:
                # 等待下载完成
                time.sleep(15)
                
                # 统计下载数量
                downloaded_count = count_downloaded_images(role)
                
                # 复制到数据集目录
                import subprocess
                os.makedirs(f"/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/loli_roles/{role}", exist_ok=True)
                subprocess.run([
                    'cp', 
                    f"/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/src/run/data/downloaded_images/{role}/*.jpg",
                    f"/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/loli_roles/{role}/"
                ], capture_output=True)
                
                results.append({
                    'role': role,
                    'success': downloaded_count,
                    'failed': 0
                })
                
                # 发送进度通知
                notifier.send_progress_notification(i+1, len(missing_roles), role, downloaded_count, 0)
            else:
                results.append({
                    'role': role,
                    'success': 0,
                    'failed': 1
                })
                notifier.send_progress_notification(i+1, len(missing_roles), role, 0, 1)
        else:
            results.append({
                'role': role,
                'success': 0,
                'failed': 1
            })
            notifier.send_error_notification(role, error_msg)
        
        # 间隔时间
        time.sleep(5)
    
    # 发送完成通知
    notifier.send_complete_notification(results)
    
    print("\n=== 批量爬取完成 ===")
    
    # 打印统计结果
    total_success = sum(r['success'] for r in results)
    print(f"总成功下载: {total_success} 张图片")

if __name__ == "__main__":
    main()
