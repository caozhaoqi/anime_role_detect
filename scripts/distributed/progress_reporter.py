#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据采集进度报告服务
定时发送采集进度到飞书
"""

import os
import sys
import json
import time
import schedule
import requests
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = "/Users/caozhaoqi/PycharmProjects/anime_role_detect"
sys.path.insert(0, PROJECT_ROOT)

# 配置
DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
CONFIG_PATH = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/notification_config.json")


def load_config():
    """加载飞书配置"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feishu_token(config):
    """获取飞书访问令牌"""
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    headers = {"Content-Type": "application/json; charset=utf-8"}
    data = {
        "app_id": config["feishu"]["app_id"],
        "app_secret": config["feishu"]["app_secret"]
    }
    
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    
    if result.get("code") == 0:
        return result.get("tenant_access_token")
    else:
        print(f"获取飞书令牌失败: {result}")
        return None


def send_feishu_message(config, message):
    """发送飞书消息"""
    token = get_feishu_token(config)
    if not token:
        return False
    
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    params = {
        "receive_id_type": config["feishu"]["receive_id_type"]
    }
    
    data = {
        "receive_id": config["feishu"]["receive_id"],
        "msg_type": "text",
        "content": json.dumps({"text": message})
    }
    
    response = requests.post(url, headers=headers, params=params, json=data)
    result = response.json()
    
    if result.get("code") == 0:
        print(f"飞书消息发送成功")
        return True
    else:
        print(f"飞书消息发送失败: {result}")
        return False


def get_dataset_stats():
    """获取数据集统计信息"""
    character_stats = defaultdict(lambda: {"jpg": 0, "png": 0})
    
    for char_dir in DATA_DIR.iterdir():
        if char_dir.is_dir():
            char_name = char_dir.name
            for img_file in char_dir.iterdir():
                if img_file.is_file():
                    ext = img_file.suffix.lower()
                    if ext in [".jpg", ".jpeg"]:
                        character_stats[char_name]["jpg"] += 1
                    elif ext == ".png":
                        character_stats[char_name]["png"] += 1
    
    total_chars = len(character_stats)
    total_images = sum(s["jpg"] + s["png"] for s in character_stats.values())
    total_jpg = sum(s["jpg"] for s in character_stats.values())
    total_png = sum(s["png"] for s in character_stats.values())
    
    # 图片数分布
    distribution = defaultdict(int)
    for char_name, stats in character_stats.items():
        count = stats["jpg"] + stats["png"]
        if count >= 100:
            distribution["100+"] += 1
        elif count >= 50:
            distribution["50-99"] += 1
        elif count >= 30:
            distribution["30-49"] += 1
        elif count >= 10:
            distribution["10-29"] += 1
        else:
            distribution["0-9"] += 1
    
    # 图片数最多的角色
    sorted_chars = sorted(character_stats.items(), 
                          key=lambda x: x[1]["jpg"] + x[1]["png"], 
                          reverse=True)[:10]
    
    return {
        "total_chars": total_chars,
        "total_images": total_images,
        "total_jpg": total_jpg,
        "total_png": total_png,
        "distribution": distribution,
        "top_chars": sorted_chars,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def generate_report(stats):
    """生成进度报告"""
    report = f"""📊 数据采集进度报告
时间: {stats['timestamp']}

📈 总体统计:
• 角色目录数: {stats['total_chars']}
• 图片总数: {stats['total_images']}
• JPG: {stats['total_jpg']} ({stats['total_jpg']/stats['total_images']*100:.1f}%)
• PNG: {stats['total_png']} ({stats['total_png']/stats['total_images']*100:.1f}%)

📊 图片数分布:
• 100+张: {stats['distribution']['100+']} 个角色
• 50-99张: {stats['distribution']['50-99']} 个角色
• 30-49张: {stats['distribution']['30-49']} 个角色
• 10-29张: {stats['distribution']['10-29']} 个角色
• 0-9张: {stats['distribution']['0-9']} 个角色

🏆 图片数最多的角色:
"""
    for i, (char_name, s) in enumerate(stats['top_chars'], 1):
        total = s["jpg"] + s["png"]
        report += f"  {i}. {char_name}: {total}张\n"
    
    # 计算进度
    target_per_char = 100
    achieved_100 = stats['distribution']['100+']
    achieved_50 = stats['distribution']['100+'] + stats['distribution']['50-99']
    progress = (stats['total_images'] / (stats['total_chars'] * target_per_char)) * 100
    
    report += f"""
🎯 目标进度:
• 目标: 每个角色 {target_per_char} 张
• 已达100张: {achieved_100} 个角色 ({achieved_100/stats['total_chars']*100:.1f}%)
• 已达50张: {achieved_50} 个角色 ({achieved_50/stats['total_chars']*100:.1f}%)
• 总体进度: {progress:.1f}%
"""
    
    return report


def send_progress_report():
    """发送进度报告"""
    try:
        config = load_config()
        stats = get_dataset_stats()
        report = generate_report(stats)
        
        print(f"\n{report}")
        send_feishu_message(config, report)
        
    except Exception as e:
        print(f"发送进度报告失败: {e}")


def main():
    """主函数"""
    print("启动数据采集进度报告服务...")
    print(f"数据目录: {DATA_DIR}")
    print(f"报告间隔: 每30分钟")
    
    # 立即发送一次报告
    send_progress_report()
    
    # 定时发送报告（每30分钟）
    schedule.every(30).minutes.do(send_progress_report)
    
    # 保持运行
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()