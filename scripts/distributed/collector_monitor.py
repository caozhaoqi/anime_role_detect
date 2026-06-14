#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采集任务监控服务
- 自动跟踪采集脚本
- 检测长时间无输出时重新启动
- 发送飞书通知
- 汇总数据采集进展
"""

import os
import sys
import json
import time
import signal
import subprocess
import requests
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

PROJECT_ROOT = "/Users/caozhaoqi/PycharmProjects/anime_role_detect"
sys.path.insert(0, PROJECT_ROOT)

# 配置
DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
CONFIG_PATH = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/notification_config.json")
COLLECTOR_SCRIPT = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/spider_image_system/src/danbooru/multi_site_enhanced_collector.py"
KEYWORDS_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/auto_spider_img/keywords"
OUTPUT_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset"

# 监控配置
MAX_IDLE_SECONDS = 900  # 15分钟无输出则重启（考虑到多站点重试）
CHECK_INTERVAL = 60     # 检查间隔（秒）
MIN_PROGRESS_INTERVAL = 3600  # 1小时无进展则重启


class CollectorMonitor:
    """采集任务监控器"""
    
    def __init__(self):
        self.process = None
        self.last_output_time = None
        self.last_stats = None
        self.start_time = datetime.now()
        self.restart_count = 0
        self.last_progress_time = datetime.now()  # 上次有进展的时间
        self.last_image_count = 0  # 上次记录的图片数量
        
    def load_config(self):
        """加载飞书配置"""
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_feishu_token(self, config):
        """获取飞书访问令牌"""
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        headers = {"Content-Type": "application/json; charset=utf-8"}
        data = {
            "app_id": config["feishu"]["app_id"],
            "app_secret": config["feishu"]["app_secret"]
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            result = response.json()
            
            if result.get("code") == 0:
                return result.get("tenant_access_token")
        except Exception as e:
            print(f"获取飞书令牌失败: {e}")
        return None
    
    def send_feishu_message(self, config, message):
        """发送飞书消息"""
        token = self.get_feishu_token(config)
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
        
        try:
            response = requests.post(url, headers=headers, params=params, json=data, timeout=10)
            result = response.json()
            
            if result.get("code") == 0:
                print("飞书消息发送成功")
                return True
            else:
                print(f"飞书消息发送失败: {result}")
        except Exception as e:
            print(f"发送飞书消息失败: {e}")
        return False
    
    def get_dataset_stats(self):
        """获取数据集统计信息"""
        character_stats = defaultdict(lambda: {"jpg": 0, "png": 0})
        
        if not DATA_DIR.exists():
            return None
        
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
                              reverse=True)[:15]
        
        return {
            "total_chars": total_chars,
            "total_images": total_images,
            "total_jpg": total_jpg,
            "total_png": total_png,
            "distribution": distribution,
            "top_chars": sorted_chars,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def generate_progress_report(self, stats, status="采集中"):
        """生成进度报告"""
        if not stats:
            return "⚠️ 数据目录不存在"
        
        target_per_char = 100
        achieved_100 = stats['distribution']['100+']
        achieved_50 = stats['distribution']['100+'] + stats['distribution']['50-99']
        progress = (stats['total_images'] / (stats['total_chars'] * target_per_char)) * 100 if stats['total_chars'] > 0 else 0
        
        # 计算运行时长
        runtime = datetime.now() - self.start_time
        hours = int(runtime.total_seconds() // 3600)
        minutes = int((runtime.total_seconds() % 3600) // 60)
        
        report = f"""🔄 角色图片采集监控报告
⏰ 时间: {stats['timestamp']}
📊 状态: {status}

━━━━━━━━ 【总体统计】━━━━━━━━
• 角色目录数: {stats['total_chars']}
• 图片总数: {stats['total_images']:,}
• JPG: {stats['total_jpg']:,} ({stats['total_jpg']/max(stats['total_images'],1)*100:.1f}%)
• PNG: {stats['total_png']:,} ({stats['total_png']/max(stats['total_images'],1)*100:.1f}%)

━━━━━━━━ 【图片数分布】━━━━━━━━
• 100+张: {stats['distribution']['100+']} 个角色
• 50-99张: {stats['distribution']['50-99']} 个角色
• 30-49张: {stats['distribution']['30-49']} 个角色
• 10-29张: {stats['distribution']['10-29']} 个角色
• 0-9张: {stats['distribution']['0-9']} 个角色

━━━━━━━━ 【TOP15角色】━━━━━━━━
"""
        for i, (char_name, s) in enumerate(stats['top_chars'], 1):
            total = s["jpg"] + s["png"]
            bar = "█" * min(int(total / 5), 20)
            report += f"{i:2d}. {char_name}: {total:3d}张 {bar}\n"
        
        report += f"""
━━━━━━━━ 【目标进度】━━━━━━━━
• 目标: 每个角色 {target_per_char} 张
• 已达100张: {achieved_100} 个 ({achieved_100/max(stats['total_chars'],1)*100:.1f}%)
• 已达50张: {achieved_50} 个 ({achieved_50/max(stats['total_chars'],1)*100:.1f}%)
• 总体进度: {progress:.1f}%
• 重启次数: {self.restart_count}
• 运行时长: {hours}小时{minutes}分钟
"""
        
        return report
    
    def start_collector(self):
        """启动采集脚本"""
        print("正在启动采集脚本...")
        
        cmd = [
            "python3", COLLECTOR_SCRIPT,
            "--input-dir", KEYWORDS_DIR,
            "--output-dir", OUTPUT_DIR,
            "--target-count", "100"
        ]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid  # 创建新进程组
            )
            self.last_output_time = datetime.now()
            self.start_time = datetime.now()
            print(f"采集脚本已启动 (PID: {self.process.pid})")
            return True
        except Exception as e:
            print(f"启动采集脚本失败: {e}")
            return False
    
    def check_process(self):
        """检查进程状态"""
        if self.process is None:
            return "stopped"
        
        # 检查进程是否退出
        retcode = self.process.poll()
        if retcode is not None:
            return "exited"
        
        # 检查是否长时间无输出
        if self.last_output_time:
            idle_time = (datetime.now() - self.last_output_time).total_seconds()
            if idle_time > MAX_IDLE_SECONDS:
                return "idle"
        
        return "running"
    
    def restart_collector(self):
        """重启采集脚本"""
        print("正在重启采集脚本...")
        
        # 终止旧进程
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                time.sleep(2)
            except:
                pass
        
        self.restart_count += 1
        self.start_collector()
    
    def send_status_notification(self, status, stats=None):
        """发送状态通知"""
        try:
            config = self.load_config()
            
            if status == "started":
                message = f"🚀 采集任务已启动！\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n🔄 目标: 每个角色采集100张图片"
            elif status == "restarted":
                message = f"🔄 采集任务已自动重启\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n📊 重启次数: {self.restart_count}\n⚠️ 原因: 长时间无输出或无进展"
            elif status == "completed":
                message = f"✅ 采集任务已完成！\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                if stats:
                    message += f"\n📊 总图片数: {stats['total_images']:,}"
            else:
                if stats:
                    message = self.generate_progress_report(stats)
                else:
                    return
            
            self.send_feishu_message(config, message)
        except Exception as e:
            print(f"发送通知失败: {e}")
    
    def run(self):
        """运行监控循环"""
        print("=" * 60)
        print("采集任务监控服务启动")
        print("=" * 60)
        print(f"数据目录: {DATA_DIR}")
        print(f"采集脚本: {COLLECTOR_SCRIPT}")
        print(f"最大空闲时间: {MAX_IDLE_SECONDS}秒")
        print(f"检查间隔: {CHECK_INTERVAL}秒")
        print("=" * 60)
        
        # 启动采集脚本
        self.start_collector()
        self.send_status_notification("started")
        
        # 发送初始状态
        stats = self.get_dataset_stats()
        if stats:
            report = self.generate_progress_report(stats)
            print(f"\n{report}")
        
        print("\n开始监控...")
        
        # 监控循环
        last_stats_report = datetime.now()
        
        try:
            while True:
                # 检查进程状态
                status = self.check_process()
                
                if status == "exited":
                    print(f"采集进程已退出 (返回码: {self.process.returncode})")
                    stats = self.get_dataset_stats()
                    if stats and stats['total_images'] > 0:
                        # 有数据，发送完成通知
                        self.send_status_notification("completed", stats)
                        print("发送完成通知")
                    # 重新启动
                    self.restart_collector()
                    self.send_status_notification("restarted")
                
                elif status == "idle":
                    print(f"检测到长时间无输出 ({MAX_IDLE_SECONDS}秒)，正在重启...")
                    self.restart_collector()
                    self.send_status_notification("restarted")
                
                # 检查是否长时间无进展（图片数未增加）
                current_stats = self.get_dataset_stats()
                if current_stats and current_stats['total_images'] > 0:
                    if current_stats['total_images'] > self.last_image_count:
                        # 有进展，更新时间
                        self.last_progress_time = datetime.now()
                        self.last_image_count = current_stats['total_images']
                        print(f"📈 检测到进展: {current_stats['total_images']:,} 张图片")
                    else:
                        # 无进展，检查是否超时
                        progress_time = (datetime.now() - self.last_progress_time).total_seconds()
                        if progress_time > MIN_PROGRESS_INTERVAL:
                            print(f"⏳ 长时间无进展 ({progress_time:.0f}秒)，正在重启...")
                            self.restart_collector()
                            self.send_status_notification("restarted")
                
                # 定期发送进度报告（每30分钟）
                now = datetime.now()
                if (now - last_stats_report).total_seconds() >= 1800:  # 30分钟
                    stats = self.get_dataset_stats()
                    if stats:
                        report = self.generate_progress_report(stats)
                        print(f"\n{report}")
                        self.send_status_notification("progress", stats)
                    last_stats_report = now
                
                # 更新最后输出时间（如果有输出）
                if self.process and self.process.stdout:
                    import select
                    if select.select([self.process.stdout], [], [], 0)[0]:
                        line = self.process.stdout.readline()
                        if line:
                            self.last_output_time = datetime.now()
                
                time.sleep(CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n监控服务已停止")
            if self.process:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except:
                    pass
        except Exception as e:
            print(f"监控出错: {e}")


def main():
    monitor = CollectorMonitor()
    monitor.run()


if __name__ == "__main__":
    main()