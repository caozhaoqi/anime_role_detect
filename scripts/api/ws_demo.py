#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket 实时进度监控 Demo
用于接收爬虫服务推送的采集进度
"""

import websocket
import json
import threading
import time
import os
from datetime import datetime


class SpiderProgressMonitor:
    """爬虫进度监控器"""
    
    def __init__(self, ws_url="ws://localhost:33333/api/v1.2.5.260305/sis/spider/progress/ws"):
        self.ws_url = ws_url
        self.ws = None
        self.running = False
        self.current_progress = {}
        self.lock = threading.Lock()
    
    def on_open(self, ws):
        """连接建立时的回调"""
        print("\n" + "=" * 70)
        print(f"✅ WebSocket 连接已建立")
        print(f"📡 连接地址: {self.ws_url}")
        print("=" * 70)
    
    def on_message(self, ws, message):
        """收到消息时的回调"""
        try:
            progress = json.loads(message)
            
            with self.lock:
                self.current_progress = progress
            
            # 格式化输出
            self._print_progress(progress)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}")
        except Exception as e:
            print(f"❌ 消息处理错误: {e}")
    
    def on_error(self, ws, error):
        """发生错误时的回调"""
        print(f"\n❌ WebSocket 错误: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """连接关闭时的回调"""
        print("\n" + "=" * 70)
        print(f"🔌 WebSocket 连接已关闭")
        if close_status_code:
            print(f"   关闭码: {close_status_code}")
        if close_msg:
            print(f"   关闭消息: {close_msg}")
        print("=" * 70)
    
    def _print_progress(self, progress):
        """格式化输出进度信息"""
        timestamp = datetime.fromtimestamp(progress.get('timestamp', time.time()))
        status_icon = {
            'running': '🔄',
            'completed': '✅',
            'error': '❌',
            'idle': '⏳'
        }.get(progress.get('status'), '❓')
        
        print(f"\n{timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"┌─────────────────────────────────────────────────────────────┐")
        print(f"│ {status_icon} 关键词: {progress.get('keyword', '未知'):^20} │")
        print(f"├─────────────────────────────────────────────────────────────┤")
        print(f"│ 状态: {progress.get('status', '未知'):^12} │ 页码: {progress.get('page', 0):^6} │")
        print(f"│ URL数量: {progress.get('current_count', 0):^8} │")
        print(f"├─────────────────────────────────────────────────────────────┤")
        print(f"│ 消息: {progress.get('message', '')[:45]:^45} │")
        print(f"└─────────────────────────────────────────────────────────────┘")
    
    def start(self):
        """启动监控"""
        print("🚀 正在连接爬虫进度监控服务...")
        
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        self.running = True
        
        # 在后台线程运行
        def run_ws():
            while self.running:
                try:
                    self.ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception as e:
                    print(f"⚠️ 连接断开，正在重连... ({e})")
                    time.sleep(5)
        
        ws_thread = threading.Thread(target=run_ws, daemon=True)
        ws_thread.start()
        
        print("⏳ 等待连接建立...")
        time.sleep(2)
        
        # 保持主线程运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """停止监控"""
        print("\n🛑 正在停止监控...")
        self.running = False
        if self.ws:
            self.ws.close()
        print("✅ 监控已停止")
    
    def get_progress(self):
        """获取当前进度"""
        with self.lock:
            return self.current_progress.copy()


def main():
    """主函数"""
    print("=" * 70)
    print("      Spider Image System - WebSocket 进度监控 Demo")
    print("=" * 70)
    print("说明: 此工具用于实时监控爬虫采集进度")
    print("提示: 先启动爬虫服务，再运行此脚本")
    print("按 Ctrl+C 退出")
    print("=" * 70)
    
    monitor = SpiderProgressMonitor()
    
    try:
        monitor.start()
    except KeyboardInterrupt:
        monitor.stop()


if __name__ == "__main__":
    main()
