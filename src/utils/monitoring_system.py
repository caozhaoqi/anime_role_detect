#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控和管理系统

实现实时监控、告警和可视化功能
"""

import os
import time
import logging
import threading
import json
import psutil
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class Monitor:
    """
    监控器基类
    """
    
    def __init__(self, name, interval=5):
        """
        初始化监控器
        
        Args:
            name: 监控器名称
            interval: 监控间隔（秒）
        """
        self.name = name
        self.interval = interval
        self.data = deque(maxlen=100)  # 最多保存100个数据点
        self.running = False
        self.thread = None
    
    def start(self):
        """
        启动监控器
        """
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"启动监控器: {self.name}")
    
    def stop(self):
        """
        停止监控器
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info(f"停止监控器: {self.name}")
    
    def _run(self):
        """
        运行监控器
        """
        while self.running:
            try:
                data = self.collect_data()
                self.data.append(data)
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"监控器错误 {self.name}: {e}")
                time.sleep(self.interval)
    
    @staticmethod
    def collect_data():
        """
        收集监控数据
        
        Returns:
            监控数据
        """
        pass
    
    def get_data(self, limit=10):
        """
        获取监控数据
        
        Args:
            limit: 数据点数量
            
        Returns:
            监控数据列表
        """
        return list(self.data)[-limit:]
    
    def get_stats(self):
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        pass


class SystemMonitor(Monitor):
    """
    系统监控器
    """
    
    def __init__(self, interval=5):
        """
        初始化系统监控器
        
        Args:
            interval: 监控间隔（秒）
        """
        super().__init__(name='system', interval=interval)
    
    def collect_data(self):
        """
        收集系统监控数据
        
        Returns:
            系统监控数据
        """
        try:
            # 获取CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 获取内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used / (1024 * 1024 * 1024)  # GB
            memory_total = memory.total / (1024 * 1024 * 1024)  # GB
            
            # 获取磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used = disk.used / (1024 * 1024 * 1024)  # GB
            disk_total = disk.total / (1024 * 1024 * 1024)  # GB
            
            # 获取网络使用情况
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent / (1024 * 1024)  # MB
            net_recv = net_io.bytes_recv / (1024 * 1024)  # MB
            
            # 获取进程数量
            process_count = len(psutil.pids())
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used': memory_used,
                'memory_total': memory_total,
                'disk_percent': disk_percent,
                'disk_used': disk_used,
                'disk_total': disk_total,
                'net_sent': net_sent,
                'net_recv': net_recv,
                'process_count': process_count
            }
        except Exception as e:
            logger.error(f"收集系统数据失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': 0,
                'memory_percent': 0,
                'memory_used': 0,
                'memory_total': 0,
                'disk_percent': 0,
                'disk_used': 0,
                'disk_total': 0,
                'net_sent': 0,
                'net_recv': 0,
                'process_count': 0
            }
    
    def get_stats(self):
        """
        获取系统统计信息
        
        Returns:
            系统统计信息
        """
        if not self.data:
            return {}
        
        # 计算平均值
        cpu_values = [d['cpu_percent'] for d in self.data]
        memory_values = [d['memory_percent'] for d in self.data]
        disk_values = [d['disk_percent'] for d in self.data]
        
        return {
            'name': self.name,
            'average_cpu': sum(cpu_values) / len(cpu_values),
            'average_memory': sum(memory_values) / len(memory_values),
            'average_disk': sum(disk_values) / len(disk_values),
            'max_cpu': max(cpu_values),
            'max_memory': max(memory_values),
            'max_disk': max(disk_values),
            'min_cpu': min(cpu_values),
            'min_memory': min(memory_values),
            'min_disk': min(disk_values),
            'data_points': len(self.data)
        }


class NetworkMonitor(Monitor):
    """
    网络监控器
    """
    
    def __init__(self, interval=10):
        """
        初始化网络监控器
        
        Args:
            interval: 监控间隔（秒）
        """
        super().__init__(name='network', interval=interval)
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0,
            'response_times': []
        }
    
    def collect_data(self):
        """
        收集网络监控数据
        
        Returns:
            网络监控数据
        """
        try:
            # 获取网络连接数
            connections = psutil.net_connections()
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            
            # 获取网络接口信息
            net_if_addrs = psutil.net_if_addrs()
            interface_count = len(net_if_addrs)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'active_connections': active_connections,
                'interface_count': interface_count,
                'request_stats': self.request_stats.copy()
            }
        except Exception as e:
            logger.error(f"收集网络数据失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'active_connections': 0,
                'interface_count': 0,
                'request_stats': self.request_stats.copy()
            }
    
    def update_request_stats(self, success, response_time=0):
        """
        更新请求统计信息
        
        Args:
            success: 是否成功
            response_time: 响应时间
        """
        self.request_stats['total_requests'] += 1
        if success:
            self.request_stats['successful_requests'] += 1
            self.request_stats['response_times'].append(response_time)
            # 只保留最近100个响应时间
            if len(self.request_stats['response_times']) > 100:
                self.request_stats['response_times'] = self.request_stats['response_times'][-100:]
            # 更新平均响应时间
            self.request_stats['avg_response_time'] = sum(self.request_stats['response_times']) / len(self.request_stats['response_times'])
        else:
            self.request_stats['failed_requests'] += 1
    
    def get_stats(self):
        """
        获取网络统计信息
        
        Returns:
            网络统计信息
        """
        if not self.data:
            return {}
        
        return {
            'name': self.name,
            'request_stats': self.request_stats,
            'success_rate': self.request_stats['successful_requests'] / self.request_stats['total_requests'] if self.request_stats['total_requests'] > 0 else 0,
            'data_points': len(self.data)
        }


class TaskMonitor(Monitor):
    """
    任务监控器
    """
    
    def __init__(self, task_queue=None, interval=5):
        """
        初始化任务监控器
        
        Args:
            task_queue: 任务队列
            interval: 监控间隔（秒）
        """
        super().__init__(name='task', interval=interval)
        self.task_queue = task_queue
        self.task_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'running_tasks': 0,
            'pending_tasks': 0,
            'task_times': []
        }
    
    def collect_data(self):
        """
        收集任务监控数据
        
        Returns:
            任务监控数据
        """
        try:
            queue_size = 0
            if self.task_queue:
                queue_size = self.task_queue.get_queue_size()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'queue_size': queue_size,
                'task_stats': self.task_stats.copy()
            }
        except Exception as e:
            logger.error(f"收集任务数据失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'queue_size': 0,
                'task_stats': self.task_stats.copy()
            }
    
    def update_task_stats(self, task_status, task_time=0):
        """
        更新任务统计信息
        
        Args:
            task_status: 任务状态
            task_time: 任务执行时间
        """
        if task_status == 'created':
            self.task_stats['total_tasks'] += 1
            self.task_stats['pending_tasks'] += 1
        elif task_status == 'running':
            self.task_stats['pending_tasks'] -= 1
            self.task_stats['running_tasks'] += 1
        elif task_status == 'completed':
            self.task_stats['running_tasks'] -= 1
            self.task_stats['completed_tasks'] += 1
            if task_time > 0:
                self.task_stats['task_times'].append(task_time)
                # 只保留最近100个任务时间
                if len(self.task_stats['task_times']) > 100:
                    self.task_stats['task_times'] = self.task_stats['task_times'][-100:]
        elif task_status == 'failed':
            self.task_stats['running_tasks'] -= 1
            self.task_stats['failed_tasks'] += 1
    
    def get_stats(self):
        """
        获取任务统计信息
        
        Returns:
            任务统计信息
        """
        if not self.data:
            return {}
        
        avg_task_time = 0
        if self.task_stats['task_times']:
            avg_task_time = sum(self.task_stats['task_times']) / len(self.task_stats['task_times'])
        
        success_rate = 0
        if self.task_stats['total_tasks'] > 0:
            success_rate = self.task_stats['completed_tasks'] / self.task_stats['total_tasks']
        
        return {
            'name': self.name,
            'total_tasks': self.task_stats['total_tasks'],
            'completed_tasks': self.task_stats['completed_tasks'],
            'failed_tasks': self.task_stats['failed_tasks'],
            'running_tasks': self.task_stats['running_tasks'],
            'pending_tasks': self.task_stats['pending_tasks'],
            'success_rate': success_rate,
            'average_task_time': avg_task_time,
            'data_points': len(self.data)
        }


class AlertManager:
    """
    告警管理器
    """
    
    def __init__(self, config=None):
        """
        初始化告警管理器
        
        Args:
            config: 告警配置
        """
        self.config = config or {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'username': 'user@example.com',
                'password': 'password',
                'from_email': 'alert@example.com',
                'to_emails': ['admin@example.com']
            },
            'thresholds': {
                'cpu': 80,
                'memory': 80,
                'disk': 80,
                'network_error_rate': 10,
                'task_failure_rate': 10
            },
            'cooldown_period': 300  # 告警冷却期（秒）
        }
        
        self.alerts = []
        self.last_alert_time = {}
        self.lock = threading.Lock()
    
    def check_thresholds(self, stats):
        """
        检查阈值
        
        Args:
            stats: 监控统计信息
            
        Returns:
            告警列表
        """
        alerts = []
        
        # 检查系统阈值
        if 'system' in stats:
            system_stats = stats['system']
            if system_stats.get('average_cpu', 0) > self.config['thresholds']['cpu']:
                alerts.append({
                    'level': 'warning',
                    'message': f'CPU使用率过高: {system_stats.get("average_cpu", 0):.2f}%',
                    'type': 'cpu'
                })
            
            if system_stats.get('average_memory', 0) > self.config['thresholds']['memory']:
                alerts.append({
                    'level': 'warning',
                    'message': f'内存使用率过高: {system_stats.get("average_memory", 0):.2f}%',
                    'type': 'memory'
                })
            
            if system_stats.get('average_disk', 0) > self.config['thresholds']['disk']:
                alerts.append({
                    'level': 'warning',
                    'message': f'磁盘使用率过高: {system_stats.get("average_disk", 0):.2f}%',
                    'type': 'disk'
                })
        
        # 检查网络阈值
        if 'network' in stats:
            network_stats = stats['network']
            request_stats = network_stats.get('request_stats', {})
            total_requests = request_stats.get('total_requests', 1)
            failed_requests = request_stats.get('failed_requests', 0)
            error_rate = (failed_requests / total_requests) * 100
            
            if error_rate > self.config['thresholds']['network_error_rate']:
                alerts.append({
                    'level': 'warning',
                    'message': f'网络错误率过高: {error_rate:.2f}%',
                    'type': 'network'
                })
        
        # 检查任务阈值
        if 'task' in stats:
            task_stats = stats['task']
            total_tasks = task_stats.get('total_tasks', 1)
            failed_tasks = task_stats.get('failed_tasks', 0)
            failure_rate = (failed_tasks / total_tasks) * 100
            
            if failure_rate > self.config['thresholds']['task_failure_rate']:
                alerts.append({
                    'level': 'warning',
                    'message': f'任务失败率过高: {failure_rate:.2f}%',
                    'type': 'task'
                })
        
        return alerts
    
    def add_alert(self, alert):
        """
        添加告警
        
        Args:
            alert: 告警信息
        """
        with self.lock:
            # 检查冷却期
            alert_type = alert.get('type', 'unknown')
            current_time = time.time()
            
            if alert_type in self.last_alert_time:
                if current_time - self.last_alert_time[alert_type] < self.config['cooldown_period']:
                    logger.info(f"告警冷却中，跳过: {alert['message']}")
                    return
            
            # 添加告警
            alert['timestamp'] = datetime.now().isoformat()
            self.alerts.append(alert)
            self.last_alert_time[alert_type] = current_time
            
            logger.warning(f"告警: {alert['message']}")
            
            # 发送告警通知
            self.send_alert(alert)
    
    def send_alert(self, alert):
        """
        发送告警通知
        
        Args:
            alert: 告警信息
        """
        # 发送邮件告警
        if self.config['email']['enabled']:
            self._send_email_alert(alert)
        
        # 这里可以添加其他告警方式，比如短信、微信等
    
    def _send_email_alert(self, alert):
        """
        发送邮件告警
        
        Args:
            alert: 告警信息
        """
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['from_email']
            msg['To'] = ', '.join(self.config['email']['to_emails'])
            msg['Subject'] = f"[{alert['level'].upper()}] 系统告警: {alert['message']}"
            
            # 邮件内容
            body = f"""
            告警时间: {alert['timestamp']}
            告警级别: {alert['level']}
            告警类型: {alert.get('type', 'unknown')}
            告警消息: {alert['message']}
            
            请及时处理！
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # 发送邮件
            server = smtplib.SMTP(
                self.config['email']['smtp_server'],
                self.config['email']['smtp_port']
            )
            server.starttls()
            server.login(
                self.config['email']['username'],
                self.config['email']['password']
            )
            server.send_message(msg)
            server.quit()
            
            logger.info(f"邮件告警已发送: {alert['message']}")
        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")
    
    def get_alerts(self, limit=10):
        """
        获取告警
        
        Args:
            limit: 告警数量
            
        Returns:
            告警列表
        """
        with self.lock:
            return self.alerts[-limit:]


class MonitoringSystem:
    """
    监控系统
    """
    
    def __init__(self, config=None):
        """
        初始化监控系统
        
        Args:
            config: 监控配置
        """
        self.config = config or {
            'system_monitor_interval': 5,
            'network_monitor_interval': 10,
            'task_monitor_interval': 5
        }
        
        # 创建监控器
        self.monitors = {
            'system': SystemMonitor(interval=self.config['system_monitor_interval']),
            'network': NetworkMonitor(interval=self.config['network_monitor_interval']),
            'task': TaskMonitor(interval=self.config['task_monitor_interval'])
        }
        
        # 创建告警管理器
        self.alert_manager = AlertManager()
        
        # 启动监控线程
        self.running = False
        self.monitoring_thread = None
        self.alert_thread = None
    
    def start(self):
        """
        启动监控系统
        """
        self.running = True
        
        # 启动所有监控器
        for monitor in self.monitors.values():
            monitor.start()
        
        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # 启动告警线程
        self.alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
        self.alert_thread.start()
        
        logger.info("监控系统已启动")
    
    def stop(self):
        """
        停止监控系统
        """
        self.running = False
        
        # 停止所有监控器
        for monitor in self.monitors.values():
            monitor.stop()
        
        # 等待线程结束
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        if self.alert_thread:
            self.alert_thread.join(timeout=10)
        
        logger.info("监控系统已停止")
    
    def _monitoring_loop(self):
        """
        监控循环
        """
        while self.running:
            time.sleep(5)
    
    def _alert_loop(self):
        """
        告警循环
        """
        while self.running:
            try:
                # 获取所有监控统计信息
                stats = self.get_all_stats()
                
                # 检查阈值
                alerts = self.alert_manager.check_thresholds(stats)
                
                # 发送告警
                for alert in alerts:
                    self.alert_manager.add_alert(alert)
                
                time.sleep(30)  # 每30秒检查一次
            except Exception as e:
                logger.error(f"告警循环错误: {e}")
                time.sleep(30)
    
    def get_all_stats(self):
        """
        获取所有监控统计信息
        
        Returns:
            监控统计信息
        """
        stats = {}
        
        for name, monitor in self.monitors.items():
            stats[name] = monitor.get_stats()
        
        return stats
    
    def get_monitor_data(self, monitor_name, limit=10):
        """
        获取监控器数据
        
        Args:
            monitor_name: 监控器名称
            limit: 数据点数量
            
        Returns:
            监控器数据
        """
        if monitor_name in self.monitors:
            return self.monitors[monitor_name].get_data(limit=limit)
        return []
    
    def get_alerts(self, limit=10):
        """
        获取告警
        
        Args:
            limit: 告警数量
            
        Returns:
            告警列表
        """
        return self.alert_manager.get_alerts(limit=limit)
    
    def update_network_stats(self, success, response_time=0):
        """
        更新网络统计信息
        
        Args:
            success: 是否成功
            response_time: 响应时间
        """
        if 'network' in self.monitors:
            self.monitors['network'].update_request_stats(success, response_time)
    
    def update_task_stats(self, task_status, task_time=0):
        """
        更新任务统计信息
        
        Args:
            task_status: 任务状态
            task_time: 任务执行时间
        """
        if 'task' in self.monitors:
            self.monitors['task'].update_task_stats(task_status, task_time)
    
    def save_stats(self, output_file='./monitoring_stats.json'):
        """
        保存统计信息
        
        Args:
            output_file: 输出文件
        """
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'all_stats': self.get_all_stats(),
                'alerts': self.get_alerts()
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"统计信息已保存到 {output_file}")
        except Exception as e:
            logger.error(f"保存统计信息失败: {e}")
    
    def get_dashboard_data(self):
        """
        获取仪表板数据
        
        Returns:
            仪表板数据
        """
        return {
            'stats': self.get_all_stats(),
            'alerts': self.get_alerts(limit=5),
            'system_data': self.get_monitor_data('system', limit=20),
            'network_data': self.get_monitor_data('network', limit=20),
            'task_data': self.get_monitor_data('task', limit=20)
        }
