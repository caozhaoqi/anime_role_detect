#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存监控和coredump保存机制

实现内存使用监控、预警和崩溃前状态保存功能
"""

import os
import time
import threading
import json
import psutil
import signal
import traceback
import subprocess
from datetime import datetime
from collections import deque

from core.logging.global_logger import get_logger

logger = get_logger("memory_monitor")


class MemoryMonitor:
    """
    内存监控器
    """
    
    def __init__(self, memory_threshold=70, critical_threshold=85, interval=10, max_coredump_files=5, max_state_files=10):
        """
        初始化内存监控器
        
        Args:
            memory_threshold: 内存使用阈值（百分比），超过此值开始预警
            critical_threshold: 临界内存使用阈值（百分比），超过此值准备保存状态
            interval: 监控间隔（秒）
            max_coredump_files: 最大coredump文件数量
            max_state_files: 最大状态文件数量
        """
        self.memory_threshold = memory_threshold
        self.critical_threshold = critical_threshold
        self.interval = interval
        self.running = False
        self.thread = None
        self.data = deque(maxlen=100)  # 最多保存100个数据点
        self.last_alert_time = 0
        self.cooldown_period = 300  # 告警冷却期（秒）
        self.save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        os.makedirs(self.save_dir, exist_ok=True)
        self.max_coredump_files = max_coredump_files
        self.max_state_files = max_state_files
        self.memory_trend = deque(maxlen=10)  # 内存趋势数据
    
    def start(self):
        """
        启动内存监控器
        """
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("内存监控器已启动")
    
    def stop(self):
        """
        停止内存监控器
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("内存监控器已停止")
    
    def _run(self):
        """
        运行内存监控器
        """
        while self.running:
            try:
                # 收集内存数据
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used = memory.used / (1024 * 1024 * 1024)  # GB
                memory_total = memory.total / (1024 * 1024 * 1024)  # GB
                
                current_time = datetime.now()
                data_point = {
                    'timestamp': current_time.isoformat(),
                    'memory_percent': memory_percent,
                    'memory_used': memory_used,
                    'memory_total': memory_total
                }
                self.data.append(data_point)
                self.memory_trend.append(memory_percent)
                
                # 检查内存使用情况
                self._check_memory_usage(memory_percent, memory_used, memory_total)
                
                # 检查内存泄漏
                self._check_memory_leak()
                
                # 清理旧文件
                self._cleanup_old_files()
                
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"内存监控器错误: {e}")
                time.sleep(self.interval)
    
    def _check_memory_usage(self, memory_percent, memory_used, memory_total):
        """
        检查内存使用情况
        
        Args:
            memory_percent: 内存使用百分比
            memory_used: 已使用内存（GB）
            memory_total: 总内存（GB）
        """
        current_time = time.time()
        
        # 内存使用预警
        if memory_percent > self.memory_threshold and current_time - self.last_alert_time > self.cooldown_period:
            logger.warning(f"内存使用预警: {memory_percent:.2f}% (已使用: {memory_used:.2f}GB, 总内存: {memory_total:.2f}GB)")
            self.last_alert_time = current_time
        
        # 临界内存使用，准备保存状态
        if memory_percent > self.critical_threshold:
            logger.critical(f"临界内存使用: {memory_percent:.2f}%，准备保存系统状态")
            self._save_system_state()
    
    def _check_memory_leak(self):
        """
        检查内存泄漏
        """
        if len(self.memory_trend) < 10:
            return
        
        # 计算内存趋势
        recent_values = list(self.memory_trend)[-5:]
        older_values = list(self.memory_trend)[:-5]
        
        if older_values and recent_values:
            avg_recent = sum(recent_values) / len(recent_values)
            avg_older = sum(older_values) / len(older_values)
            
            # 如果内存使用持续增长超过3%，可能存在内存泄漏
            if avg_recent > avg_older + 3:
                logger.warning(f"检测到内存泄漏迹象: 内存使用从 {avg_older:.2f}% 增长到 {avg_recent:.2f}%")
                # 执行垃圾回收
                import gc
                gc.collect()
                logger.info("已执行垃圾回收，尝试释放内存")
    
    def _cleanup_old_files(self):
        """
        清理旧的coredump和状态文件
        """
        try:
            # 清理coredump文件
            coredump_files = sorted(
                [f for f in os.listdir(self.save_dir) if f.startswith('coredump_')],
                key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x))
            )
            
            while len(coredump_files) > self.max_coredump_files:
                file_to_delete = coredump_files.pop(0)
                file_path = os.path.join(self.save_dir, file_to_delete)
                try:
                    os.remove(file_path)
                    logger.info(f"已清理旧coredump文件: {file_to_delete}")
                except Exception as e:
                    logger.error(f"清理coredump文件失败: {e}")
            
            # 清理状态文件
            state_files = sorted(
                [f for f in os.listdir(self.save_dir) if f.startswith('system_state_')],
                key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x))
            )
            
            while len(state_files) > self.max_state_files:
                file_to_delete = state_files.pop(0)
                file_path = os.path.join(self.save_dir, file_to_delete)
                try:
                    os.remove(file_path)
                    logger.info(f"已清理旧状态文件: {file_to_delete}")
                except Exception as e:
                    logger.error(f"清理状态文件失败: {e}")
                    
        except Exception as e:
            logger.error(f"清理旧文件失败: {e}")
    
    def _save_system_state(self):
        """
        保存系统状态，包括内存使用情况、进程信息等
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state_file = os.path.join(self.save_dir, f"system_state_{timestamp}.json")
            
            # 收集系统状态
            system_state = {
                'timestamp': datetime.now().isoformat(),
                'memory': {
                    'percent': psutil.virtual_memory().percent,
                    'used': psutil.virtual_memory().used / (1024 * 1024 * 1024),
                    'total': psutil.virtual_memory().total / (1024 * 1024 * 1024)
                },
                'cpu': {
                    'percent': psutil.cpu_percent(interval=0.1),
                    'count': psutil.cpu_count()
                },
                'processes': self._get_process_info(),
                'memory_data': list(self.data),
                'memory_trend': list(self.memory_trend)
            }
            
            # 保存状态文件
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(system_state, f, ensure_ascii=False, indent=2)
            
            logger.info(f"系统状态已保存到: {state_file}")
            
            # 尝试生成coredump
            self._generate_coredump()
            
        except Exception as e:
            logger.error(f"保存系统状态失败: {e}")
    
    def _get_process_info(self):
        """
        获取进程信息
        
        Returns:
            进程信息列表
        """
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
                try:
                    proc_info = proc.info
                    # 检查memory_percent是否为None
                    memory_percent = proc_info.get('memory_percent')
                    if memory_percent is not None and memory_percent > 1.0:  # 只保存内存使用超过1%的进程
                        processes.append({
                            'pid': proc_info['pid'],
                            'name': proc_info['name'],
                            'memory_percent': memory_percent,
                            'cpu_percent': proc_info['cpu_percent']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # 按内存使用百分比排序
            processes.sort(key=lambda x: x['memory_percent'], reverse=True)
            
            # 只返回前10个内存使用最高的进程
            return processes[:10]
        except Exception as e:
            logger.error(f"获取进程信息失败: {e}")
            return []
    
    def _generate_coredump(self):
        """
        生成coredump文件
        """
        try:
            # 获取当前进程ID
            current_pid = os.getpid()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            coredump_file = os.path.join(self.save_dir, f"coredump_{current_pid}_{timestamp}.dump")
            
            # 检查磁盘空间
            disk_usage = psutil.disk_usage(self.save_dir)
            if disk_usage.percent > 90:
                logger.warning("磁盘空间不足，跳过coredump生成")
                return
            
            # 检查是否有gcore命令
            import shutil
            if not shutil.which('gcore'):
                logger.warning("gcore命令不可用，使用备用方法")
                self._generate_coredump_fallback()
                return
            
            # 使用gcore命令生成coredump
            # 注意：需要足够的权限
            result = subprocess.run(
                ['gcore', '-o', coredump_file, str(current_pid)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Coredump已生成: {coredump_file}")
            else:
                logger.warning(f"生成coredump失败: {result.stderr}")
                
                # 如果gcore失败，尝试使用其他方法
                self._generate_coredump_fallback()
                
        except Exception as e:
            logger.error(f"生成coredump错误: {e}")
            self._generate_coredump_fallback()
    
    def _generate_coredump_fallback(self):
        """
        生成coredump的备用方法
        """
        try:
            # 保存详细的内存使用信息作为备用
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            memory_file = os.path.join(self.save_dir, f"memory_dump_{timestamp}.json")
            
            # 收集更详细的内存信息
            memory_dump = {
                'timestamp': datetime.now().isoformat(),
                'memory': dict(psutil.virtual_memory()._asdict()),
                'swap': dict(psutil.swap_memory()._asdict()),
                'process_memory': self._get_process_memory_detail()
            }
            
            # 转换字节为GB
            for key in ['total', 'available', 'used', 'free']:
                if key in memory_dump['memory']:
                    memory_dump['memory'][f'{key}_gb'] = memory_dump['memory'][key] / (1024 * 1024 * 1024)
            
            for key in ['total', 'used', 'free']:
                if key in memory_dump['swap']:
                    memory_dump['swap'][f'{key}_gb'] = memory_dump['swap'][key] / (1024 * 1024 * 1024)
            
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_dump, f, ensure_ascii=False, indent=2)
            
            logger.info(f"内存详细信息已保存到: {memory_file}")
            
        except Exception as e:
            logger.error(f"保存内存详细信息失败: {e}")
    
    def _get_process_memory_detail(self):
        """
        获取进程内存详细信息
        
        Returns:
            进程内存详细信息
        """
        process_memory = []
        try:
            current_pid = os.getpid()
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'memory_percent']):
                try:
                    proc_info = proc.info
                    # 检查memory_percent是否为None
                    memory_percent = proc_info.get('memory_percent')
                    if proc_info['pid'] == current_pid or (memory_percent is not None and memory_percent > 0.5):
                        memory_info = proc_info['memory_info']
                        process_memory.append({
                            'pid': proc_info['pid'],
                            'name': proc_info['name'],
                            'memory_percent': memory_percent,
                            'rss': memory_info.rss / (1024 * 1024),  # MB
                            'vms': memory_info.vms / (1024 * 1024),  # MB
                            'shared': memory_info.shared / (1024 * 1024),  # MB
                            'text': memory_info.text / (1024 * 1024),  # MB
                            'data': memory_info.data / (1024 * 1024)  # MB
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # 按内存使用排序
            process_memory.sort(key=lambda x: x['rss'], reverse=True)
            
            return process_memory[:15]  # 只返回前15个进程
        except Exception as e:
            logger.error(f"获取进程内存详细信息失败: {e}")
            return []
    
    def get_data(self, limit=10):
        """
        获取内存监控数据
        
        Args:
            limit: 数据点数量
            
        Returns:
            内存监控数据列表
        """
        return list(self.data)[-limit:]
    
    def get_stats(self):
        """
        获取内存统计信息
        
        Returns:
            内存统计信息
        """
        if not self.data:
            return {}
        
        memory_values = [d['memory_percent'] for d in self.data]
        
        return {
            'average_memory': sum(memory_values) / len(memory_values),
            'max_memory': max(memory_values),
            'min_memory': min(memory_values),
            'data_points': len(self.data)
        }


class MemoryEmergencyHandler:
    """
    内存紧急处理机制
    """
    
    def __init__(self, memory_monitor, max_memory_usage=85):
        """
        初始化内存紧急处理机制
        
        Args:
            memory_monitor: 内存监控器实例
            max_memory_usage: 最大内存使用阈值（百分比）
        """
        self.memory_monitor = memory_monitor
        self.max_memory_usage = max_memory_usage
        self.running = False
        self.thread = None
        self.last_emergency_time = 0
        self.cooldown_period = 60  # 紧急措施冷却期（秒）
    
    def start(self):
        """
        启动内存紧急处理机制
        """
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("内存紧急处理机制已启动")
    
    def stop(self):
        """
        停止内存紧急处理机制
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("内存紧急处理机制已停止")
    
    def _run(self):
        """
        运行内存紧急处理机制
        """
        while self.running:
            try:
                # 检查内存使用情况
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # 检查是否在冷却期内
                current_time = time.time()
                if current_time - self.last_emergency_time < self.cooldown_period:
                    time.sleep(5)  # 冷却期内减少检查频率
                    continue
                
                # 如果内存使用超过最大阈值，执行紧急措施
                if memory_percent > self.max_memory_usage:
                    logger.critical(f"内存使用超过最大阈值: {memory_percent:.2f}%，执行紧急措施")
                    self._execute_emergency_measures()
                    self.last_emergency_time = current_time
                
                time.sleep(2)  # 更频繁地检查
            except Exception as e:
                logger.error(f"内存紧急处理机制错误: {e}")
                time.sleep(2)
    
    def _execute_emergency_measures(self):
        """
        执行紧急措施
        """
        try:
            # 1. 保存系统状态
            self.memory_monitor._save_system_state()
            
            # 2. 尝试释放内存
            self._release_memory()
            
            # 3. 记录紧急情况
            self._log_emergency()
            
        except Exception as e:
            logger.error(f"执行紧急措施失败: {e}")
    
    def _release_memory(self):
        """
        尝试释放内存
        """
        try:
            # 这里可以添加具体的内存释放逻辑
            # 例如：清理缓存、关闭不必要的进程等
            logger.info("尝试释放内存...")
            
            # 强制垃圾回收
            import gc
            gc.collect()
            logger.info("已执行垃圾回收")
            
            # 尝试清理系统缓存
            self._clear_system_cache()
            
        except Exception as e:
            logger.error(f"释放内存失败: {e}")
    
    def _clear_system_cache(self):
        """
        尝试清理系统缓存
        """
        try:
            # 尝试清理页面缓存、 dentries 和 inodes
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3')
            logger.info("已尝试清理系统缓存")
        except Exception as e:
            logger.warning(f"清理系统缓存失败: {e} (可能需要root权限)")
    
    def _log_emergency(self):
        """
        记录紧急情况
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            emergency_file = os.path.join(self.memory_monitor.save_dir, f"emergency_{timestamp}.log")
            
            emergency_info = {
                'timestamp': datetime.now().isoformat(),
                'event': 'memory_emergency',
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used': psutil.virtual_memory().used / (1024 * 1024 * 1024),
                'memory_total': psutil.virtual_memory().total / (1024 * 1024 * 1024),
                'system_state': self._get_current_system_state()
            }
            
            with open(emergency_file, 'w', encoding='utf-8') as f:
                json.dump(emergency_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"紧急情况已记录到: {emergency_file}")
            
        except Exception as e:
            logger.error(f"记录紧急情况失败: {e}")
    
    def _get_current_system_state(self):
        """
        获取当前系统状态
        
        Returns:
            当前系统状态
        """
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'process_count': len(psutil.pids()),
                'disk_usage': psutil.disk_usage('/').percent
            }
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {}


# 全局内存监控实例
memory_monitor = None
memory_emergency_handler = None


def init_memory_monitoring():
    """
    初始化内存监控
    """
    global memory_monitor, memory_emergency_handler
    
    if memory_monitor is None:
        memory_monitor = MemoryMonitor()
        memory_monitor.start()
        
        memory_emergency_handler = MemoryEmergencyHandler(memory_monitor)
        memory_emergency_handler.start()
        
        logger.info("内存监控系统初始化完成")


def shutdown_memory_monitoring():
    """
    关闭内存监控
    """
    global memory_monitor, memory_emergency_handler
    
    if memory_emergency_handler:
        memory_emergency_handler.stop()
    
    if memory_monitor:
        memory_monitor.stop()
        
    logger.info("内存监控系统已关闭")


def get_memory_monitor():
    """
    获取内存监控实例
    
    Returns:
        内存监控实例
    """
    global memory_monitor
    return memory_monitor


def get_memory_emergency_handler():
    """
    获取内存紧急处理实例
    
    Returns:
        内存紧急处理实例
    """
    global memory_emergency_handler
    return memory_emergency_handler


if __name__ == "__main__":
    # 测试内存监控
    init_memory_monitoring()
    
    try:
        # 运行一段时间
        print("内存监控已启动，按Ctrl+C退出...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_memory_monitoring()
