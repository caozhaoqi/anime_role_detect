#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prometheus 监控指标集成
用于监控服务性能和资源使用情况
"""

from prometheus_client import (
    start_http_server,
    Counter,
    Histogram,
    Gauge,
    Summary
)
import psutil
import time
import threading

# 请求计数器
REQUESTS = Counter(
    'anime_role_detect_requests_total',
    'Total number of requests',
    ['endpoint', 'method', 'status']
)

# 推理耗时直方图
INFERENCE_TIME = Histogram(
    'anime_role_detect_inference_duration_seconds',
    'Time spent on model inference',
    ['model_name']
)

# API 响应时间直方图
API_RESPONSE_TIME = Histogram(
    'anime_role_detect_api_response_duration_seconds',
    'API response time',
    ['endpoint']
)

# 内存使用量
MEMORY_USAGE = Gauge(
    'anime_role_detect_memory_usage_bytes',
    'Current memory usage',
    ['type']
)

# GPU 内存使用量
GPU_MEMORY_USAGE = Gauge(
    'anime_role_detect_gpu_memory_usage_bytes',
    'Current GPU memory usage'
)

# 模型加载状态
MODEL_LOADED = Gauge(
    'anime_role_detect_model_loaded',
    'Whether the model is loaded',
    ['model_name']
)

# 请求成功率
SUCCESS_RATE = Summary(
    'anime_role_detect_success_rate',
    'Request success rate'
)

# 并发请求数
ACTIVE_REQUESTS = Gauge(
    'anime_role_detect_active_requests',
    'Number of active requests'
)

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, port=9090):
        """
        初始化指标收集器
        
        Args:
            port: Prometheus 暴露端口
        """
        self.port = port
        self.running = False
        self.thread = None
        
    def start(self):
        """启动指标服务器"""
        if self.running:
            return
        
        # 启动 Prometheus HTTP 服务器
        start_http_server(self.port)
        
        # 启动定期指标收集
        self.running = True
        self.thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self.thread.start()
        
        print(f"Prometheus 指标服务器已启动，端口: {self.port}")
    
    def stop(self):
        """停止指标服务器"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _collect_metrics(self):
        """定期收集系统指标"""
        while self.running:
            try:
                # 收集内存使用信息
                mem = psutil.virtual_memory()
                MEMORY_USAGE.labels(type='total').set(mem.total)
                MEMORY_USAGE.labels(type='available').set(mem.available)
                MEMORY_USAGE.labels(type='used').set(mem.used)
                MEMORY_USAGE.labels(type='percent').set(mem.percent)
                
                # 收集 CPU 信息
                cpu_percent = psutil.cpu_percent()
                MEMORY_USAGE.labels(type='cpu_percent').set(cpu_percent)
                
                # 收集 GPU 信息（如果可用）
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        GPU_MEMORY_USAGE.set(gpu.memoryUsed * 1024 * 1024)
                except ImportError:
                    pass
                
            except Exception as e:
                print(f"收集指标时发生错误: {e}")
            
            time.sleep(5)  # 每5秒收集一次
    
    @staticmethod
    def record_request(endpoint, method, status):
        """记录请求"""
        REQUESTS.labels(endpoint=endpoint, method=method, status=status).inc()
    
    @staticmethod
    def record_inference_time(model_name, duration):
        """记录推理时间"""
        INFERENCE_TIME.labels(model_name=model_name).observe(duration)
    
    @staticmethod
    def record_api_response_time(endpoint, duration):
        """记录 API 响应时间"""
        API_RESPONSE_TIME.labels(endpoint=endpoint).observe(duration)
    
    @staticmethod
    def set_model_loaded(model_name, loaded):
        """设置模型加载状态"""
        MODEL_LOADED.labels(model_name=model_name).set(1 if loaded else 0)
    
    @staticmethod
    def inc_active_requests():
        """增加并发请求数"""
        ACTIVE_REQUESTS.inc()
    
    @staticmethod
    def dec_active_requests():
        """减少并发请求数"""
        ACTIVE_REQUESTS.dec()


def main():
    """测试指标收集器"""
    collector = MetricsCollector(port=9090)
    collector.start()
    
    print("按 Ctrl+C 停止...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        collector.stop()
        print("已停止")

if __name__ == "__main__":
    main()
