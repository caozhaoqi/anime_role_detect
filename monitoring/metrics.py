#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控模块
集成 Prometheus 监控，提供技能执行指标收集和暴露
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    start_http_server, CollectorRegistry
)
from datetime import datetime
from typing import Dict, Optional, Any
import time


class SkillMetrics:
    """技能执行指标收集器"""
    
    def __init__(self, registry: CollectorRegistry = None):
        """
        初始化指标收集器
        
        :param registry: Prometheus 收集器注册中心，默认为全局注册中心
        """
        self.registry = registry if registry else CollectorRegistry()
        
        # 技能执行指标
        self.skill_executions = Counter(
            'ardc_skill_executions_total',
            'Total skill executions',
            ['skill_id', 'status', 'version'],
            registry=self.registry
        )
        
        self.skill_latency = Histogram(
            'ardc_skill_execution_latency_seconds',
            'Skill execution latency in seconds',
            ['skill_id', 'version'],
            registry=self.registry,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.skill_errors = Counter(
            'ardc_skill_errors_total',
            'Total skill execution errors',
            ['skill_id', 'error_type'],
            registry=self.registry
        )
        
        # 工作流执行指标
        self.workflow_executions = Counter(
            'ardc_workflow_executions_total',
            'Total workflow executions',
            ['workflow_id', 'status'],
            registry=self.registry
        )
        
        self.workflow_latency = Histogram(
            'ardc_workflow_execution_latency_seconds',
            'Workflow execution latency in seconds',
            ['workflow_id'],
            registry=self.registry,
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
        )
        
        # 技能状态指标
        self.skill_status = Gauge(
            'ardc_skill_status',
            'Skill status (1=enabled, 0=disabled)',
            ['skill_id', 'version'],
            registry=self.registry
        )
        
        self.skill_memory_usage = Gauge(
            'ardc_skill_memory_usage_bytes',
            'Skill memory usage in bytes',
            ['skill_id'],
            registry=self.registry
        )
        
        # 系统指标
        self.active_connections = Gauge(
            'ardc_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.uptime = Gauge(
            'ardc_uptime_seconds',
            'Service uptime in seconds',
            registry=self.registry
        )
        
        # 请求指标
        self.requests_total = Counter(
            'ardc_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status_code'],
            registry=self.registry
        )
        
        self.request_latency = Summary(
            'ardc_request_latency_seconds',
            'API request latency',
            ['endpoint', 'method'],
            registry=self.registry
        )
        
        # 记录启动时间
        self._start_time = time.time()
    
    def record_skill_execution(self, skill_id: str, version: str, status: str, 
                              latency: float = 0.0, error_type: str = None):
        """
        记录技能执行指标
        
        :param skill_id: 技能ID
        :param version: 技能版本
        :param status: 执行状态 (success/failed)
        :param latency: 执行耗时（秒）
        :param error_type: 错误类型（失败时）
        """
        self.skill_executions.labels(skill_id=skill_id, status=status, version=version).inc()
        
        if latency > 0:
            self.skill_latency.labels(skill_id=skill_id, version=version).observe(latency)
        
        if status == 'failed' and error_type:
            self.skill_errors.labels(skill_id=skill_id, error_type=error_type).inc()
    
    def record_workflow_execution(self, workflow_id: str, status: str, latency: float = 0.0):
        """
        记录工作流执行指标
        
        :param workflow_id: 工作流ID
        :param status: 执行状态 (success/failed/running)
        :param latency: 执行耗时（秒）
        """
        self.workflow_executions.labels(workflow_id=workflow_id, status=status).inc()
        
        if latency > 0:
            self.workflow_latency.labels(workflow_id=workflow_id).observe(latency)
    
    def update_skill_status(self, skill_id: str, version: str, enabled: bool):
        """
        更新技能状态
        
        :param skill_id: 技能ID
        :param version: 技能版本
        :param enabled: 是否启用
        """
        self.skill_status.labels(skill_id=skill_id, version=version).set(1 if enabled else 0)
    
    def update_skill_memory_usage(self, skill_id: str, usage_bytes: int):
        """
        更新技能内存使用
        
        :param skill_id: 技能ID
        :param usage_bytes: 内存使用量（字节）
        """
        self.skill_memory_usage.labels(skill_id=skill_id).set(usage_bytes)
    
    def record_request(self, endpoint: str, method: str, status_code: int, latency: float):
        """
        记录API请求指标
        
        :param endpoint: 端点路径
        :param method: HTTP方法
        :param status_code: 状态码
        :param latency: 请求耗时（秒）
        """
        self.requests_total.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).inc()
        
        self.request_latency.labels(endpoint=endpoint, method=method).observe(latency)
    
    def update_system_metrics(self, active_connections: int = None):
        """
        更新系统指标
        
        :param active_connections: 活跃连接数
        """
        # 更新运行时间
        self.uptime.set(time.time() - self._start_time)
        
        if active_connections is not None:
            self.active_connections.set(active_connections)
    
    def start_server(self, port: int = 8000, addr: str = '0.0.0.0'):
        """
        启动 Prometheus 指标暴露服务
        
        :param port: 服务端口
        :param addr: 绑定地址
        """
        start_http_server(port, addr, registry=self.registry)
        print(f"Prometheus metrics server started at http://{addr}:{port}/metrics")


class MetricsMiddleware:
    """
    FastAPI 中间件，自动记录请求指标
    """
    
    def __init__(self, metrics: SkillMetrics):
        self.metrics = metrics
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        latency = time.time() - start_time
        endpoint = request.url.path
        method = request.method
        status_code = response.status_code
        
        self.metrics.record_request(endpoint, method, status_code, latency)
        
        return response


class ExecutionTimer:
    """
    技能执行计时器上下文管理器
    """
    
    def __init__(self, metrics: SkillMetrics, skill_id: str, version: str):
        self.metrics = metrics
        self.skill_id = skill_id
        self.version = version
        self.start_time = None
        self.error_type = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time
        
        if exc_type is not None:
            # 执行失败
            self.error_type = exc_type.__name__
            self.metrics.record_skill_execution(
                skill_id=self.skill_id,
                version=self.version,
                status='failed',
                latency=latency,
                error_type=self.error_type
            )
        else:
            # 执行成功
            self.metrics.record_skill_execution(
                skill_id=self.skill_id,
                version=self.version,
                status='success',
                latency=latency
            )
        
        return False  # 不抑制异常


class MetricCollector:
    """
    指标收集器基类
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def collect(self) -> Dict[str, Any]:
        """
        收集指标数据
        
        :return: 指标数据字典
        """
        raise NotImplementedError("子类必须实现 collect 方法")


class SkillExecutionCollector(MetricCollector):
    """
    技能执行指标收集器
    """
    
    def __init__(self):
        super().__init__("skill_execution")
        self.executions = []
    
    def record(self, skill_id: str, version: str, status: str, 
               latency: float, timestamp: Optional[datetime] = None):
        """
        记录执行记录
        
        :param skill_id: 技能ID
        :param version: 技能版本
        :param status: 执行状态
        :param latency: 执行耗时
        :param timestamp: 时间戳
        """
        self.executions.append({
            'skill_id': skill_id,
            'version': version,
            'status': status,
            'latency': latency,
            'timestamp': timestamp or datetime.now()
        })
    
    def collect(self) -> Dict[str, Any]:
        """收集统计数据"""
        if not self.executions:
            return {
                'total_executions': 0,
                'success_rate': 0.0,
                'avg_latency': 0.0,
                'min_latency': 0.0,
                'max_latency': 0.0
            }
        
        total = len(self.executions)
        success = sum(1 for e in self.executions if e['status'] == 'success')
        latencies = [e['latency'] for e in self.executions]
        
        return {
            'total_executions': total,
            'success_count': success,
            'failed_count': total - success,
            'success_rate': success / total * 100,
            'avg_latency': sum(latencies) / total,
            'min_latency': min(latencies),
            'max_latency': max(latencies)
        }


# 全局指标实例
_global_metrics = None


def get_metrics() -> SkillMetrics:
    """
    获取全局指标实例
    
    :return: SkillMetrics 实例
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = SkillMetrics()
    return _global_metrics


def init_metrics(port: int = 8000) -> SkillMetrics:
    """
    初始化并启动指标服务
    
    :param port: 服务端口
    :return: SkillMetrics 实例
    """
    metrics = get_metrics()
    metrics.start_server(port)
    return metrics