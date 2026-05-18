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
        self.registry = registry if registry else CollectorRegistry()
        
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
        
        self._start_time = time.time()
    
    def record_skill_execution(self, skill_id: str, version: str, status: str, 
                              latency: float = 0.0, error_type: str = None):
        self.skill_executions.labels(skill_id=skill_id, status=status, version=version).inc()
        
        if latency > 0:
            self.skill_latency.labels(skill_id=skill_id, version=version).observe(latency)
        
        if status == 'failed' and error_type:
            self.skill_errors.labels(skill_id=skill_id, error_type=error_type).inc()
    
    def record_workflow_execution(self, workflow_id: str, status: str, latency: float = 0.0):
        self.workflow_executions.labels(workflow_id=workflow_id, status=status).inc()
        
        if latency > 0:
            self.workflow_latency.labels(workflow_id=workflow_id).observe(latency)
    
    def update_skill_status(self, skill_id: str, version: str, enabled: bool):
        self.skill_status.labels(skill_id=skill_id, version=version).set(1 if enabled else 0)
    
    def record_request(self, endpoint: str, method: str, status_code: int, latency: float):
        self.requests_total.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).inc()
        
        self.request_latency.labels(endpoint=endpoint, method=method).observe(latency)
    
    def update_system_metrics(self, active_connections: int = None):
        self.uptime.set(time.time() - self._start_time)
        
        if active_connections is not None:
            self.active_connections.set(active_connections)
    
    def start_server(self, port: int = 8000, addr: str = '0.0.0.0'):
        start_http_server(port, addr, registry=self.registry)
        print(f"Prometheus metrics server started at http://{addr}:{port}/metrics")


class MetricsMiddleware:
    """FastAPI 中间件，自动记录请求指标"""
    
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
    """技能执行计时器上下文管理器"""
    
    def __init__(self, metrics: SkillMetrics, skill_id: str, version: str):
        self.metrics = metrics
        self.skill_id = skill_id
        self.version = version
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time
        
        if exc_type is not None:
            self.metrics.record_skill_execution(
                skill_id=self.skill_id,
                version=self.version,
                status='failed',
                latency=latency,
                error_type=exc_type.__name__
            )
        else:
            self.metrics.record_skill_execution(
                skill_id=self.skill_id,
                version=self.version,
                status='success',
                latency=latency
            )
        
        return False


_global_metrics = None


def get_metrics() -> SkillMetrics:
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = SkillMetrics()
    return _global_metrics


def init_metrics(port: int = 8000) -> SkillMetrics:
    metrics = get_metrics()
    metrics.start_server(port)
    return metrics