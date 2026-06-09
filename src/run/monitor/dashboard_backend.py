#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控仪表板后端逻辑 - 服务状态检查和追踪数据获取
微服务api调用拓扑图 
动态显示服务状态和调用关系
"""

import os
import sys
import time
import requests
from datetime import datetime
from typing import Dict, List

# 添加 src/ 和 src/run/ 到Python路径
_current_dir = os.path.dirname(os.path.abspath(__file__))         # .../src/run/monitor/
_src_dir = os.path.dirname(os.path.dirname(_current_dir))         # .../src/
_run_dir = os.path.dirname(_current_dir)                           # .../src/run/
sys.path.insert(0, _src_dir)
sys.path.insert(0, _run_dir)

from services_config import SERVICES

# 追踪存储服务（延迟导入）
_trace_storage_service = None


def get_trace_storage_service():
    """延迟获取追踪存储服务"""
    global _trace_storage_service
    if _trace_storage_service is None:
        from src.services.support.trace_storage_service import get_trace_storage_service as get_service
        _trace_storage_service = get_service()
    return _trace_storage_service


def check_service_health(service_config: dict) -> dict:
    """检查单个服务的健康状态"""
    result = {
        "name": service_config["name"],
        "port": service_config["port"],
        "status": "unknown",
        "response_time": 0,
        "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "enabled": service_config.get("enabled", True),
        "is_core": service_config.get("is_core", False),
    }

    if not result["enabled"]:
        result["status"] = "disabled"
        return result

    try:
        health_path = service_config.get("health_path", "/health")
        url = f"http://localhost:{service_config['port']}{health_path}"

        start_time = time.time()
        response = requests.get(url, timeout=2)
        end_time = time.time()

        result["response_time"] = round((end_time - start_time) * 1000, 2)

        if response.status_code == 200:
            result["status"] = "healthy"
        else:
            result["status"] = f"error_{response.status_code}"
    except requests.exceptions.Timeout:
        result["status"] = "timeout"
    except requests.exceptions.ConnectionError:
        result["status"] = "unreachable"
    except Exception as e:
        result["status"] = f"error: {str(e)[:20]}"

    return result


def get_all_services_status() -> List[dict]:
    """获取所有服务的状态"""
    services_status = []

    for key, config in SERVICES.items():
        status = check_service_health(config)
        status["key"] = key
        services_status.append(status)

    return services_status


def get_tracing_stats():
    """获取追踪统计信息"""
    storage = get_trace_storage_service()
    return storage.get_aggregated_stats(24)


def get_recent_traces(limit: int = 20):
    """获取最近的追踪记录"""
    storage = get_trace_storage_service()
    return storage.get_recent_traces(limit)


def get_trace_details(trace_id: str):
    """获取追踪详情"""
    storage = get_trace_storage_service()
    return storage.get_trace(trace_id)


def get_service_relations():
    """获取服务调用关系数据"""
    # 定义服务之间的调用关系
    service_relations = {
        "api_gateway": ["api_service", "model_service", "multimedia_service"],
        "api_service": ["model_service"],
        "multimedia_service": ["model_service"],
    }
    
    return service_relations


def get_topology_data():
    """获取拓扑图数据，包含服务状态和调用关系"""
    services_status = get_all_services_status()
    service_relations = get_service_relations()
    
    nodes = []
    edges = []
    
    # 创建节点
    for service in services_status:
        nodes.append({
            "id": service["key"],
            "name": service["name"],
            "port": service["port"],
            "status": service["status"],
            "is_core": service["is_core"],
            "response_time": service["response_time"],
        })
    
    # 创建边（调用关系）
    for source, targets in service_relations.items():
        for target in targets:
            edges.append({
                "source": source,
                "target": target,
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
    }