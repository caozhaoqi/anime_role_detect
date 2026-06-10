#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务配置文件 - 统一管理所有服务的端口和配置
"""

# 端口分配计划
# 3000-3009: 前端
# 8000-8009: 核心服务
# 8010-8019: 辅助服务
# 8080-8089: 网关
# 9000-9009: 监控和管理服务

SERVICES = {
    # ==================== 核心服务 ====================
    "api_service": {
        "name": "主API服务",
        "script": "api/run_api.py",
        "port": 8001,
        "health_path": "/api/health",
        "api_base": "/api/",
        "has_swagger": False,
        "description": "主API服务，提供角色识别和综合接口",
        "enabled": True,
        "is_core": True,
    },
    "multimedia_service": {
        "name": "多媒体服务",
        "script": "services/multimedia_service/multimedia_service_app.py",
        "port": 8002,
        "health_path": "/api/health",
        "api_base": "/api/",
        "has_swagger": True,
        "description": "整合图像搜索和视频识别功能，支持以图搜图和视频实时抽帧识别",
        "enabled": True,
        "is_core": True,
    },
    "model_service": {
        "name": "模型服务",
        "script": "services/model_service/app.py",
        "port": 8000,
        "health_path": "/api/health",
        "api_base": "/api/",
        "has_swagger": True,
        "description": "AI模型推理服务，支持角色分类和标签识别，被所有核心服务依赖",
        "enabled": True,
        "is_core": True,
    },
    # ==================== 搜索服务 ====================
    "search_service": {
        "name": "搜索服务",
        "script": "services/search_service/app_queue.py",
        "port": 8003,
        "health_path": "/api/health",
        "api_base": "/api/",
        "has_swagger": True,
        "description": "图像搜索队列服务，异步处理以图搜图请求",
        "enabled": True,
        "is_core": False,
    },
    # ==================== API网关 ====================
    "api_gateway": {
        "name": "API网关",
        "script": "services/api_gateway/app.py",
        "port": 8080,
        "health_path": "/api/health",
        "api_base": "/",
        "has_swagger": True,
        "description": "统一API网关，聚合所有后端服务，前端唯一入口",
        "enabled": True,
        "is_core": True,
    },
    # ==================== 前端 ====================
    "frontend": {
        "name": "前端页面",
        "script": "frontend/package.json",
        "port": 3000,
        "health_path": "/",
        "api_base": "/",
        "has_swagger": False,
        "description": "Vue.js前端应用，用户交互界面",
        "enabled": True,
        "is_core": False,
    },
    # ==================== 监控和管理服务 ====================
    "monitor_dashboard": {
        "name": "监控面板",
        "script": "run/monitor/monitor_dashboard.py",
        "port": 9000,
        "health_path": "/api/health",
        "api_base": "/",
        "has_swagger": False,
        "description": "服务状态监控、微服务拓扑图、API链路追踪仪表板",
        "enabled": True,
        "is_core": False,
    },
}

# 服务分组
SERVICE_GROUPS = {
    "core": [key for key, config in SERVICES.items() if config.get("is_core")],
    "ai": ["model_service"],
    "search": ["search_service"],
    "multimedia": ["multimedia_service"],
    "gateway": ["api_gateway"],
    "frontend": ["frontend"],
    "monitoring": ["monitor_dashboard"],
    "all": list(SERVICES.keys()),
}


def get_service_by_name(name):
    """根据名称获取服务配置"""
    return SERVICES.get(name)


def get_services_by_group(group_name):
    """根据分组获取服务列表"""
    return SERVICE_GROUPS.get(group_name, [])


def list_all_services():
    """列出所有服务"""
    return SERVICES.keys()


def get_service_port(name):
    """获取服务端口"""
    service = SERVICES.get(name)
    return service["port"] if service else None