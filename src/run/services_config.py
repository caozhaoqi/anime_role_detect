#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务配置文件 - 统一管理所有服务的端口和配置
"""

# 端口分配计划
# 8000-8009: 核心服务
# 8010-8019: 辅助服务
# 9000-9009: 监控和管理服务

SERVICES = {
    # ==================== 核心服务 ====================
    "multimedia_service": {
        "name": "多媒体服务",
        "script": "services/multimedia_service/multimedia_service_app.py",
        "port": 8002,
        "health_path": "/api/health",
        "description": "整合图像搜索和视频识别功能，支持以图搜图和视频实时抽帧识别",
        "enabled": True,
        "is_core": True,
    },
    "api_service": {
        "name": "主API服务",
        "script": "api/run_api.py",
        "port": 8001,
        "health_path": "/api/health",
        "description": "主API服务，提供角色识别和综合接口",
        "enabled": True,
        "is_core": True,
    },
    # ==================== AI模型服务 ====================
    "model_service": {
        "name": "模型服务",
        "script": "services/model_service/app_simple.py",
        "port": 8000,
        "health_path": "/api/health",
        "description": "AI模型推理服务，支持角色分类和标签识别",
        "enabled": True,
        "is_core": False,
    },
    # ==================== API网关 ====================
    "api_gateway": {
        "name": "API网关",
        "script": "services/api_gateway/app.py",
        "port": 8080,
        "health_path": "/api/health",
        "description": "统一API网关，聚合所有后端服务",
        "enabled": True,
        "is_core": False,
    },
}

# 服务分组
SERVICE_GROUPS = {
    "core": [key for key, config in SERVICES.items() if config.get("is_core")],
    "ai": ["model_service"],
    "multimedia": ["multimedia_service"],
    "gateway": ["api_gateway"],
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
