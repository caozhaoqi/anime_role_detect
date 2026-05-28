#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
存储模块公共工具函数
消除 registry.py 和 version/manager.py 中的重复代码
"""

import json
import logging
from typing import Tuple, Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ==================== 版本号处理工具 ====================

def parse_version(version_str: str) -> Tuple[int, int, int, str]:
    """
    解析版本号为可比较的元组
    
    Args:
        version_str: 版本号字符串（如 "1.2.3" 或 "1.2.3-beta.1"）
    
    Returns:
        版本号元组 (major, minor, patch, original)
    """
    try:
        # 分离主版本号和预发布/构建元数据
        main_version = version_str.split('-')[0].split('+')[0]
        parts = main_version.split('.')
        return (int(parts[0]), int(parts[1]), int(parts[2]), version_str)
    except (IndexError, ValueError):
        logger.warning(f"无效的版本号格式: {version_str}")
        return (0, 0, 0, version_str)


def compare_versions(version1: str, version2: str) -> int:
    """
    比较两个版本号
    
    Args:
        version1: 第一个版本号
        version2: 第二个版本号
    
    Returns:
        -1 if version1 < version2, 0 if equal, 1 if version1 > version2
    """
    v1 = parse_version(version1)
    v2 = parse_version(version2)
    
    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0


def is_valid_version(version: str) -> bool:
    """
    验证版本号格式是否符合 Semantic Versioning
    
    Args:
        version: 版本号字符串
    
    Returns:
        True 如果格式有效，否则 False
    """
    import re
    pattern = r'^(\d+)\.(\d+)\.(\d+)(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$'
    return re.match(pattern, version) is not None


# ==================== 日期时间序列化工具 ====================

def datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """
    将 datetime 对象转换为 ISO 格式字符串
    
    Args:
        dt: datetime 对象
    
    Returns:
        ISO 格式字符串或 None
    """
    if dt is None:
        return None
    return dt.isoformat()


def iso_to_datetime(iso_str: Optional[str]) -> Optional[datetime]:
    """
    将 ISO 格式字符串转换为 datetime 对象
    
    Args:
        iso_str: ISO 格式字符串
    
    Returns:
        datetime 对象或 None
    """
    if iso_str is None:
        return None
    try:
        return datetime.fromisoformat(iso_str)
    except ValueError:
        logger.warning(f"无效的 ISO 日期格式: {iso_str}")
        return None


def serialize_datetime_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归序列化字典中的所有 datetime 字段
    
    Args:
        data: 包含可能 datetime 字段的字典
    
    Returns:
        序列化后的字典
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, dict):
            result[key] = serialize_datetime_fields(value)
        else:
            result[key] = value
    return result


def deserialize_datetime_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归反序列化字典中的所有 datetime 字段
    
    Args:
        data: 包含 ISO 日期字符串的字典
    
    Returns:
        反序列化后的字典
    """
    result = {}
    datetime_keys = {'released_at', 'created_at', 'updated_at', 'installed_at'}
    
    for key, value in data.items():
        if isinstance(value, str) and key in datetime_keys:
            try:
                result[key] = datetime.fromisoformat(value)
            except ValueError:
                result[key] = value
        elif isinstance(value, dict):
            result[key] = deserialize_datetime_fields(value)
        else:
            result[key] = value
    return result


# ==================== JSON 持久化工具 ====================

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    从 JSON 文件加载数据，并自动处理 datetime 字段
    
    Args:
        file_path: 文件路径
    
    Returns:
        加载并反序列化后的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return deserialize_datetime_fields(data)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.error(f"加载 JSON 文件失败: {file_path}, 错误: {e}")
        return {}


def save_json_file(file_path: str, data: Dict[str, Any], ensure_ascii: bool = False, indent: int = 2):
    """
    将数据保存到 JSON 文件，并自动处理 datetime 字段
    
    Args:
        file_path: 文件路径
        data: 要保存的数据
        ensure_ascii: 是否确保 ASCII 编码
        indent: 缩进空格数
    """
    try:
        # 确保父目录存在
        from pathlib import Path
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            serialized_data = serialize_datetime_fields(data)
            json.dump(serialized_data, f, ensure_ascii=ensure_ascii, indent=indent)
    except Exception as e:
        logger.error(f"保存 JSON 文件失败: {file_path}, 错误: {e}")


# ==================== 元数据处理工具 ====================

def convert_to_version_info(data: Dict[str, Any], metadata_class) -> Any:
    """
    将字典数据转换为 VersionInfo 对象
    
    Args:
        data: 版本信息字典
        metadata_class: SkillMetadata 类引用
    
    Returns:
        VersionInfo 对象
    """
    from ardc.store.metadata import VersionInfo
    
    # 处理 metadata 字段
    if isinstance(data.get('metadata'), dict):
        data['metadata'] = metadata_class(**data['metadata'])
    
    return VersionInfo(**data)


def version_info_to_dict(version_info) -> Dict[str, Any]:
    """
    将 VersionInfo 对象转换为字典
    
    Args:
        version_info: VersionInfo 对象
    
    Returns:
        字典表示
    """
    if hasattr(version_info, 'dict'):
        return version_info.dict()
    return dict(version_info)
