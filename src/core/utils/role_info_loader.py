import json
import os
from pathlib import Path

ROLE_INFO_PATH = Path(__file__).parent.parent / "data" / "role_info.json"

_role_info_cache = None

def load_role_info():
    """加载角色信息缓存"""
    global _role_info_cache
    if _role_info_cache is None:
        if ROLE_INFO_PATH.exists():
            with open(ROLE_INFO_PATH, 'r', encoding='utf-8') as f:
                _role_info_cache = json.load(f)
        else:
            _role_info_cache = {}
    return _role_info_cache

def get_role_info(role_name: str) -> dict:
    """
    根据角色英文名获取完整角色信息

    Args:
        role_name: 角色英文名

    Returns:
        包含 cn, en, jp, anime 的字典，如果未找到返回基础信息
    """
    role_info = load_role_info()

    if role_name in role_info:
        return role_info[role_name]

    role_name_lower = role_name.lower()
    for key, info in role_info.items():
        if key.lower() == role_name_lower:
            return info

    for key, info in role_info.items():
        if role_name_lower.startswith(key.lower()) or key.lower().startswith(role_name_lower):
            return info

    for key, info in role_info.items():
        if role_name_lower.split()[0] == key.lower().split()[0] if role_name_lower and key.lower() else False:
            return info

    return {
        "cn": role_name,
        "en": role_name,
        "jp": "",
        "anime": ""
    }

def get_all_roles() -> dict:
    """获取所有角色信息"""
    return load_role_info()
