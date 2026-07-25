"""
实用工具函数
工具类设计，提供常用功能

功能：
1. 类型检查工具
2. 日期时间工具
3. JSON 工具
4. 列表工具
5. 文件工具
6. 字符串工具
7. 安全工具
"""

import base64
import hashlib
import json
import os
import random
import re
import string
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union


class TypeUtils:
    @staticmethod
    def is_int(value: Any) -> bool:
        try:
            int(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def is_float(value: Any) -> bool:
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def is_str(value: Any) -> bool:
        return isinstance(value, str)

    @staticmethod
    def is_list(value: Any) -> bool:
        return isinstance(value, list)

    @staticmethod
    def is_dict(value: Any) -> bool:
        return isinstance(value, dict)

    @staticmethod
    def is_bool(value: Any) -> bool:
        return isinstance(value, bool)

    @staticmethod
    def is_none(value: Any) -> bool:
        return value is None

    @staticmethod
    def is_empty(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (str, list, dict, tuple)):
            return len(value) == 0
        return False

    @staticmethod
    def to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def to_str(value: Any, default: str = '') -> str:
        if value is None:
            return default
        return str(value)

    @staticmethod
    def to_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower_value = value.lower()
            if lower_value in ('true', '1', 'yes', 'y', '是'):
                return True
            if lower_value in ('false', '0', 'no', 'n', '否'):
                return False
        if isinstance(value, int):
            return value != 0
        return default


class DateTimeUtils:
    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def now_local() -> datetime:
        return datetime.now()

    @staticmethod
    def to_isoformat(value: datetime) -> str:
        return value.isoformat()

    @staticmethod
    def from_isoformat(value: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def to_timestamp(value: datetime) -> float:
        return value.timestamp()

    @staticmethod
    def from_timestamp(value: float) -> datetime:
        return datetime.fromtimestamp(value, timezone.utc)

    @staticmethod
    def format(value: datetime, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
        return value.strftime(fmt)

    @staticmethod
    def parse(value: str, fmt: str = '%Y-%m-%d %H:%M:%S') -> Optional[datetime]:
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def days_ago(days: int) -> datetime:
        return DateTimeUtils.now() - timedelta(days=days)

    @staticmethod
    def hours_ago(hours: int) -> datetime:
        return DateTimeUtils.now() - timedelta(hours=hours)

    @staticmethod
    def minutes_ago(minutes: int) -> datetime:
        return DateTimeUtils.now() - timedelta(minutes=minutes)

    @staticmethod
    def add_days(value: datetime, days: int) -> datetime:
        return value + timedelta(days=days)

    @staticmethod
    def add_hours(value: datetime, hours: int) -> datetime:
        return value + timedelta(hours=hours)

    @staticmethod
    def add_minutes(value: datetime, minutes: int) -> datetime:
        return value + timedelta(minutes=minutes)

    @staticmethod
    def diff_days(start: datetime, end: datetime) -> float:
        return (end - start).total_seconds() / (24 * 3600)

    @staticmethod
    def diff_hours(start: datetime, end: datetime) -> float:
        return (end - start).total_seconds() / 3600

    @staticmethod
    def is_expired(expire_time: datetime) -> bool:
        return DateTimeUtils.now() > expire_time

    @staticmethod
    def get_expire_time(days: int = 0, hours: int = 0, minutes: int = 0) -> datetime:
        return DateTimeUtils.now() + timedelta(days=days, hours=hours, minutes=minutes)


class JSONUtils:
    @staticmethod
    def dumps(obj: Any, **kwargs) -> str:
        defaults = {
            'ensure_ascii': False,
            'indent': 2,
            'sort_keys': True,
            'default': lambda o: o.isoformat() if isinstance(o, datetime) else None,
        }
        defaults.update(kwargs)
        return json.dumps(obj, **defaults)

    @staticmethod
    def loads(s: str, **kwargs) -> Any:
        return json.loads(s, **kwargs)

    @staticmethod
    def get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
        keys = key.split('.')
        result = data
        for k in keys:
            if isinstance(result, dict) and k in result:
                result = result[k]
            else:
                return default
        return result

    @staticmethod
    def merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = JSONUtils.merge(result[key], value)
            else:
                result[key] = value
        return result


class ListUtils:
    @staticmethod
    def find_by_key(items: List[Dict[str, Any]], key: str, value: Any, default: Any = None) -> Optional[Dict[str, Any]]:
        for item in items:
            if item.get(key) == value:
                return item
        return default

    @staticmethod
    def filter_by_key(items: List[Dict[str, Any]], key: str, value: Any) -> List[Dict[str, Any]]:
        return [item for item in items if item.get(key) == value]

    @staticmethod
    def group_by_key(items: List[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
        groups = {}
        for item in items:
            group_key = item.get(key)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
        return groups

    @staticmethod
    def unique(items: List[Any], key: Optional[callable] = None) -> List[Any]:
        seen = set()
        result = []
        for item in items:
            item_key = key(item) if key else item
            if item_key not in seen:
                seen.add(item_key)
                result.append(item)
        return result

    @staticmethod
    def chunk(items: List[Any], size: int) -> List[List[Any]]:
        return [items[i:i + size] for i in range(0, len(items), size)]

    @staticmethod
    def sort_by_key(items: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
        return sorted(items, key=lambda x: x.get(key), reverse=reverse)

    @staticmethod
    def pluck(items: List[Dict[str, Any]], key: str) -> List[Any]:
        return [item.get(key) for item in items]

    @staticmethod
    def intersect(list1: List[Any], list2: List[Any]) -> List[Any]:
        return list(set(list1) & set(list2))

    @staticmethod
    def union(list1: List[Any], list2: List[Any]) -> List[Any]:
        return list(set(list1) | set(list2))


class StringUtils:
    @staticmethod
    def is_empty(s: str) -> bool:
        return s is None or len(s.strip()) == 0

    @staticmethod
    def trim(s: str) -> str:
        return s.strip() if s else ''

    @staticmethod
    def to_upper(s: str) -> str:
        return s.upper() if s else ''

    @staticmethod
    def to_lower(s: str) -> str:
        return s.lower() if s else ''

    @staticmethod
    def camel_case(s: str) -> str:
        words = re.sub(r'[^a-zA-Z0-9]', ' ', s).split()
        return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

    @staticmethod
    def snake_case(s: str) -> str:
        s = re.sub(r'[^a-zA-Z0-9]', '_', s)
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
        return s.lower()

    @staticmethod
    def contains(s: str, substr: str, case_insensitive: bool = False) -> bool:
        if s is None or substr is None:
            return False
        if case_insensitive:
            return substr.lower() in s.lower()
        return substr in s

    @staticmethod
    def truncate(s: str, max_length: int, suffix: str = '...') -> str:
        if s is None:
            return ''
        if len(s) <= max_length:
            return s
        return s[:max_length - len(suffix)] + suffix

    @staticmethod
    def generate_random(length: int = 16, chars: str = None) -> str:
        chars = chars or string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

    @staticmethod
    def generate_uuid() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def generate_short_id(length: int = 8) -> str:
        timestamp = str(int(time.time()))[-6:]
        random_str = StringUtils.generate_random(length - 6)
        return timestamp + random_str

    @staticmethod
    def md5(s: str) -> str:
        return hashlib.md5(s.encode('utf-8')).hexdigest()

    @staticmethod
    def sha256(s: str) -> str:
        return hashlib.sha256(s.encode('utf-8')).hexdigest()

    @staticmethod
    def base64_encode(s: str) -> str:
        return base64.b64encode(s.encode('utf-8')).decode('utf-8')

    @staticmethod
    def base64_decode(s: str) -> str:
        try:
            return base64.b64decode(s).decode('utf-8')
        except Exception:
            return ''

    @staticmethod
    def is_email(s: str) -> bool:
        pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        return bool(re.match(pattern, s or ''))

    @staticmethod
    def is_phone(s: str) -> bool:
        pattern = r'^1[3-9]\d{9}$'
        return bool(re.match(pattern, s or ''))


class SecurityUtils:
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        if filename is None:
            return ''
        sanitized = re.sub(r'[^\w\.\-]', '_', filename)
        sanitized = sanitized.replace('..', '.')
        return sanitized.strip('.')

    @staticmethod
    def sanitize_path(path: str) -> str:
        if path is None:
            return ''
        path = path.replace('..', '')
        path = re.sub(r'[^\w\./\-]', '', path)
        return path

    @staticmethod
    def validate_path(base_dir: str, target_path: str) -> bool:
        abs_base = os.path.abspath(base_dir)
        abs_target = os.path.abspath(target_path)
        return abs_target.startswith(abs_base + os.sep) or abs_target == abs_base

    @staticmethod
    def generate_token(length: int = 32) -> str:
        return StringUtils.generate_random(length)

    @staticmethod
    def generate_api_key() -> str:
        prefix = 'sk_'
        timestamp = str(int(time.time()))[-8:]
        random_part = StringUtils.generate_random(24)
        return prefix + timestamp + random_part


type_utils = TypeUtils
datetime_utils = DateTimeUtils
json_utils = JSONUtils
list_utils = ListUtils
string_utils = StringUtils
security_utils = SecurityUtils