#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数单元测试
"""

import pytest
from datetime import datetime, timezone
from ardc.store.utils import (
    parse_version, compare_versions, is_valid_version,
    serialize_datetime_fields, deserialize_datetime_fields
)


class TestVersionUtils:
    """版本号处理工具测试"""
    
    def test_parse_version(self):
        """测试版本号解析"""
        assert parse_version("1.0.0") == (1, 0, 0, "1.0.0")
        assert parse_version("2.1.3") == (2, 1, 3, "2.1.3")
        assert parse_version("1.0.0-beta.1") == (1, 0, 0, "1.0.0-beta.1")
        assert parse_version("invalid") == (0, 0, 0, "invalid")
    
    def test_compare_versions(self):
        """测试版本号比较"""
        assert compare_versions("1.0.0", "1.0.0") == 0
        assert compare_versions("1.0.0", "1.0.1") == -1
        assert compare_versions("1.1.0", "1.0.0") == 1
        assert compare_versions("2.0.0", "1.9.9") == 1
    
    def test_is_valid_version(self):
        """测试版本号格式验证"""
        assert is_valid_version("1.0.0") is True
        assert is_valid_version("1.0.0-beta.1") is True
        assert is_valid_version("1.0.0+build.123") is True
        assert is_valid_version("1.0") is False
        assert is_valid_version("1") is False
        assert is_valid_version("v1.0.0") is False


class TestDateTimeUtils:
    """日期时间序列化工具测试"""
    
    def test_serialize_datetime_fields(self):
        """测试日期时间序列化"""
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        data = {
            "name": "test",
            "created_at": dt,
            "nested": {
                "updated_at": dt
            }
        }
        
        result = serialize_datetime_fields(data)
        
        assert result["created_at"] == "2024-01-15T10:30:00+00:00"
        assert result["nested"]["updated_at"] == "2024-01-15T10:30:00+00:00"
    
    def test_deserialize_datetime_fields(self):
        """测试日期时间反序列化"""
        data = {
            "name": "test",
            "created_at": "2024-01-15T10:30:00+00:00",
            "nested": {
                "updated_at": "2024-01-15T10:30:00+00:00"
            }
        }
        
        result = deserialize_datetime_fields(data)
        
        assert isinstance(result["created_at"], datetime)
        assert isinstance(result["nested"]["updated_at"], datetime)
        assert result["created_at"].year == 2024
