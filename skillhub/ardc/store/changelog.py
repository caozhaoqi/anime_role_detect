#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新日志存储模块
提供系统更新日志的持久化存储功能
"""

import json
import os
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from pathlib import Path


class ChangelogEntry(BaseModel):
    """更新日志条目"""

    version: str
    title: str
    description: str
    changes: List[str]
    release_date: str
    author: Optional[str] = None
    is_major: bool = False
    affected_components: Optional[List[str]] = None


class ChangelogStore:
    """更新日志存储"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.changelog_file = self.data_dir / "changelog.json"
        self._init_data()

    def _init_data(self):
        """初始化数据文件"""
        if not self.changelog_file.exists():
            # 创建初始更新日志
            initial_data = [
                {
                    "version": "1.0.1",
                    "title": "系统初始化",
                    "description": "ARD Skill Hub 系统正式上线",
                    "changes": [
                        "✅ 技能仓库核心功能",
                        "✅ 用户认证系统",
                        "✅ 技能搜索与索引",
                        "✅ 版本管理系统",
                        "✅ API文档支持",
                    ],
                    "release_date": "2026-05-20",
                    "author": "System",
                    "is_major": True,
                    "affected_components": ["core", "api", "auth"],
                }
            ]
            self._save_data(initial_data)

    def _load_data(self) -> List[dict]:
        """加载更新日志数据"""
        if not self.changelog_file.exists():
            return []
        try:
            with open(self.changelog_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_data(self, data: List[dict]):
        """保存更新日志数据"""
        with open(self.changelog_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_entry(self, entry: ChangelogEntry) -> bool:
        """添加更新日志条目"""
        try:
            data = self._load_data()

            # 检查版本是否已存在
            for existing in data:
                if existing["version"] == entry.version:
                    raise ValueError(f"版本 {entry.version} 已存在")

            data.insert(0, entry.dict())
            self._save_data(data)
            return True
        except Exception as e:
            raise ValueError(f"添加更新日志失败: {str(e)}")

    def get_all_entries(self, limit: int = 20) -> List[ChangelogEntry]:
        """获取所有更新日志条目"""
        data = self._load_data()
        return [ChangelogEntry(**item) for item in data[:limit]]

    def get_entry_by_version(self, version: str) -> Optional[ChangelogEntry]:
        """根据版本号获取更新日志"""
        data = self._load_data()
        for item in data:
            if item["version"] == version:
                return ChangelogEntry(**item)
        return None

    def get_latest_entry(self) -> Optional[ChangelogEntry]:
        """获取最新的更新日志"""
        data = self._load_data()
        if data:
            return ChangelogEntry(**data[0])
        return None

    def get_entries_after_date(self, date: str) -> List[ChangelogEntry]:
        """获取指定日期之后的更新日志"""
        data = self._load_data()
        result = []
        for item in data:
            if item["release_date"] >= date:
                result.append(ChangelogEntry(**item))
        return result

    def get_entries_by_component(self, component: str) -> List[ChangelogEntry]:
        """获取影响特定组件的更新日志"""
        data = self._load_data()
        result = []
        for item in data:
            components = item.get("affected_components", [])
            if component in components:
                result.append(ChangelogEntry(**item))
        return result

    def delete_entry(self, version: str) -> bool:
        """删除指定版本的更新日志"""
        data = self._load_data()
        original_length = len(data)
        data = [item for item in data if item["version"] != version]
        if len(data) < original_length:
            self._save_data(data)
            return True
        return False
