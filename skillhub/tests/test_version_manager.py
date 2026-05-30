#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
版本管理器单元测试
"""

import pytest
import tempfile
from pathlib import Path
from ardc.version.manager import VersionManager
from ardc.store.metadata import SkillMetadata


class TestVersionManager:
    """版本管理器测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir) / "versions"
        self.manager = VersionManager(str(self.data_path))

    def teardown_method(self):
        """测试后清理"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_release_version(self):
        """测试发布版本"""
        metadata = SkillMetadata(
            id="test_skill",
            name="Test Skill",
            version="1.0.0",
            description="Test",
            author="Author",
            category="utility",
            entry_point="main.py",
        )

        result = self.manager.release_version(metadata, "Initial release")

        assert result is True
        assert "test_skill" in self.manager._versions
        assert "1.0.0" in self.manager._versions["test_skill"]

    def test_list_versions(self):
        """测试列出版本"""
        metadata_v1 = SkillMetadata(
            id="test_skill",
            name="Test Skill",
            version="1.0.0",
            description="Test",
            author="Author",
            category="utility",
            entry_point="main.py",
        )

        metadata_v2 = SkillMetadata(
            id="test_skill",
            name="Test Skill",
            version="2.0.0",
            description="Test v2",
            author="Author",
            category="utility",
            entry_point="main.py",
        )

        self.manager.release_version(metadata_v1)
        self.manager.release_version(metadata_v2)

        versions = self.manager.list_versions("test_skill")

        assert len(versions) == 2
        assert versions[0].version == "2.0.0"
        assert versions[1].version == "1.0.0"

    def test_get_latest_version(self):
        """测试获取最新版本"""
        metadata_v1 = SkillMetadata(
            id="test_skill",
            name="Test Skill",
            version="1.0.0",
            description="Test",
            author="Author",
            category="utility",
            entry_point="main.py",
        )

        metadata_v2 = SkillMetadata(
            id="test_skill",
            name="Test Skill",
            version="2.0.0",
            description="Test v2",
            author="Author",
            category="utility",
            entry_point="main.py",
        )

        self.manager.release_version(metadata_v1)
        self.manager.release_version(metadata_v2)

        latest = self.manager.get_latest_version("test_skill")

        assert latest is not None
        assert latest.version == "2.0.0"

    def test_delete_version(self):
        """测试删除版本"""
        metadata = SkillMetadata(
            id="test_skill",
            name="Test Skill",
            version="1.0.0",
            description="Test",
            author="Author",
            category="utility",
            entry_point="main.py",
        )

        self.manager.release_version(metadata)

        result = self.manager.delete_version("test_skill", "1.0.0")

        assert result is True
        assert "test_skill" not in self.manager._versions

    def test_compare_versions(self):
        """测试版本比较"""
        assert self.manager.compare_versions("1.0.0", "1.0.0") == 0
        assert self.manager.compare_versions("1.0.0", "1.0.1") == -1
        assert self.manager.compare_versions("2.0.0", "1.0.0") == 1
