#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能注册中心单元测试
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from ardc.store.registry import SkillRegistry
from ardc.store.metadata import SkillMetadata


class TestSkillRegistry:
    """技能注册中心测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "registry.json"
        self.skills_dir = Path(self.temp_dir) / "skills"
        self.registry = SkillRegistry(str(self.registry_path))

    def teardown_method(self):
        """测试后清理"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_skill(self):
        """测试注册技能"""
        metadata = SkillMetadata(
            id="test_skill",
            name="Test Skill",
            version="1.0.0",
            description="Test description",
            author="Test Author",
            category="utility",
            entry_point="scripts/main.py",
            tags=["test", "demo"],
        )

        result = self.registry.register_skill(metadata)

        assert result is True
        assert "test_skill" in self.registry._registry
        assert "1.0.0" in self.registry._registry["test_skill"]

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

        self.registry.register_skill(metadata_v1)
        self.registry.register_skill(metadata_v2)

        latest = self.registry.get_latest_version("test_skill")

        assert latest is not None
        assert latest.version == "2.0.0"

    def test_get_skill_by_version(self):
        """测试按版本获取技能"""
        metadata = SkillMetadata(
            id="test_skill",
            name="Test Skill",
            version="1.0.0",
            description="Test",
            author="Author",
            category="utility",
            entry_point="main.py",
        )

        self.registry.register_skill(metadata)

        result = self.registry.get_skill_by_version("test_skill", "1.0.0")

        assert result is not None
        assert result.id == "test_skill"

    def test_search_skills(self):
        """测试技能搜索"""
        metadata1 = SkillMetadata(
            id="python_skill",
            name="Python Skill",
            version="1.0.0",
            description="Python test",
            author="Author",
            category="classifier",
            entry_point="main.py",
            tags=["python", "dev"],
        )

        metadata2 = SkillMetadata(
            id="java_skill",
            name="Java Skill",
            version="1.0.0",
            description="Java test",
            author="Author",
            category="classifier",
            entry_point="main.py",
            tags=["java", "dev"],
        )

        self.registry.register_skill(metadata1)
        self.registry.register_skill(metadata2)

        results = self.registry.search_skills(keyword="python")

        assert len(results) == 1
        assert results[0].id == "python_skill"
