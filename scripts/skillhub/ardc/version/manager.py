#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能版本管理模块
负责技能版本的发布、升级和回滚
"""

import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timezone

from ardc.store.metadata import SkillMetadata, VersionInfo
from ardc.store.utils import (
    parse_version,
    compare_versions,
    is_valid_version,
    serialize_datetime_fields,
    load_json_file,
)

logger = logging.getLogger(__name__)


class VersionManager:
    """版本管理器"""

    def __init__(self, data_path: str = None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path.home() / ".ardc" / "versions"

        self.data_path.mkdir(parents=True, exist_ok=True)
        self._versions: Dict[str, Dict[str, VersionInfo]] = self._load_versions()

    def _load_versions(self) -> Dict[str, Dict[str, VersionInfo]]:
        versions = {}

        for skill_dir in self.data_path.iterdir():
            if skill_dir.is_dir():
                skill_id = skill_dir.name
                versions[skill_id] = {}

                for version_file in skill_dir.glob("*.json"):
                    version = version_file.stem
                    try:
                        data = load_json_file(str(version_file))

                        if isinstance(data.get("metadata"), dict):
                            data["metadata"] = SkillMetadata(**data["metadata"])

                        versions[skill_id][version] = VersionInfo(**data)
                    except Exception as e:
                        logger.error(f"加载版本 {skill_id}-{version} 失败: {e}")

        return versions

    def _save_version(self, skill_id: str, version_info: VersionInfo):
        import json

        skill_dir = self.data_path / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)

        version_file = skill_dir / f"{version_info.version}.json"

        data = version_info.dict()
        data = serialize_datetime_fields(data)

        with open(version_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def release_version(self, metadata: SkillMetadata, release_notes: str = "") -> bool:
        try:
            skill_id = metadata.id
            version = metadata.version

            if not is_valid_version(version):
                logger.error(f"无效的版本号格式: {version}")
                return False

            if skill_id in self._versions and version in self._versions[skill_id]:
                logger.warning(f"警告: 版本 {version} 已存在，将被覆盖")

            version_info = VersionInfo(
                version=version,
                metadata=metadata,
                release_notes=release_notes,
                released_at=datetime.now(timezone.utc),
                download_count=0,
            )

            if skill_id not in self._versions:
                self._versions[skill_id] = {}
            self._versions[skill_id][version] = version_info

            self._save_version(skill_id, version_info)
            logger.info(f"技能 {skill_id} v{version} 发布成功")
            return True
        except Exception as e:
            logger.error(f"发布版本失败: {e}")
            return False

    def list_versions(self, skill_id: str) -> List[VersionInfo]:
        if skill_id not in self._versions:
            return []

        versions = list(self._versions[skill_id].values())
        versions.sort(key=lambda v: parse_version(v.version), reverse=True)
        return versions

    def get_version(self, skill_id: str, version: str) -> Optional[VersionInfo]:
        if skill_id not in self._versions:
            return None

        if version == "latest":
            return self.get_latest_version(skill_id)

        if version in self._versions[skill_id]:
            return self._versions[skill_id][version]

        return self._match_version_pattern(skill_id, version)

    def _match_version_pattern(self, skill_id: str, pattern: str) -> Optional[VersionInfo]:
        versions = self.list_versions(skill_id)

        if pattern.isdigit():
            for v in versions:
                if v.version.startswith(f"{pattern}."):
                    return v

        parts = pattern.split(".")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            for v in versions:
                if v.version.startswith(f"{pattern}."):
                    return v

        return None

    def get_latest_version(self, skill_id: str) -> Optional[VersionInfo]:
        versions = self.list_versions(skill_id)
        return versions[0] if versions else None

    def get_latest_stable_version(self, skill_id: str) -> Optional[VersionInfo]:
        versions = self.list_versions(skill_id)
        for v in versions:
            if v.metadata.status == "stable":
                return v
        return None

    def rollback(self, skill_id: str, target_version: str) -> bool:
        try:
            target_info = self.get_version(skill_id, target_version)
            if not target_info:
                logger.warning(f"未找到目标版本 {target_version}")
                return False

            rollback_info = {
                "skill_id": skill_id,
                "from_version": (
                    self.get_latest_version(skill_id).version
                    if self.get_latest_version(skill_id)
                    else None
                ),
                "to_version": target_version,
                "rolled_back_at": datetime.now(timezone.utc).isoformat(),
            }

            rollback_dir = self.data_path / skill_id / "rollbacks"
            rollback_dir.mkdir(parents=True, exist_ok=True)
            rollback_file = (
                rollback_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(rollback_file, "w", encoding="utf-8") as f:
                json.dump(rollback_info, f, ensure_ascii=False, indent=2)

            logger.info(f"技能 {skill_id} 已回滚到版本 {target_version}")
            return True
        except Exception as e:
            logger.error(f"回滚失败: {e}")
            return False

    def delete_version(self, skill_id: str, version: str) -> bool:
        try:
            if skill_id not in self._versions:
                logger.warning(f"技能 {skill_id} 不存在")
                return False

            if version not in self._versions[skill_id]:
                logger.warning(f"版本 {version} 不存在")
                return False

            version_file = self.data_path / skill_id / f"{version}.json"
            if version_file.exists():
                version_file.unlink()

            del self._versions[skill_id][version]

            if not self._versions[skill_id]:
                del self._versions[skill_id]
                skill_dir = self.data_path / skill_id
                if skill_dir.exists():
                    shutil.rmtree(skill_dir)

            logger.info(f"版本 {version} 删除成功")
            return True
        except Exception as e:
            logger.error(f"删除版本失败: {e}")
            return False

    def get_version_history(self, skill_id: str) -> List[Dict[str, str]]:
        history = []

        versions = self.list_versions(skill_id)
        for v in versions:
            history.append(
                {
                    "type": "release",
                    "version": v.version,
                    "timestamp": v.released_at.isoformat(),
                    "notes": v.release_notes,
                }
            )

        rollback_dir = self.data_path / skill_id / "rollbacks"
        if rollback_dir.exists():
            for rollback_file in sorted(rollback_dir.glob("*.json")):
                try:
                    with open(rollback_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        history.append(
                            {
                                "type": "rollback",
                                "from_version": data.get("from_version"),
                                "to_version": data.get("to_version"),
                                "timestamp": data.get("rolled_back_at"),
                            }
                        )
                except Exception as e:
                    logger.error(f"加载回滚记录失败: {e}")

        history.sort(key=lambda h: h["timestamp"])
        return history

    def compare_versions(self, version1: str, version2: str) -> int:
        """比较两个版本号"""
        return compare_versions(version1, version2)
