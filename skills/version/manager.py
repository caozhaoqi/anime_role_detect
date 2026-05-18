#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能版本管理模块
负责技能版本的发布、升级和回滚
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re

from skills.store.metadata import SkillMetadata, VersionInfo


class VersionManager:
    """版本管理器"""
    
    def __init__(self, data_path: str = None):
        """
        初始化版本管理器
        
        :param data_path: 版本数据存储路径，默认为 ~/.ardc/versions
        """
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path.home() / ".ardc" / "versions"
        
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self._versions: Dict[str, Dict[str, VersionInfo]] = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Dict[str, VersionInfo]]:
        """加载版本数据"""
        versions = {}
        
        for skill_dir in self.data_path.iterdir():
            if skill_dir.is_dir():
                skill_id = skill_dir.name
                versions[skill_id] = {}
                
                for version_file in skill_dir.glob("*.json"):
                    version = version_file.stem
                    try:
                        with open(version_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            # 转换时间字段
                            if 'released_at' in data:
                                data['released_at'] = datetime.fromisoformat(data['released_at'])
                            
                            if isinstance(data.get('metadata'), dict):
                                meta_data = data['metadata']
                                if 'created_at' in meta_data:
                                    meta_data['created_at'] = datetime.fromisoformat(meta_data['created_at'])
                                if 'updated_at' in meta_data:
                                    meta_data['updated_at'] = datetime.fromisoformat(meta_data['updated_at'])
                                data['metadata'] = SkillMetadata(**meta_data)
                            
                            versions[skill_id][version] = VersionInfo(**data)
                    except Exception as e:
                        print(f"加载版本 {skill_id}-{version} 失败: {e}")
        
        return versions
    
    def _save_version(self, skill_id: str, version_info: VersionInfo):
        """保存版本信息"""
        skill_dir = self.data_path / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)
        
        version_file = skill_dir / f"{version_info.version}.json"
        
        data = version_info.dict()
        
        # 转换datetime为字符串
        for key in ['released_at', 'created_at', 'updated_at']:
            if isinstance(data.get(key), datetime):
                data[key] = data[key].isoformat()
            if isinstance(data.get('metadata', {}).get(key), datetime):
                data['metadata'][key] = data['metadata'][key].isoformat()
        
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def release_version(self, metadata: SkillMetadata, release_notes: str = "") -> bool:
        """
        发布新版本
        
        :param metadata: 技能元数据（包含版本号）
        :param release_notes: 版本更新说明
        :return: 是否发布成功
        """
        try:
            skill_id = metadata.id
            version = metadata.version
            
            # 验证版本号格式
            if not self._is_valid_version(version):
                print(f"无效的版本号格式: {version}")
                return False
            
            # 检查版本是否已存在
            if skill_id in self._versions and version in self._versions[skill_id]:
                print(f"警告: 版本 {version} 已存在，将被覆盖")
            
            version_info = VersionInfo(
                version=version,
                metadata=metadata,
                release_notes=release_notes,
                released_at=datetime.now(),
                download_count=0
            )
            
            # 保存到内存
            if skill_id not in self._versions:
                self._versions[skill_id] = {}
            self._versions[skill_id][version] = version_info
            
            # 保存到文件
            self._save_version(skill_id, version_info)
            
            print(f"技能 {skill_id} v{version} 发布成功")
            return True
        except Exception as e:
            print(f"发布版本失败: {e}")
            return False
    
    def _is_valid_version(self, version: str) -> bool:
        """验证版本号格式是否符合语义化版本规范"""
        pattern = r'^(\d+)\.(\d+)\.(\d+)(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$'
        return re.match(pattern, version) is not None
    
    def list_versions(self, skill_id: str) -> List[VersionInfo]:
        """
        获取技能的所有版本
        
        :param skill_id: 技能ID
        :return: 版本信息列表（按版本号降序排列）
        """
        if skill_id not in self._versions:
            return []
        
        versions = list(self._versions[skill_id].values())
        versions.sort(key=lambda v: self._version_to_tuple(v.version), reverse=True)
        return versions
    
    def _version_to_tuple(self, version: str) -> Tuple[int, int, int, str]:
        """将版本号转换为元组以便排序"""
        try:
            # 移除预发布和构建元数据
            main_version = version.split('-')[0].split('+')[0]
            parts = main_version.split('.')
            return (int(parts[0]), int(parts[1]), int(parts[2]), version)
        except:
            return (0, 0, 0, version)
    
    def get_version(self, skill_id: str, version: str) -> Optional[VersionInfo]:
        """
        获取指定版本信息
        
        :param skill_id: 技能ID
        :param version: 版本号
        :return: 版本信息
        """
        if skill_id not in self._versions:
            return None
        
        # 支持语义化版本范围匹配
        if version == "latest":
            return self.get_latest_version(skill_id)
        
        if version in self._versions[skill_id]:
            return self._versions[skill_id][version]
        
        # 尝试模糊匹配（如 1.0 匹配 1.0.0, 1.0.1 等）
        return self._match_version_pattern(skill_id, version)
    
    def _match_version_pattern(self, skill_id: str, pattern: str) -> Optional[VersionInfo]:
        """根据版本模式匹配版本"""
        versions = self.list_versions(skill_id)
        
        # 匹配主版本号（如 1 匹配 1.x.x）
        if pattern.isdigit():
            for v in versions:
                if v.version.startswith(f"{pattern}."):
                    return v
        
        # 匹配主版本.次版本（如 1.0 匹配 1.0.x）
        parts = pattern.split('.')
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            for v in versions:
                if v.version.startswith(f"{pattern}."):
                    return v
        
        return None
    
    def get_latest_version(self, skill_id: str) -> Optional[VersionInfo]:
        """
        获取技能的最新版本
        
        :param skill_id: 技能ID
        :return: 最新版本信息
        """
        versions = self.list_versions(skill_id)
        return versions[0] if versions else None
    
    def get_latest_stable_version(self, skill_id: str) -> Optional[VersionInfo]:
        """
        获取技能的最新稳定版本
        
        :param skill_id: 技能ID
        :return: 最新稳定版本信息
        """
        versions = self.list_versions(skill_id)
        for v in versions:
            if v.metadata.status == "stable":
                return v
        return None
    
    def rollback(self, skill_id: str, target_version: str) -> bool:
        """
        回滚技能到指定版本
        
        :param skill_id: 技能ID
        :param target_version: 目标版本号
        :return: 是否回滚成功
        """
        try:
            # 获取目标版本信息
            target_info = self.get_version(skill_id, target_version)
            if not target_info:
                print(f"未找到目标版本 {target_version}")
                return False
            
            # 创建回滚记录
            rollback_info = {
                "skill_id": skill_id,
                "from_version": self.get_latest_version(skill_id).version if self.get_latest_version(skill_id) else None,
                "to_version": target_version,
                "rolled_back_at": datetime.now().isoformat()
            }
            
            # 保存回滚记录
            rollback_dir = self.data_path / skill_id / "rollbacks"
            rollback_dir.mkdir(parents=True, exist_ok=True)
            rollback_file = rollback_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(rollback_file, 'w', encoding='utf-8') as f:
                json.dump(rollback_info, f, ensure_ascii=False, indent=2)
            
            print(f"技能 {skill_id} 已回滚到版本 {target_version}")
            return True
        except Exception as e:
            print(f"回滚失败: {e}")
            return False
    
    def delete_version(self, skill_id: str, version: str) -> bool:
        """
        删除指定版本
        
        :param skill_id: 技能ID
        :param version: 版本号
        :return: 是否删除成功
        """
        try:
            if skill_id not in self._versions:
                print(f"技能 {skill_id} 不存在")
                return False
            
            if version not in self._versions[skill_id]:
                print(f"版本 {version} 不存在")
                return False
            
            # 删除版本文件
            version_file = self.data_path / skill_id / f"{version}.json"
            if version_file.exists():
                version_file.unlink()
            
            # 从内存中移除
            del self._versions[skill_id][version]
            
            # 如果没有版本了，删除技能目录
            if not self._versions[skill_id]:
                del self._versions[skill_id]
                skill_dir = self.data_path / skill_id
                if skill_dir.exists():
                    shutil.rmtree(skill_dir)
            
            print(f"版本 {version} 删除成功")
            return True
        except Exception as e:
            print(f"删除版本失败: {e}")
            return False
    
    def get_version_history(self, skill_id: str) -> List[Dict[str, str]]:
        """
        获取技能的版本历史记录（包括回滚记录）
        
        :param skill_id: 技能ID
        :return: 版本历史列表
        """
        history = []
        
        # 添加版本发布记录
        versions = self.list_versions(skill_id)
        for v in versions:
            history.append({
                "type": "release",
                "version": v.version,
                "timestamp": v.released_at.isoformat(),
                "notes": v.release_notes
            })
        
        # 添加回滚记录
        rollback_dir = self.data_path / skill_id / "rollbacks"
        if rollback_dir.exists():
            for rollback_file in sorted(rollback_dir.glob("*.json")):
                try:
                    with open(rollback_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        history.append({
                            "type": "rollback",
                            "from_version": data.get("from_version"),
                            "to_version": data.get("to_version"),
                            "timestamp": data.get("rolled_back_at")
                        })
                except Exception as e:
                    print(f"加载回滚记录失败: {e}")
        
        # 按时间排序
        history.sort(key=lambda h: h["timestamp"])
        return history
    
    def check_updates(self, current_version: str) -> Optional[str]:
        """
        检查是否有更新版本
        
        :param current_version: 当前版本号
        :return: 可用的更新版本号，如果没有更新则返回None
        """
        # 这个方法需要知道具体的技能ID
        # 通常在调用时需要传入技能ID
        # 这里提供一个通用的版本比较方法
        return None
    
    def compare_versions(self, version1: str, version2: str) -> int:
        """
        比较两个版本号
        
        :param version1: 版本号1
        :param version2: 版本号2
        :return: -1 (version1 < version2), 0 (相等), 1 (version1 > version2)
        """
        v1 = self._version_to_tuple(version1)
        v2 = self._version_to_tuple(version2)
        
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        else:
            return 0