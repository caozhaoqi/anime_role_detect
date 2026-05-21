#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能注册中心
负责技能的注册、查询、安装和卸载管理
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .metadata import SkillMetadata, VersionInfo, InstalledSkill


class SkillRegistry:
    """技能注册中心"""
    
    def __init__(self, registry_path: str = None):
        if registry_path:
            self.registry_path = Path(registry_path)
        else:
            self.registry_path = Path.home() / ".ardc" / "registry.json"
        
        self.skills_dir = Path.home() / ".ardc" / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        
        self._registry = self._load_registry()
        self._installed_skills: Dict[str, InstalledSkill] = self._load_installed_skills()
    
    def _load_registry(self) -> Dict[str, Dict[str, VersionInfo]]:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for skill_id, versions in data.items():
                        for version, info in versions.items():
                            if 'released_at' in info:
                                info['released_at'] = datetime.fromisoformat(info['released_at'])
                            if isinstance(info.get('metadata'), dict):
                                if 'created_at' in info['metadata']:
                                    info['metadata']['created_at'] = datetime.fromisoformat(info['metadata']['created_at'])
                                if 'updated_at' in info['metadata']:
                                    info['metadata']['updated_at'] = datetime.fromisoformat(info['metadata']['updated_at'])
                    return data
            except Exception as e:
                print(f"加载注册表失败: {e}")
        return {}
    
    def _save_registry(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            data = {}
            for skill_id, versions in self._registry.items():
                data[skill_id] = {}
                for version, info in versions.items():
                    info_dict = info.dict() if hasattr(info, 'dict') else dict(info)
                    for key in ['released_at', 'created_at', 'updated_at']:
                        if isinstance(info_dict.get(key), datetime):
                            info_dict[key] = info_dict[key].isoformat()
                    data[skill_id][version] = info_dict
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_installed_skills(self) -> Dict[str, InstalledSkill]:
        installed = {}
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_id = skill_dir.name
                meta_file = skill_dir / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta_data = json.load(f)
                            if 'created_at' in meta_data:
                                meta_data['created_at'] = datetime.fromisoformat(meta_data['created_at'])
                            if 'updated_at' in meta_data:
                                meta_data['updated_at'] = datetime.fromisoformat(meta_data['updated_at'])
                            metadata = SkillMetadata(**meta_data)
                            
                            installed_info = {
                                "metadata": metadata,
                                "install_path": str(skill_dir),
                                "installed_at": datetime.now(),
                                "enabled": True,
                                "config": {}
                            }
                            installed[skill_id] = InstalledSkill(**installed_info)
                    except Exception as e:
                        print(f"加载已安装技能 {skill_id} 失败: {e}")
        return installed
    
    def register_skill(self, metadata: SkillMetadata, release_notes: str = "") -> bool:
        try:
            skill_id = metadata.id
            version = metadata.version
            
            if skill_id not in self._registry:
                self._registry[skill_id] = {}
            
            if version in self._registry[skill_id]:
                print(f"警告: 技能 {skill_id} 版本 {version} 已存在，将被覆盖")
            
            version_info = VersionInfo(
                version=version,
                metadata=metadata,
                release_notes=release_notes,
                released_at=datetime.now(),
                download_count=0
            )
            
            self._registry[skill_id][version] = version_info
            self._save_registry()
            return True
        except Exception as e:
            print(f"注册技能失败: {e}")
            return False
    
    def get_skill_versions(self, skill_id: str) -> List[VersionInfo]:
        if skill_id not in self._registry:
            return []
        versions = list(self._registry[skill_id].values())
        versions.sort(key=lambda v: self._parse_version(v.version), reverse=True)
        return versions
    
    def _parse_version(self, version_str: str) -> Tuple[int, int, int]:
        try:
            parts = version_str.split('.')
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        except:
            return (0, 0, 0)
    
    def get_latest_version(self, skill_id: str) -> Optional[VersionInfo]:
        versions = self.get_skill_versions(skill_id)
        return versions[0] if versions else None
    
    def get_skill_by_version(self, skill_id: str, version: str = None) -> Optional[SkillMetadata]:
        if skill_id not in self._registry:
            return None
        
        if version is None:
            latest = self.get_latest_version(skill_id)
            return latest.metadata if latest else None
        
        if version in self._registry[skill_id]:
            return self._registry[skill_id][version].metadata
        
        return None
    
    def search_skills(self, keyword: str = None, category: str = None) -> List[SkillMetadata]:
        results = []
        seen = set()
        
        for skill_id, versions in self._registry.items():
            latest = self.get_latest_version(skill_id)
            if not latest:
                continue
            
            metadata = latest.metadata
            
            if category and metadata.category != category:
                continue
            
            if keyword:
                keyword_lower = keyword.lower()
                matches = (
                    keyword_lower in metadata.id.lower() or
                    keyword_lower in metadata.name.lower() or
                    keyword_lower in metadata.description.lower() or
                    any(keyword_lower in tag.lower() for tag in metadata.tags)
                )
                if not matches:
                    continue
            
            if skill_id not in seen:
                results.append(metadata)
                seen.add(skill_id)
        
        return results
    
    def install_skill(self, skill_id: str, version: str = None) -> bool:
        try:
            metadata = self.get_skill_by_version(skill_id, version)
            
            if not metadata:
                from .index import SkillIndex
                index = SkillIndex()
                skills = index.search(skill_id)
                if skills:
                    metadata = skills[0]
                    print(f"从索引获取技能 {skill_id}")
            
            if not metadata:
                print(f"未找到技能 {skill_id} 的版本 {version or 'latest'}")
                return False
            
            if skill_id in self._installed_skills:
                print(f"技能 {skill_id} 已安装")
                return False
            
            skill_dir = self.skills_dir / skill_id
            skill_dir.mkdir(parents=True, exist_ok=True)
            
            meta_file = skill_dir / "metadata.json"
            with open(meta_file, 'w', encoding='utf-8') as f:
                meta_dict = metadata.dict()
                for key in ['created_at', 'updated_at']:
                    if isinstance(meta_dict.get(key), datetime):
                        meta_dict[key] = meta_dict[key].isoformat()
                json.dump(meta_dict, f, ensure_ascii=False, indent=2)
            
            entry_dir = skill_dir / "scripts"
            entry_dir.mkdir(exist_ok=True)
            
            entry_file = entry_dir / os.path.basename(metadata.entry_point)
            with open(entry_file, 'w', encoding='utf-8') as f:
                f.write(f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
{metadata.name} (v{metadata.version})
{metadata.description}
\"\"\"

def execute(**kwargs):
    print(f"执行 {metadata.id} v{metadata.version}")
    return {{"success": True, "message": "技能执行成功"}}

if __name__ == "__main__":
    execute()
""")
            entry_file.chmod(0o755)
            
            installed_info = InstalledSkill(
                metadata=metadata,
                install_path=str(skill_dir),
                installed_at=datetime.now(),
                enabled=True,
                config={}
            )
            self._installed_skills[skill_id] = installed_info
            
            version_key = version or metadata.version
            if skill_id in self._registry and version_key in self._registry[skill_id]:
                self._registry[skill_id][version_key].download_count += 1
                self._save_registry()
            
            print(f"技能 {skill_id} v{metadata.version} 安装成功")
            return True
        except Exception as e:
            print(f"安装技能失败: {e}")
            return False
    
    def uninstall_skill(self, skill_id: str) -> bool:
        try:
            if skill_id not in self._installed_skills:
                print(f"技能 {skill_id} 未安装")
                return False
            
            skill_dir = self.skills_dir / skill_id
            if skill_dir.exists():
                shutil.rmtree(skill_dir)
            
            del self._installed_skills[skill_id]
            print(f"技能 {skill_id} 卸载成功")
            return True
        except Exception as e:
            print(f"卸载技能失败: {e}")
            return False
    
    def list_installed_skills(self) -> List[InstalledSkill]:
        return list(self._installed_skills.values())
    
    def get_installed_skill(self, skill_id: str) -> Optional[InstalledSkill]:
        return self._installed_skills.get(skill_id)
    
    def enable_skill(self, skill_id: str, enabled: bool) -> bool:
        if skill_id not in self._installed_skills:
            print(f"技能 {skill_id} 未安装")
            return False
        
        self._installed_skills[skill_id].enabled = enabled
        print(f"技能 {skill_id} {'启用' if enabled else '禁用'}成功")
        return True