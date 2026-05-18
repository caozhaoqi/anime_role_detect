#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能索引与搜索模块
提供技能的全文搜索、过滤和排序功能
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import re

from .metadata import SkillMetadata


class SkillIndex:
    """技能索引类"""
    
    def __init__(self, index_path: str = None):
        if index_path:
            self.index_path = Path(index_path)
        else:
            self.index_path = Path.home() / ".ardc" / "skill_index.json"
        
        self._index: Dict[str, Any] = self._load_index()
        self._inverted_index: Dict[str, List[str]] = self._build_inverted_index()
    
    def _load_index(self) -> Dict[str, Any]:
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载索引失败: {e}")
        return {
            "skills": {},
            "categories": {},
            "tags": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_index(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index["last_updated"] = datetime.now().isoformat()
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)
    
    def _build_inverted_index(self) -> Dict[str, List[str]]:
        inverted = {}
        for skill_id, skill_data in self._index.get("skills", {}).items():
            keywords = set()
            keywords.add(skill_id.lower())
            if 'name' in skill_data:
                keywords.update(self._tokenize(skill_data['name']))
            if 'description' in skill_data:
                keywords.update(self._tokenize(skill_data['description']))
            if 'tags' in skill_data:
                for tag in skill_data['tags']:
                    keywords.update(self._tokenize(tag))
            
            for keyword in keywords:
                if keyword not in inverted:
                    inverted[keyword] = []
                if skill_id not in inverted[keyword]:
                    inverted[keyword].append(skill_id)
        
        return inverted
    
    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = []
        current_token = []
        
        for char in text.lower():
            if char.isalnum():
                current_token.append(char)
            else:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
        
        if current_token:
            tokens.append(''.join(current_token))
        
        return tokens
    
    def add_skill(self, metadata: SkillMetadata):
        skill_id = metadata.id
        
        skill_data = metadata.dict()
        for key in ['created_at', 'updated_at']:
            if isinstance(skill_data.get(key), datetime):
                skill_data[key] = skill_data[key].isoformat()
        
        self._index["skills"][skill_id] = skill_data
        
        category = metadata.category
        if category not in self._index["categories"]:
            self._index["categories"][category] = 0
        self._index["categories"][category] += 1
        
        for tag in metadata.tags:
            if tag not in self._index["tags"]:
                self._index["tags"][tag] = 0
            self._index["tags"][tag] += 1
        
        self._inverted_index = self._build_inverted_index()
        self._save_index()
    
    def remove_skill(self, skill_id: str):
        if skill_id not in self._index["skills"]:
            return
        
        skill_data = self._index["skills"][skill_id]
        
        category = skill_data.get("category")
        if category in self._index["categories"]:
            self._index["categories"][category] -= 1
            if self._index["categories"][category] <= 0:
                del self._index["categories"][category]
        
        for tag in skill_data.get("tags", []):
            if tag in self._index["tags"]:
                self._index["tags"][tag] -= 1
                if self._index["tags"][tag] <= 0:
                    del self._index["tags"][tag]
        
        del self._index["skills"][skill_id]
        self._inverted_index = self._build_inverted_index()
        self._save_index()
    
    def search(self, keyword: str, category: str = None, status: str = None, limit: int = 20) -> List[SkillMetadata]:
        if not keyword and not category and not status:
            return self.get_all_skills()[:limit]
        
        candidate_ids = set()
        
        if keyword:
            tokens = self._tokenize(keyword)
            if tokens:
                for token in tokens:
                    if token in self._inverted_index:
                        if not candidate_ids:
                            candidate_ids = set(self._inverted_index[token])
                        else:
                            candidate_ids.intersection_update(self._inverted_index[token])
            else:
                candidate_ids = set(self._index["skills"].keys())
        else:
            candidate_ids = set(self._index["skills"].keys())
        
        results = []
        for skill_id in candidate_ids:
            skill_data = self._index["skills"].get(skill_id)
            if not skill_data:
                continue
            
            if category and skill_data.get("category") != category:
                continue
            
            if status and skill_data.get("status") != status:
                continue
            
            metadata = self._dict_to_metadata(skill_data)
            results.append(metadata)
        
        results.sort(key=lambda s: s.updated_at, reverse=True)
        return results[:limit]
    
    def _dict_to_metadata(self, data: Dict[str, Any]) -> SkillMetadata:
        data = data.copy()
        for key in ['created_at', 'updated_at']:
            if isinstance(data.get(key), str):
                data[key] = datetime.fromisoformat(data[key])
        return SkillMetadata(**data)
    
    def get_all_skills(self) -> List[SkillMetadata]:
        results = []
        for skill_data in self._index["skills"].values():
            metadata = self._dict_to_metadata(skill_data)
            results.append(metadata)
        results.sort(key=lambda s: s.updated_at, reverse=True)
        return results
    
    def get_by_category(self, category: str) -> List[SkillMetadata]:
        results = []
        for skill_data in self._index["skills"].values():
            if skill_data.get("category") == category:
                metadata = self._dict_to_metadata(skill_data)
                results.append(metadata)
        results.sort(key=lambda s: s.updated_at, reverse=True)
        return results
    
    def get_categories(self) -> Dict[str, int]:
        return self._index.get("categories", {})
    
    def get_tags(self) -> Dict[str, int]:
        return self._index.get("tags", {})
    
    def get_suggestions(self, prefix: str) -> List[str]:
        suggestions = set()
        prefix_lower = prefix.lower()
        
        for skill_data in self._index["skills"].values():
            skill_id = skill_data.get("id", "")
            name = skill_data.get("name", "")
            
            if skill_id.lower().startswith(prefix_lower):
                suggestions.add(skill_id)
            if name.lower().startswith(prefix_lower):
                suggestions.add(name)
            
            for tag in skill_data.get("tags", []):
                if tag.lower().startswith(prefix_lower):
                    suggestions.add(tag)
        
        return sorted(list(suggestions))[:10]
    
    def get_statistics(self) -> Dict[str, Any]:
        total_skills = len(self._index["skills"])
        total_categories = len(self._index["categories"])
        total_tags = len(self._index["tags"])
        
        status_counts = {}
        for skill_data in self._index["skills"].values():
            status = skill_data.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_skills": total_skills,
            "total_categories": total_categories,
            "total_tags": total_tags,
            "status_counts": status_counts,
            "categories": self._index["categories"],
            "last_updated": self._index.get("last_updated")
        }