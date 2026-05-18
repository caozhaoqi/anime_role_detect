#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技能索引与搜索模块
提供技能的全文搜索、过滤和排序功能
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import re

from .metadata import SkillMetadata


class SkillIndex:
    """技能索引类"""
    
    def __init__(self, index_path: str = None):
        """
        初始化索引
        
        :param index_path: 索引文件路径，默认为 ~/.ardc/skill_index.json
        """
        if index_path:
            self.index_path = Path(index_path)
        else:
            self.index_path = Path.home() / ".ardc" / "skill_index.json"
        
        self._index: Dict[str, Any] = self._load_index()
        self._inverted_index: Dict[str, List[str]] = self._build_inverted_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """加载索引文件"""
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
        """保存索引文件"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index["last_updated"] = datetime.now().isoformat()
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)
    
    def _build_inverted_index(self) -> Dict[str, List[str]]:
        """构建倒排索引"""
        inverted = {}
        for skill_id, skill_data in self._index.get("skills", {}).items():
            # 从技能数据中提取关键词
            keywords = set()
            
            # ID和名称
            keywords.add(skill_id.lower())
            if 'name' in skill_data:
                keywords.update(self._tokenize(skill_data['name']))
            
            # 描述
            if 'description' in skill_data:
                keywords.update(self._tokenize(skill_data['description']))
            
            # 标签
            if 'tags' in skill_data:
                for tag in skill_data['tags']:
                    keywords.update(self._tokenize(tag))
            
            # 添加到倒排索引
            for keyword in keywords:
                if keyword not in inverted:
                    inverted[keyword] = []
                if skill_id not in inverted[keyword]:
                    inverted[keyword].append(skill_id)
        
        return inverted
    
    def _tokenize(self, text: str) -> List[str]:
        """
        将文本分词
        
        :param text: 输入文本
        :return: 分词结果列表
        """
        if not text:
            return []
        
        # 移除标点符号
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 分词（支持中英文）
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
        """
        添加技能到索引
        
        :param metadata: 技能元数据
        """
        skill_id = metadata.id
        
        # 转换为字典（处理datetime）
        skill_data = metadata.dict()
        for key in ['created_at', 'updated_at']:
            if isinstance(skill_data.get(key), datetime):
                skill_data[key] = skill_data[key].isoformat()
        
        # 添加到技能索引
        self._index["skills"][skill_id] = skill_data
        
        # 更新分类统计
        category = metadata.category
        if category not in self._index["categories"]:
            self._index["categories"][category] = 0
        self._index["categories"][category] += 1
        
        # 更新标签统计
        for tag in metadata.tags:
            if tag not in self._index["tags"]:
                self._index["tags"][tag] = 0
            self._index["tags"][tag] += 1
        
        # 重建倒排索引
        self._inverted_index = self._build_inverted_index()
        
        self._save_index()
    
    def remove_skill(self, skill_id: str):
        """
        从索引中移除技能
        
        :param skill_id: 技能ID
        """
        if skill_id not in self._index["skills"]:
            return
        
        # 获取技能数据
        skill_data = self._index["skills"][skill_id]
        
        # 更新分类统计
        category = skill_data.get("category")
        if category in self._index["categories"]:
            self._index["categories"][category] -= 1
            if self._index["categories"][category] <= 0:
                del self._index["categories"][category]
        
        # 更新标签统计
        for tag in skill_data.get("tags", []):
            if tag in self._index["tags"]:
                self._index["tags"][tag] -= 1
                if self._index["tags"][tag] <= 0:
                    del self._index["tags"][tag]
        
        # 移除技能
        del self._index["skills"][skill_id]
        
        # 重建倒排索引
        self._inverted_index = self._build_inverted_index()
        
        self._save_index()
    
    def search(self, keyword: str, category: str = None, 
               status: str = None, limit: int = 20) -> List[SkillMetadata]:
        """
        搜索技能
        
        :param keyword: 搜索关键词
        :param category: 分类筛选
        :param status: 状态筛选（development/testing/stable/deprecated）
        :param limit: 返回数量限制
        :return: 匹配的技能列表
        """
        if not keyword and not category and not status:
            return self.get_all_skills()[:limit]
        
        # 获取候选技能ID
        candidate_ids = set()
        
        if keyword:
            # 使用倒排索引搜索
            tokens = self._tokenize(keyword)
            if tokens:
                # 获取所有匹配token的技能ID
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
        
        # 过滤结果
        results = []
        for skill_id in candidate_ids:
            skill_data = self._index["skills"].get(skill_id)
            if not skill_data:
                continue
            
            # 分类过滤
            if category and skill_data.get("category") != category:
                continue
            
            # 状态过滤
            if status and skill_data.get("status") != status:
                continue
            
            # 转换为SkillMetadata对象
            metadata = self._dict_to_metadata(skill_data)
            results.append(metadata)
        
        # 排序（按更新时间降序）
        results.sort(key=lambda s: s.updated_at, reverse=True)
        
        return results[:limit]
    
    def _dict_to_metadata(self, data: Dict[str, Any]) -> SkillMetadata:
        """将字典转换为SkillMetadata对象"""
        data = data.copy()
        # 转换时间字段
        for key in ['created_at', 'updated_at']:
            if isinstance(data.get(key), str):
                data[key] = datetime.fromisoformat(data[key])
        return SkillMetadata(**data)
    
    def get_all_skills(self) -> List[SkillMetadata]:
        """获取所有技能"""
        results = []
        for skill_data in self._index["skills"].values():
            metadata = self._dict_to_metadata(skill_data)
            results.append(metadata)
        results.sort(key=lambda s: s.updated_at, reverse=True)
        return results
    
    def get_by_category(self, category: str) -> List[SkillMetadata]:
        """
        按分类获取技能
        
        :param category: 分类名称
        :return: 技能列表
        """
        results = []
        for skill_data in self._index["skills"].values():
            if skill_data.get("category") == category:
                metadata = self._dict_to_metadata(skill_data)
                results.append(metadata)
        results.sort(key=lambda s: s.updated_at, reverse=True)
        return results
    
    def get_categories(self) -> Dict[str, int]:
        """获取所有分类及其数量"""
        return self._index.get("categories", {})
    
    def get_tags(self) -> Dict[str, int]:
        """获取所有标签及其数量"""
        return self._index.get("tags", {})
    
    def get_suggestions(self, prefix: str) -> List[str]:
        """
        获取搜索建议
        
        :param prefix: 关键词前缀
        :return: 建议的关键词列表
        """
        suggestions = set()
        prefix_lower = prefix.lower()
        
        # 从技能ID和名称中获取建议
        for skill_data in self._index["skills"].values():
            skill_id = skill_data.get("id", "")
            name = skill_data.get("name", "")
            
            if skill_id.lower().startswith(prefix_lower):
                suggestions.add(skill_id)
            if name.lower().startswith(prefix_lower):
                suggestions.add(name)
            
            # 标签建议
            for tag in skill_data.get("tags", []):
                if tag.lower().startswith(prefix_lower):
                    suggestions.add(tag)
        
        return sorted(list(suggestions))[:10]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        total_skills = len(self._index["skills"])
        total_categories = len(self._index["categories"])
        total_tags = len(self._index["tags"])
        
        # 按状态统计
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