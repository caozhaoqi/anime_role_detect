#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色名称映射与识别机制
确保不同语言（中文、英文、日文、拼音）采集的角色都能被正确识别和下载
"""

import os
import sys
import re
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import ROLE_MAPPING


# 角色主数据结构
class RoleIdentifier:
    def __init__(self, name, en_name=None, jp_name=None, pinyin=None):
        """
        初始化角色标识符
        :param name: 中文名（主标识）
        :param en_name: 英文名
        :param jp_name: 日文名
        :param pinyin: 拼音
        """
        self.name = name.strip() if name else None
        self.en_name = en_name.strip() if en_name else None
        self.jp_name = jp_name.strip() if jp_name else None
        self.pinyin = pinyin.strip() if pinyin else None

    def get_all_identifiers(self):
        """获取该角色的所有可能标识符"""
        identifiers = []
        if self.name:
            identifiers.append(self.name)
        if self.en_name:
            identifiers.append(self.en_name)
            # 处理带空格的英文名（多种变体）
            identifiers.append(self.en_name.replace(" ", "_"))
            identifiers.append(self.en_name.replace(" ", ""))
            identifiers.append(self.en_name.replace(" ", "-"))
        if self.jp_name:
            identifiers.append(self.jp_name)
        if self.pinyin:
            identifiers.append(self.pinyin)
        return [i for i in identifiers if i]

    def get_search_patterns(self):
        """获取用于搜索文件的模式列表（按优先级排序）"""
        patterns = []
        identifiers = self.get_all_identifiers()

        for ident in identifiers:
            # 精确匹配
            patterns.append(ident)
            # 小写版本
            patterns.append(ident.lower())
            # 大写首字母版本
            patterns.append(ident.title())

        return patterns

    def __repr__(self):
        return f"RoleIdentifier(name='{self.name}', en='{self.en_name}', jp='{self.jp_name}', pinyin='{self.pinyin}')"


# 全局角色注册表
class RoleRegistry:
    def __init__(self):
        self.roles = {}  # name -> RoleIdentifier
        self.identifier_map = {}  # any_identifier -> name (反向映射)
        self.pinyin_map = {}  # pinyin -> name

    def register_role(self, name, en_name=None, jp_name=None, pinyin=None):
        """注册角色"""
        if not name:
            return None

        role = RoleIdentifier(name, en_name, jp_name, pinyin)
        self.roles[name] = role

        # 建立反向映射
        for ident in role.get_all_identifiers():
            if ident in self.identifier_map:
                if name not in self.identifier_map[ident]:
                    self.identifier_map[ident].append(name)
            else:
                self.identifier_map[ident] = [name]

        if pinyin:
            self.pinyin_map[pinyin] = name

        return role

    def find_by_identifier(self, identifier):
        """
        通过任意标识符查找角色
        :param identifier: 中文名、英文名、日文名或拼音
        :return: RoleIdentifier列表（可能多个）
        """
        if identifier in self.identifier_map:
            return [self.roles[name] for name in self.identifier_map[identifier]]

        normalized_id = identifier.lower().replace(" ", "").replace("_", "").replace("-", "")

        matches = []
        for ident, names in self.identifier_map.items():
            normalized_ident = ident.lower().replace(" ", "").replace("_", "").replace("-", "")
            if normalized_id in normalized_ident or normalized_ident in normalized_id:
                for name in names:
                    if name not in [r.name for r in matches]:
                        matches.append(self.roles[name])

        return matches

    def find_by_pinyin(self, pinyin):
        """通过拼音查找角色"""
        if pinyin in self.pinyin_map:
            name = self.pinyin_map[pinyin]
            return self.roles.get(name)
        return None

    def get_all_roles(self):
        """获取所有角色"""
        return list(self.roles.values())

    def get_role_count(self):
        """获取角色数量"""
        return len(self.roles)


# 文件搜索器
class RoleFileFinder:
    def __init__(self, url_dirs=None):
        self.url_dirs = url_dirs or [
            "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/href_url",
            "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url",
        ]
        self.cache = {}

    def find_url_files(self, role_identifier):
        """
        为角色查找所有可能的URL文件
        :param role_identifier: RoleIdentifier对象
        :return: URL文件路径列表
        """
        if not isinstance(role_identifier, RoleIdentifier):
            return []

        cache_key = role_identifier.name
        if cache_key in self.cache:
            return self.cache[cache_key]

        found_files = []
        patterns = role_identifier.get_search_patterns()

        for url_dir in self.url_dirs:
            if not os.path.exists(url_dir):
                continue

            for filename in os.listdir(url_dir):
                name_without_ext = os.path.splitext(filename)[0]

                for pattern in patterns:
                    if (
                        pattern.lower() in name_without_ext.lower()
                        or name_without_ext.lower() in pattern.lower()
                    ):
                        filepath = os.path.join(url_dir, filename)
                        if filepath not in found_files:
                            found_files.append(filepath)

        self.cache[cache_key] = found_files
        return found_files

    def has_url_file(self, role_identifier):
        """检查角色是否有URL文件"""
        return len(self.find_url_files(role_identifier)) > 0


# 统一采集管理器
class CollectionManager:
    def __init__(self):
        self.registry = RoleRegistry()
        self.file_finder = RoleFileFinder()

    def load_from_role_file(self, role_file_path):
        """从角色文件加载角色列表（空格分隔格式）"""
        if not os.path.exists(role_file_path):
            print(f"❌ 角色文件不存在: {role_file_path}")
            return

        with open(role_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 解析格式: 中文名 作品 英文名 日文名（空格分隔）
            parts = line.split()
            if len(parts) >= 1:
                name = parts[0]
                en_name = parts[2] if len(parts) > 2 else ""
                jp_name = parts[3] if len(parts) > 3 else ""

                # 从ROLE_MAPPING获取拼音
                pinyin = ""
                if name in ROLE_MAPPING:
                    pinyin = ROLE_MAPPING[name].get("pinyin", "")

                if name:
                    self.registry.register_role(name, en_name, jp_name, pinyin)

        print(f"✅ 已加载 {self.registry.get_role_count()} 个角色")

    def suggest_keywords(self, role_name):
        """为角色生成推荐的采集关键词"""
        role = self.registry.roles.get(role_name)
        if not role:
            return []

        keywords = []
        if role.en_name:
            keywords.append(role.en_name)
            if " " in role.en_name:
                keywords.append(role.en_name.replace(" ", ""))
        if role.name:
            keywords.append(role.name)
        if role.jp_name:
            keywords.append(role.jp_name)

        return keywords

    def get_role_download_info(self, role_name):
        """获取角色的下载信息"""
        role = self.registry.roles.get(role_name)
        if not role:
            return None

        url_files = self.file_finder.find_url_files(role)

        return {
            "name": role.name,
            "en_name": role.en_name,
            "jp_name": role.jp_name,
            "pinyin": role.pinyin,
            "identifiers": role.get_all_identifiers(),
            "url_files": url_files,
            "has_url_files": len(url_files) > 0,
        }

    def batch_get_download_info(self):
        """批量获取所有角色的下载信息"""
        results = []
        for role in self.registry.get_all_roles():
            results.append(self.get_role_download_info(role.name))
        return results


# 全局实例
manager = CollectionManager()


def init_manager(role_file_path):
    """初始化管理器"""
    manager.load_from_role_file(role_file_path)
    return manager


def find_role(identifier):
    """通过任意标识符查找角色"""
    return manager.registry.find_by_identifier(identifier)


def get_role_by_pinyin(pinyin):
    """通过拼音获取角色"""
    return manager.registry.find_by_pinyin(pinyin)


def get_role_download_info(role_name):
    """获取角色下载信息"""
    return manager.get_role_download_info(role_name)


def find_url_files_for_role(role_name):
    """查找角色的URL文件"""
    role = manager.registry.roles.get(role_name)
    if role:
        return manager.file_finder.find_url_files(role)
    return []


# 示例用法
if __name__ == "__main__":
    ROLE_FILE = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt"

    init_manager(ROLE_FILE)

    print("=== 测试角色查找 ===")

    results = find_role("阿洛娜")
    print(f"通过中文名'阿洛娜'找到: {[r.name for r in results]}")

    results = find_role("Arona")
    print(f"通过英文名'Arona'找到: {[r.name for r in results]}")

    role = get_role_by_pinyin("a1luo4na4")
    print(f"通过拼音'a1luo4na4'找到: {role.name if role else None}")

    print("\n=== 测试URL文件查找 ===")
    url_files = find_url_files_for_role("阿洛娜")
    print(f"阿洛娜的URL文件: {len(url_files)} 个")
    for f in url_files[:3]:
        print(f"  - {os.path.basename(f)}")

    print("\n=== 测试推荐关键词 ===")
    keywords = manager.suggest_keywords("鹿目圆")
    print(f"鹿目圆的推荐关键词: {keywords}")

    print("\n=== 测试角色信息 ===")
    info = get_role_download_info("阿洛娜")
    if info:
        print(f"阿洛娜的所有标识符: {info['identifiers']}")
        print(f"阿洛娜的拼音: {info['pinyin']}")
