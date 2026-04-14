#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色名称采集器

优化角色名称的采集和验证策略
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

from .validators import CharacterNameValidator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('character_name_collector')


class CharacterNameCollector:
    """
    角色名称采集器
    负责从角色文件中提取和验证角色名称
    """
    
    def __init__(self):
        """初始化角色名称采集器"""
        self.validator = CharacterNameValidator()
    
    def is_character_name(self, text: str) -> bool:
        """
        判断文本是否为角色名称
        
        Args:
            text: 待判断的文本
            
        Returns:
            bool: 是否为角色名称
        """
        return self.validator.is_character_name(text)
    
    def load_characters_from_file(self, file_path: str) -> List[str]:
        """
        从文件加载角色名称
        
        Args:
            file_path: 文件路径
            
        Returns:
            角色名称列表
        """
        characters = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if self.is_character_name(line):
                        characters.append(line)
            
            logger.info(f"从 {file_path} 加载了 {len(characters)} 个角色名称")
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
        
        return characters
    
    def load_characters_from_directory(self, directory_path: str) -> Dict[str, List[str]]:
        """
        从目录加载所有角色文件
        
        Args:
            directory_path: 目录路径
            
        Returns:
            角色名称字典 {文件名: [角色名称]}
        """
        characters_dict = {}
        
        try:
            directory = Path(directory_path)
            if not directory.exists():
                logger.error(f"目录不存在: {directory_path}")
                return characters_dict
            
            for file_path in directory.iterdir():
                if file_path.is_file():
                    characters = self.load_characters_from_file(str(file_path))
                    if characters:
                        characters_dict[file_path.name] = characters
            
            total = sum(len(v) for v in characters_dict.values())
            logger.info(f"从 {directory_path} 加载了 {total} 个角色名称，共 {len(characters_dict)} 个文件")
        except Exception as e:
            logger.error(f"加载目录失败 {directory_path}: {e}")
        
        return characters_dict
    
    def save_characters_to_file(self, characters: List[str], file_path: str) -> bool:
        """
        保存角色名称到文件
        
        Args:
            characters: 角色名称列表
            file_path: 文件路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for character in characters:
                    f.write(f"{character}\n")
            
            logger.info(f"保存了 {len(characters)} 个角色名称到 {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存文件失败 {file_path}: {e}")
            return False
    
    def filter_characters(self, characters: List[str], min_length: int = 2, 
                         max_length: int = 30) -> List[str]:
        """
        过滤角色名称
        
        Args:
            characters: 角色名称列表
            min_length: 最小长度
            max_length: 最大长度
            
        Returns:
            过滤后的角色名称列表
        """
        filtered = []
        for character in characters:
            if min_length <= len(character) <= max_length:
                if self.is_character_name(character):
                    filtered.append(character)
        return filtered
    
    def merge_characters(self, characters_dict: Dict[str, List[str]]) -> List[str]:
        """
        合并多个文件的角色名称
        
        Args:
            characters_dict: 角色名称字典
            
        Returns:
            合并后的角色名称列表（去重）
        """
        merged = set()
        for characters in characters_dict.values():
            merged.update(characters)
        return sorted(list(merged))
    
    def get_statistics(self, characters_dict: Dict[str, List[str]]) -> Dict:
        """
        获取角色名称统计信息
        
        Args:
            characters_dict: 角色名称字典
            
        Returns:
            统计信息字典
        """
        total_files = len(characters_dict)
        total_characters = sum(len(v) for v in characters_dict.values())
        unique_characters = len(self.merge_characters(characters_dict))
        
        file_stats = {
            name: len(characters) 
            for name, characters in characters_dict.items()
        }
        
        return {
            'total_files': total_files,
            'total_characters': total_characters,
            'unique_characters': unique_characters,
            'file_stats': file_stats,
        }


# 便捷函数
def collect_characters_from_file(file_path: str) -> List[str]:
    """
    从文件采集角色名称
    
    Args:
        file_path: 文件路径
        
    Returns:
        角色名称列表
    """
    collector = CharacterNameCollector()
    return collector.load_characters_from_file(file_path)


def collect_characters_from_directory(directory_path: str) -> Dict[str, List[str]]:
    """
    从目录采集角色名称
    
    Args:
        directory_path: 目录路径
        
    Returns:
        角色名称字典
    """
    collector = CharacterNameCollector()
    return collector.load_characters_from_directory(directory_path)


def validate_character_name(text: str) -> bool:
    """
    验证文本是否为角色名称
    
    Args:
        text: 待验证的文本
        
    Returns:
        bool: 是否为角色名称
    """
    collector = CharacterNameCollector()
    return collector.is_character_name(text)


if __name__ == '__main__':
    # 测试代码
    collector = CharacterNameCollector()
    
    # 测试角色名称验证
    test_names = [
        "雷电将军",
        "胡桃",
        "钟离",
        "版本更新",
        "活动说明",
        "123",
        "test",
    ]
    
    for name in test_names:
        result = collector.is_character_name(name)
        print(f"{name}: {'是' if result else '否'}")
