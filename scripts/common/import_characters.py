#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从数据目录导入角色到数据库
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.database.init_db import Character, init_database


def import_characters_from_directory(data_dir: str = "data/final_dataset"):
    """从数据目录导入角色到数据库"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return 0
    
    # 初始化数据库
    engine, Session = init_database()
    db = Session()
    
    imported_count = 0
    skipped_count = 0
    
    print(f"\n📥 开始从 {data_dir} 导入角色...")
    
    # 遍历每个角色目录
    for item in data_path.iterdir():
        if not item.is_dir() or item.name.startswith('.'):
            continue
        
        character_name = item.name
        
        # 检查角色是否已存在
        existing = db.query(Character).filter_by(name=character_name).first()
        if existing:
            skipped_count += 1
            continue
        
        # 创建新角色记录
        character = Character(
            name=character_name,
            series="Unknown",  # 默认系列
            aliases=[],
            search_terms=[character_name]
        )
        
        db.add(character)
        imported_count += 1
        
        if imported_count % 10 == 0:
            print(f"已导入 {imported_count} 个角色...")
    
    # 提交事务
    db.commit()
    db.close()
    
    print(f"\n✅ 导入完成！新增: {imported_count} 个，跳过: {skipped_count} 个")
    return imported_count


if __name__ == "__main__":
    import_characters_from_directory()
