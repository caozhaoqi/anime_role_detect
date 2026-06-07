"""
从角色名单文件加载角色到数据库
Load characters from role list file to database
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data_pipeline.database.init_db import Character, Base


def parse_role_line(line: str) -> dict:
    """解析角色行"""
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    
    result = {
        'name': parts[0],
        'series': parts[1],
        'aliases': [],
        'search_terms': []
    }
    
    # 提取别名和搜索词
    aliases = set()
    search_terms = set()
    
    for part in parts[2:]:
        aliases.add(part)
        search_terms.add(part)
    
    search_terms.add(result['name'])
    search_terms.add(f"{result['series']} {result['name']}")
    
    result['aliases'] = list(aliases)
    result['search_terms'] = list(search_terms)
    
    return result


def load_characters_from_file(file_path: str, db_url: str = "sqlite:///./data/data_pipeline.db"):
    """从文件加载角色"""
    # 创建引擎和会话
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    loaded_count = 0
    skipped_count = 0
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        data = parse_role_line(line)
        if not data:
            print(f"⚠️ 第{line_num}行解析失败: {line}")
            skipped_count += 1
            continue
        
        # 检查是否已存在
        existing = session.query(Character).filter_by(name=data['name']).first()
        if existing:
            print(f"⚠️ 角色已存在，跳过: {data['name']}")
            skipped_count += 1
            continue
        
        # 生成ID（从10000开始，避免与现有角色冲突）
        max_id = session.query(Character.id).order_by(Character.id.desc()).first()
        new_id = (max_id[0] + 1) if max_id else 10000
        
        character = Character(
            id=new_id,
            name=data['name'],
            series=data['series'],
            aliases=data['aliases'],
            search_terms=data['search_terms']
        )
        session.add(character)
        loaded_count += 1
        print(f"✅ 加载角色: {data['name']} (ID: {new_id})")
    
    session.commit()
    session.close()
    engine.dispose()
    
    print(f"\n🎉 加载完成！")
    print(f"   成功加载: {loaded_count} 个角色")
    print(f"   跳过/已存在: {skipped_count} 个角色")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从角色名单文件加载角色到数据库")
    parser.add_argument("-f", "--file", required=True, help="角色名单文件路径")
    parser.add_argument("-d", "--db", default="sqlite:///./data/data_pipeline.db", help="数据库URL")
    
    args = parser.parse_args()
    
    load_characters_from_file(args.file, args.db)