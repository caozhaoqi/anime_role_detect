#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入所有txt文件中的URL到数据库
按照文件名创建角色记录
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from pathlib import Path

# 配置 - 扫描所有包含URL的目录
SPIDER_DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data'
ROLE_LIST_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
DATABASE_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/spider_data.db'


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """连接数据库"""
        try:
            os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
            self.conn = sqlite3.connect(self.db_file)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return False
    
    def execute_query(self, query, params=None):
        """执行查询"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"查询执行失败: {e}")
            return []
    
    def execute_update(self, query, params=None):
        """执行更新"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.rowcount
        except Exception as e:
            print(f"更新执行失败: {e}")
            return 0
    
    def commit(self):
        """提交事务"""
        if self.conn:
            self.conn.commit()
    
    def rollback(self):
        """回滚事务"""
        if self.conn:
            self.conn.rollback()
    
    def close(self):
        """关闭连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


def load_role_list():
    """加载角色列表"""
    roles = []
    with open(ROLE_LIST_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                roles.append({
                    'cn_name': parts[0],
                    'game': parts[1],
                    'en_name': parts[2],
                    'jp_name': parts[3] if len(parts) > 3 else ''
                })
    return roles


def normalize_role_name(name):
    """标准化角色名称"""
    name = name.replace('_zip', '').replace('_img', '').replace('.txt', '')
    name = name.replace('_url', '').replace('_result', '')
    try:
        from urllib.parse import unquote
        name = unquote(name)
    except:
        pass
    return name


def find_matching_role(role_name, roles):
    """查找匹配的角色"""
    normalized = normalize_role_name(role_name).lower()
    
    for role in roles:
        if (normalized == role['en_name'].lower() or
            normalized == role['cn_name'] or
            normalized == role['jp_name']):
            return role
    
    for role in roles:
        if (role['en_name'].lower() in normalized or
            role['cn_name'] in normalized or
            role['jp_name'] in normalized):
            return role
    
    return None


def ensure_role_exists(db_manager, role_name, roles):
    """确保角色存在，如果不存在则创建"""
    normalized_name = normalize_role_name(role_name)
    
    # 先尝试匹配现有角色
    matched_role = find_matching_role(role_name, roles)
    
    if matched_role:
        # 检查是否已在数据库中
        existing = db_manager.execute_query(
            "SELECT id FROM roles WHERE name = ?",
            (matched_role['en_name'],)
        )
        
        if existing:
            return existing[0][0], matched_role
        else:
            # 插入角色
            db_manager.execute_update(
                """INSERT INTO roles (name, display_name, origin) 
                   VALUES (?, ?, ?)""",
                (matched_role['en_name'], matched_role['cn_name'], matched_role['game'])
            )
            role_id = db_manager.execute_query("SELECT last_insert_rowid()")[0][0]
            return role_id, matched_role
    
    # 未匹配到角色，创建新角色记录
    existing = db_manager.execute_query(
        "SELECT id FROM roles WHERE name = ?",
        (normalized_name,)
    )
    
    if existing:
        return existing[0][0], None
    
    # 创建新角色
    db_manager.execute_update(
        """INSERT INTO roles (name, display_name, origin) 
           VALUES (?, ?, ?)""",
        (normalized_name, normalized_name, 'unknown')
    )
    role_id = db_manager.execute_query("SELECT last_insert_rowid()")[0][0]
    
    return role_id, None


def import_all_url_files(db_manager, roles):
    """导入所有URL文件数据"""
    print("=" * 60)
    print("导入所有URL文件数据")
    print("=" * 60)
    
    total_urls = 0
    imported_files = 0
    skipped_files = 0
    new_roles = 0
    processed_files = set()
    
    # 找出所有包含URL文件的目录
    url_dirs = []
    for entry in os.listdir(SPIDER_DATA_DIR):
        entry_path = os.path.join(SPIDER_DATA_DIR, entry)
        if os.path.isdir(entry_path) and entry.lower().endswith('_url'):
            url_dirs.append(entry_path)
    
    print(f"发现 {len(url_dirs)} 个URL目录:")
    for url_dir in url_dirs:
        print(f"  - {os.path.basename(url_dir)}")
    
    for url_dir in url_dirs:
        if not os.path.exists(url_dir):
            print(f"目录不存在: {url_dir}")
            continue
        
        print(f"\n处理目录: {url_dir}")
        
        for filename in sorted(os.listdir(url_dir)):
            if not filename.endswith('.txt'):
                continue
            
            # 避免重复处理同名文件
            file_key = (url_dir, filename)
            if file_key in processed_files:
                continue
            processed_files.add(file_key)
            
            file_path = os.path.join(url_dir, filename)
            role_name = filename.replace('_zip.txt', '').replace('_img.txt', '').replace('_url.txt', '').replace('_result_url.txt', '').replace('.txt', '')
            
            # 确定数据源类型
            source_type = os.path.basename(url_dir)
            
            # 确保角色存在
            role_id, matched_role = ensure_role_exists(db_manager, role_name, roles)
            
            if matched_role:
                role_display_name = matched_role['cn_name']
                role_en_name = matched_role['en_name']
            else:
                role_display_name = role_name
                role_en_name = normalize_role_name(role_name)
                new_roles += 1
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    urls = [line.strip() for line in f if line.strip()]
                
                if not urls:
                    print(f"  ⚠ 跳过空文件: {filename}")
                    skipped_files += 1
                    continue
                
                imported_count = 0
                for url in urls:
                    try:
                        existing = db_manager.execute_query(
                            "SELECT id FROM raw_urls WHERE url = ?",
                            (url,)
                        )
                        
                        if not existing:
                            metadata = {
                                'source_file': filename,
                                'source_dir': os.path.basename(url_dir),
                                'imported_at': datetime.now().isoformat()
                            }
                            
                            if matched_role:
                                metadata.update({
                                    'cn_name': matched_role['cn_name'],
                                    'game': matched_role['game']
                                })
                            
                            db_manager.execute_update(
                                """INSERT INTO raw_urls (url, source, role_name, status, metadata) 
                                   VALUES (?, ?, ?, ?, ?)""",
                                (url, source_type, role_en_name, 'pending',
                                 json.dumps(metadata, ensure_ascii=False))
                            )
                            total_urls += 1
                            imported_count += 1
                    except Exception as e:
                        print(f"    ✗ 插入URL失败: {e}")
                
                if imported_count > 0:
                    imported_files += 1
                    print(f"  ✓ {filename}: {imported_count}/{len(urls)} 个URL (角色: {role_display_name})")
                else:
                    skipped_files += 1
                    print(f"  ⚠ 跳过已导入: {filename}")
                
            except Exception as e:
                print(f"  ✗ 读取文件失败 {filename}: {e}")
    
    print(f"\n导入文件数: {imported_files}")
    print(f"跳过文件数: {skipped_files}")
    print(f"新建角色数: {new_roles}")
    print(f"导入URL总数: {total_urls}")
    return total_urls


def generate_import_report(db_manager):
    """生成导入报告"""
    print("\n" + "=" * 60)
    print("导入统计报告")
    print("=" * 60)
    
    # 角色统计
    role_count = db_manager.execute_query("SELECT COUNT(*) FROM roles")[0][0]
    print(f"角色总数: {role_count}")
    
    # URL统计
    url_stats = db_manager.execute_query(
        "SELECT status, COUNT(*) FROM raw_urls GROUP BY status"
    )
    print("\nURL统计:")
    for status, count in url_stats:
        print(f"  {status}: {count}")
    
    # 按数据源统计
    print("\n按数据源统计:")
    source_stats = db_manager.execute_query(
        "SELECT source, COUNT(*) FROM raw_urls GROUP BY source"
    )
    for source, count in source_stats:
        print(f"  {source}: {count}")
    
    # 按角色统计URL数量
    print("\n各角色URL数量（前20）:")
    role_url_stats = db_manager.execute_query(
        """SELECT role_name, COUNT(*) as url_count
           FROM raw_urls
           GROUP BY role_name
           ORDER BY url_count DESC
           LIMIT 20"""
    )
    for role_name, count in role_url_stats:
        print(f"  {role_name}: {count}")


def main():
    print("=" * 60)
    print("导入所有URL文件到数据库")
    print("=" * 60)
    print(f"数据源目录: {SPIDER_DATA_DIR}")
    print(f"数据库文件: {DATABASE_FILE}")
    print(f"角色列表: {ROLE_LIST_PATH}")
    print("=" * 60)
    
    db_manager = DatabaseManager(DATABASE_FILE)
    if not db_manager.connect():
        print("数据库连接失败")
        return
    
    try:
        # 创建表（如果不存在）
        print("\n创建数据库表...")
        db_manager.execute_update("""
            CREATE TABLE IF NOT EXISTS roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                display_name TEXT,
                origin TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        db_manager.execute_update("""
            CREATE TABLE IF NOT EXISTS raw_urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                source TEXT,
                role_name TEXT,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 1,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        print("✓ 数据库表创建完成")
        
        # 加载角色列表
        print("\n加载角色列表...")
        roles = load_role_list()
        print(f"✓ 加载 {len(roles)} 个角色")
        
        # 导入所有URL文件
        import_all_url_files(db_manager, roles)
        
        # 生成报告
        generate_import_report(db_manager)
        
        # 提交事务
        db_manager.commit()
        
        print("\n" + "=" * 60)
        print("✓ 数据导入完成")
        print("=" * 60)
        print(f"数据库位置: {DATABASE_FILE}")
        
    except Exception as e:
        print(f"\n✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        db_manager.rollback()
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()