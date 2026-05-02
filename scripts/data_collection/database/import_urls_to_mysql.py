#!/usr/bin/env python3
"""
将清洗后的角色图片URL导入到数据库
- 支持MySQL和SQLite两种数据库
- 从 spider_image_system/data/img_url/ 读取所有角色URL文件
- 将URL保存到 raw_urls 表中
- 自动处理重复URL
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_collection.database.database_functions import DatabaseManager
from spider_image_system.src.run.constants import PINYIN_MAPPING

def get_role_name_from_pinyin(pinyin: str) -> str:
    """从拼音获取角色名"""
    for name, py in PINYIN_MAPPING.items():
        if py == pinyin:
            return name
    return pinyin  # 如果找不到映射，返回拼音本身

def import_urls_to_db(db: DatabaseManager, url_dir: str, db_type: str):
    """将URL导入到数据库"""
    print(f"正在从目录读取URL文件: {url_dir}")
    
    total_urls = 0
    total_files = 0
    
    for filename in os.listdir(url_dir):
        if not filename.endswith('_img.txt'):
            continue
        
        total_files += 1
        pinyin_name = filename.replace('_img.txt', '')
        role_name = get_role_name_from_pinyin(pinyin_name)
        
        file_path = os.path.join(url_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            if not urls:
                print(f"  {filename}: 空文件，跳过")
                continue
            
            print(f"  {filename} ({role_name}): {len(urls)} URLs")
            
            # 批量添加URL
            success, fail = db.add_batch_urls(urls, source='pixiv', role_name=role_name)
            total_urls += success
            
        except Exception as e:
            print(f"  {filename}: 读取失败 - {e}")
    
    print(f"\n导入完成！共处理 {total_files} 个文件，成功导入 {total_urls} 条URL")

def main():
    """主函数"""
    print("=" * 60)
    print("将角色图片URL导入数据库")
    print("=" * 60)
    
    # URL文件目录
    url_dir = os.path.join(project_root, 'spider_image_system', 'data', 'img_url')
    
    if not os.path.exists(url_dir):
        print(f"❌ URL目录不存在: {url_dir}")
        return
    
    # 尝试连接MySQL
    print("尝试连接MySQL数据库...")
    db = DatabaseManager(db_type='mysql')
    
    if db.connect():
        db_type = 'MySQL'
        print("✅ MySQL数据库连接成功")
    else:
        # MySQL连接失败，使用SQLite
        print("⚠️ MySQL连接失败，切换到SQLite数据库")
        db = DatabaseManager(db_type='sqlite')
        db.connect()
        db_type = 'SQLite'
    
    # 导入URL
    import_urls_to_db(db, url_dir, db_type)
    
    # 获取统计信息
    stats = db.get_collection_statistics()
    print(f"\n📊 当前{db_type}数据库统计:")
    print(f"  总URL数: {stats.get('total_urls', 0)}")
    print(f"  待处理: {stats.get('pending_urls', 0)}")
    print(f"  已下载: {stats.get('downloaded_urls', 0)}")
    print(f"  失败: {stats.get('failed_urls', 0)}")
    
    if stats.get('role_stats'):
        print("\n📋 各角色URL统计（前10名）:")
        for role_name, count in stats['role_stats'][:10]:
            print(f"  {role_name}: {count} 条")
    
    db.close()
    
    print("\n" + "=" * 60)
    print(f"{db_type}数据库导入任务完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
