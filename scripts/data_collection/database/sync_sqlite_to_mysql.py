#!/usr/bin/env python3
"""
SQLite到MySQL数据同步脚本
将SQLite数据库中的数据迁移到MySQL数据库
"""

import os
import sys
import sqlite3

try:
    import mysql.connector
    from mysql.connector import Error
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

BATCH_SIZE = 100  # 每批提交的记录数

def load_mysql_config():
    """从.env文件加载MySQL配置"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env_file = os.path.join(project_root, '.env')
    
    config = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': '',
        'database': 'anime_role_db'
    }
    
    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('mysql_host'):
                    config['host'] = line.split('=', 1)[1].strip()
                elif line.startswith('mysql_port'):
                    config['port'] = int(line.split('=', 1)[1].strip())
                elif line.startswith('mysql_user'):
                    config['user'] = line.split('=', 1)[1].strip()
                elif line.startswith('mysql_password'):
                    config['password'] = line.split('=', 1)[1].strip()
                elif line.startswith('mysql_db'):
                    config['database'] = line.split('=', 1)[1].strip()
    
    return config

def get_sqlite_connection():
    """获取SQLite数据库连接"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    db_path = os.path.join(project_root, 'data', 'role_images.db')
    
    if not os.path.exists(db_path):
        print(f"错误: SQLite数据库文件不存在 - {db_path}")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        print(f"成功连接SQLite数据库: {db_path}")
        return conn
    except sqlite3.Error as e:
        print(f"SQLite连接错误: {e}")
        return None

def get_mysql_connection(config):
    """获取MySQL数据库连接"""
    if not MYSQL_AVAILABLE:
        print("错误: mysql-connector-python 未安装")
        return None
    
    try:
        conn = mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database=config['database'],
            autocommit=False
        )
        
        if conn.is_connected():
            print(f"成功连接MySQL数据库: {config['host']}:{config['port']}/{config['database']}")
            return conn
    except Error as e:
        print(f"MySQL连接错误: {e}")
        return None

def sync_table(sqlite_conn, mysql_conn, table_name, columns):
    """
    同步单个表的数据（批量提交）
    """
    sqlite_cursor = sqlite_conn.cursor()
    mysql_cursor = mysql_conn.cursor()
    
    # 获取记录总数
    try:
        sqlite_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_count = sqlite_cursor.fetchone()[0]
        print(f"\n表 {table_name}: 共 {total_count} 条记录")
        
        if total_count == 0:
            return 0, 0, 0
    except sqlite3.Error as e:
        print(f"  读取SQLite计数失败: {e}")
        return 0, 0, 0
    
    # 构建INSERT语句
    placeholders = ', '.join(['%s'] * len(columns))
    columns_str = ', '.join([f'`{col}`' if col in ['key', 'precision'] else col for col in columns])
    
    update_columns = []
    for col in columns:
        if col != 'id' and col not in ['created_at', 'updated_at']:
            update_columns.append(f'`{col}`=VALUES(`{col}`)' if col in ['key', 'precision'] else f'{col}=VALUES({col})')
    update_str = ', '.join(update_columns)
    
    if update_str:
        sql = f"""
        INSERT INTO {table_name} ({columns_str})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_str}
        """
    else:
        sql = f"""
        INSERT IGNORE INTO {table_name} ({columns_str})
        VALUES ({placeholders})
        """
    
    # 分批处理
    offset = 0
    inserted = 0
    updated = 0
    failed = 0
    
    while offset < total_count:
        try:
            sqlite_cursor.execute(f"SELECT {', '.join(columns)} FROM {table_name} LIMIT {BATCH_SIZE} OFFSET {offset}")
            rows = sqlite_cursor.fetchall()
            
            if not rows:
                break
            
            # 准备数据
            data = []
            for row in rows:
                processed_row = []
                for val in row:
                    if val is None:
                        processed_row.append(None)
                    elif isinstance(val, bytes):
                        processed_row.append(val.decode('utf-8', errors='replace'))
                    else:
                        processed_row.append(val)
                data.append(tuple(processed_row))
            
            # 批量执行
            mysql_cursor.executemany(sql, data)
            mysql_conn.commit()
            
            # 更新计数
            inserted += mysql_cursor.rowcount
            offset += BATCH_SIZE
            
            # 显示进度
            progress = min(100, int(offset / total_count * 100))
            print(f"  进度: {progress}% ({offset}/{total_count})", end='\r')
            
        except Error as e:
            mysql_conn.rollback()
            failed += len(rows)
            print(f"\n  批量插入失败: {e}")
            offset += BATCH_SIZE
        except Exception as e:
            mysql_conn.rollback()
            failed += len(rows)
            print(f"\n  未知错误: {e}")
            offset += BATCH_SIZE
    
    print(f"\n  完成: 处理 {inserted} 条, 失败 {failed} 条")
    return inserted, failed, 0

def main():
    """主函数"""
    print("=" * 60)
    print("SQLite到MySQL数据同步")
    print("=" * 60)
    
    # 获取数据库连接
    sqlite_conn = get_sqlite_connection()
    if not sqlite_conn:
        print("无法连接SQLite数据库")
        return
    
    mysql_config = load_mysql_config()
    mysql_conn = get_mysql_connection(mysql_config)
    if not mysql_conn:
        print("无法连接MySQL数据库")
        sqlite_conn.close()
        return
    
    total_processed = 0
    total_failed = 0
    
    # 同步顺序：先同步没有外键依赖的表
    tables_to_sync = [
        ('roles', ['id', 'name', 'display_name', 'origin', 'gender', 'age', 'hair_color', 'eye_color', 'created_at', 'updated_at']),
        ('images', ['id', 'file_path', 'file_name', 'width', 'height', 'format', 'image_hash', 'size', 'created_at', 'updated_at']),
        ('users', ['id', 'username', 'password_hash', 'email', 'full_name', 'role', 'created_at', 'updated_at', 'last_login']),
        ('configs', ['id', 'key', 'value', 'description', 'category', 'created_at', 'updated_at']),
        ('models', ['id', 'name', 'path', 'type', 'architecture', 'version', 'accuracy', 'precision', 'recall', 'f1_score', 'created_at', 'updated_at']),
        ('annotations', ['id', 'image_id', 'role_id', 'features', 'annotation_json', 'created_at', 'updated_at']),
        ('artworks', ['id', 'title', 'artist', 'source', 'source_url', 'original_url', 'thumbnail_url', 'tags', 'resolution', 'file_size', 'file_format', 'rating', 'favorites', 'views', 'published_at', 'metadata', 'created_at', 'updated_at']),
        ('raw_urls', ['id', 'url', 'source', 'role_name', 'artwork_id', 'status', 'priority', 'metadata', 'created_at', 'updated_at']),
        ('download_records', ['id', 'url_id', 'artwork_id', 'role_name', 'save_path', 'file_name', 'download_status', 'error_message', 'http_status', 'file_size', 'download_time', 'retry_count', 'created_at', 'updated_at']),
        ('image_records', ['id', 'user_id', 'image_id', 'role_id', 'classification', 'confidence', 'tags', 'notes', 'created_at', 'updated_at']),
        ('training_records', ['id', 'model_id', 'start_time', 'end_time', 'duration', 'epochs', 'batch_size', 'learning_rate', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'best_epoch', 'best_val_accuracy', 'notes', 'created_at']),
    ]
    
    for table_name, columns in tables_to_sync:
        processed, failed, _ = sync_table(sqlite_conn, mysql_conn, table_name, columns)
        total_processed += processed
        total_failed += failed
    
    # 关闭连接
    sqlite_conn.close()
    mysql_conn.close()
    
    print("\n" + "=" * 60)
    print("数据同步完成!")
    print(f"总计: 处理 {total_processed} 条, 失败 {total_failed} 条")
    print("=" * 60)

if __name__ == "__main__":
    main()