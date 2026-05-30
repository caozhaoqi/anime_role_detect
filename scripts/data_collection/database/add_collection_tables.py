#!/usr/bin/env python3
"""
添加数据采集相关表
- 创建原始URL地址表 (raw_urls)
- 创建角色信息表增强 (roles_enhanced)
- 创建艺术品信息表 (artworks)
- 创建下载信息表 (download_records)
- 提供相关的CRUD操作
"""

import os
import json
import sqlite3
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="add_collection_tables.log",
    filemode="a",
)
logger = logging.getLogger(__name__)

# 全局配置
GLOBAL_CONFIG = {"database_file": "../../data/role_images.db"}


def get_database_path():
    """获取数据库文件路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, GLOBAL_CONFIG["database_file"])


def create_tables():
    """创建数据采集相关表"""
    logger.info("开始创建数据采集相关表")

    database_file = get_database_path()

    # 确保目录存在
    os.makedirs(os.path.dirname(database_file), exist_ok=True)

    # 连接数据库
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # 创建原始URL地址表
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS raw_urls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE NOT NULL,
        source TEXT,
        role_name TEXT,
        artwork_id INTEGER,
        status TEXT DEFAULT 'pending',
        priority INTEGER DEFAULT 1,
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (artwork_id) REFERENCES artworks (id)
    )
    """
    )
    logger.info("创建原始URL地址表完成")

    # 创建艺术品信息表
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS artworks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        artist TEXT,
        source TEXT,
        source_url TEXT,
        original_url TEXT,
        thumbnail_url TEXT,
        tags TEXT,
        resolution TEXT,
        file_size INTEGER,
        file_format TEXT,
        rating TEXT,
        favorites INTEGER DEFAULT 0,
        views INTEGER DEFAULT 0,
        published_at TEXT,
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (source, source_url)
    )
    """
    )
    logger.info("创建艺术品信息表完成")

    # 创建下载信息表
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS download_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url_id INTEGER,
        artwork_id INTEGER,
        role_name TEXT,
        save_path TEXT,
        file_name TEXT,
        download_status TEXT DEFAULT 'pending',
        error_message TEXT,
        http_status INTEGER,
        file_size INTEGER,
        download_time REAL,
        retry_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (url_id) REFERENCES raw_urls (id),
        FOREIGN KEY (artwork_id) REFERENCES artworks (id),
        UNIQUE (url_id, save_path)
    )
    """
    )
    logger.info("创建下载信息表完成")

    # 创建索引
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_urls_url ON raw_urls (url)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_urls_source ON raw_urls (source)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_urls_role_name ON raw_urls (role_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_urls_status ON raw_urls (status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_artworks_source ON artworks (source)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_artworks_artist ON artworks (artist)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_download_records_url_id ON download_records (url_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_download_records_role_name ON download_records (role_name)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_download_records_download_status ON download_records (download_status)"
    )
    logger.info("创建索引完成")

    # 提交更改
    conn.commit()
    conn.close()

    logger.info("数据采集相关表创建完成")


def add_raw_url(url, source=None, role_name=None, artwork_id=None, priority=1, metadata=None):
    """添加原始URL地址"""
    database_file = get_database_path()

    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
        INSERT OR IGNORE INTO raw_urls (url, source, role_name, artwork_id, priority, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                url,
                source,
                role_name,
                artwork_id,
                priority,
                json.dumps(metadata) if metadata else None,
            ),
        )

        conn.commit()

        # 获取插入的ID或已存在的ID
        cursor.execute("SELECT id FROM raw_urls WHERE url = ?", (url,))
        result = cursor.fetchone()
        return result[0] if result else None

    except Exception as e:
        logger.error(f"添加原始URL失败: {url} - {str(e)}")
        return None
    finally:
        conn.close()


def add_artwork(
    title=None,
    artist=None,
    source=None,
    source_url=None,
    original_url=None,
    thumbnail_url=None,
    tags=None,
    resolution=None,
    file_size=None,
    file_format=None,
    rating=None,
    favorites=0,
    views=0,
    published_at=None,
    metadata=None,
):
    """添加艺术品信息"""
    database_file = get_database_path()

    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
        INSERT OR IGNORE INTO artworks (title, artist, source, source_url, original_url, 
                                       thumbnail_url, tags, resolution, file_size, 
                                       file_format, rating, favorites, views, published_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                title,
                artist,
                source,
                source_url,
                original_url,
                thumbnail_url,
                json.dumps(tags) if tags else None,
                resolution,
                file_size,
                file_format,
                rating,
                favorites,
                views,
                published_at,
                json.dumps(metadata) if metadata else None,
            ),
        )

        conn.commit()

        # 获取插入的ID或已存在的ID
        cursor.execute(
            "SELECT id FROM artworks WHERE source = ? AND source_url = ?", (source, source_url)
        )
        result = cursor.fetchone()
        return result[0] if result else None

    except Exception as e:
        logger.error(f"添加艺术品失败: {title} - {str(e)}")
        return None
    finally:
        conn.close()


def add_download_record(
    url_id,
    artwork_id=None,
    role_name=None,
    save_path=None,
    file_name=None,
    download_status="pending",
    error_message=None,
    http_status=None,
    file_size=None,
    download_time=None,
    retry_count=0,
):
    """添加下载记录"""
    database_file = get_database_path()

    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
        INSERT INTO download_records (url_id, artwork_id, role_name, save_path, file_name, 
                                      download_status, error_message, http_status, 
                                      file_size, download_time, retry_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                url_id,
                artwork_id,
                role_name,
                save_path,
                file_name,
                download_status,
                error_message,
                http_status,
                file_size,
                download_time,
                retry_count,
            ),
        )

        conn.commit()
        return cursor.lastrowid

    except Exception as e:
        logger.error(f"添加下载记录失败: url_id={url_id} - {str(e)}")
        return None
    finally:
        conn.close()


def update_download_record(record_id, **kwargs):
    """更新下载记录"""
    database_file = get_database_path()

    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    try:
        set_clause = ", ".join([f"{key} = ?" for key in kwargs])
        values = list(kwargs.values()) + [record_id]
        sql = (
            f"UPDATE download_records SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
        )
        cursor.execute(sql, values)

        conn.commit()
        return cursor.rowcount > 0

    except Exception as e:
        logger.error(f"更新下载记录失败: record_id={record_id} - {str(e)}")
        return False
    finally:
        conn.close()


def update_raw_url_status(url_id, status):
    """更新URL状态"""
    database_file = get_database_path()

    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "UPDATE raw_urls SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, url_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    except Exception as e:
        logger.error(f"更新URL状态失败: url_id={url_id} - {str(e)}")
        return False
    finally:
        conn.close()


def get_pending_urls(limit=100):
    """获取待处理的URL列表"""
    database_file = get_database_path()

    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
        SELECT * FROM raw_urls 
        WHERE status = 'pending' 
        ORDER BY priority DESC, created_at ASC 
        LIMIT ?
        """,
            (limit,),
        )
        return cursor.fetchall()

    except Exception as e:
        logger.error(f"获取待处理URL失败: {str(e)}")
        return []
    finally:
        conn.close()


def get_url_by_url(url):
    """根据URL获取记录"""
    database_file = get_database_path()

    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM raw_urls WHERE url = ?", (url,))
        return cursor.fetchone()

    except Exception as e:
        logger.error(f"获取URL记录失败: {url} - {str(e)}")
        return None
    finally:
        conn.close()


def get_download_records_by_role(role_name):
    """获取角色的下载记录"""
    database_file = get_database_path()

    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
        SELECT dr.*, ru.url, a.title 
        FROM download_records dr 
        LEFT JOIN raw_urls ru ON dr.url_id = ru.id 
        LEFT JOIN artworks a ON dr.artwork_id = a.id 
        WHERE dr.role_name = ? 
        ORDER BY dr.created_at DESC
        """,
            (role_name,),
        )
        return cursor.fetchall()

    except Exception as e:
        logger.error(f"获取角色下载记录失败: {role_name} - {str(e)}")
        return []
    finally:
        conn.close()


def get_collection_statistics():
    """获取采集统计信息"""
    database_file = get_database_path()

    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    try:
        # URL统计
        cursor.execute("SELECT COUNT(*) FROM raw_urls")
        total_urls = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM raw_urls WHERE status = "pending"')
        pending_urls = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM raw_urls WHERE status = "downloaded"')
        downloaded_urls = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM raw_urls WHERE status = "failed"')
        failed_urls = cursor.fetchone()[0]

        # Artwork统计
        cursor.execute("SELECT COUNT(*) FROM artworks")
        total_artworks = cursor.fetchone()[0]

        # 下载记录统计
        cursor.execute("SELECT COUNT(*) FROM download_records")
        total_downloads = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM download_records WHERE download_status = "success"')
        success_downloads = cursor.fetchone()[0]

        # 按角色统计
        cursor.execute(
            """
        SELECT role_name, COUNT(*) as count 
        FROM raw_urls 
        WHERE role_name IS NOT NULL 
        GROUP BY role_name 
        ORDER BY count DESC
        """
        )
        role_stats = cursor.fetchall()

        return {
            "total_urls": total_urls,
            "pending_urls": pending_urls,
            "downloaded_urls": downloaded_urls,
            "failed_urls": failed_urls,
            "total_artworks": total_artworks,
            "total_downloads": total_downloads,
            "success_downloads": success_downloads,
            "role_stats": role_stats,
        }

    except Exception as e:
        logger.error(f"获取采集统计失败: {str(e)}")
        return {}
    finally:
        conn.close()


def add_batch_urls(urls, source=None, role_name=None):
    """批量添加URL"""
    database_file = get_database_path()

    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    success_count = 0
    fail_count = 0

    try:
        for url in urls:
            try:
                cursor.execute(
                    """
                INSERT OR IGNORE INTO raw_urls (url, source, role_name)
                VALUES (?, ?, ?)
                """,
                    (url, source, role_name),
                )
                if cursor.rowcount > 0:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                logger.warning(f"添加URL失败: {url} - {str(e)}")

        conn.commit()
        logger.info(f"批量添加URL完成: 成功 {success_count} 条，失败 {fail_count} 条")
        return success_count, fail_count

    except Exception as e:
        logger.error(f"批量添加URL失败: {str(e)}")
        return 0, len(urls)
    finally:
        conn.close()


def main():
    """主函数"""
    logger.info("============================================================")
    logger.info("开始创建数据采集相关表")
    logger.info("============================================================")

    # 创建表
    create_tables()

    # 输出统计信息
    stats = get_collection_statistics()
    logger.info("\n采集数据库统计信息:")
    logger.info(f"总URL数: {stats.get('total_urls', 0)}")
    logger.info(f"待处理URL数: {stats.get('pending_urls', 0)}")
    logger.info(f"已下载URL数: {stats.get('downloaded_urls', 0)}")
    logger.info(f"失败URL数: {stats.get('failed_urls', 0)}")
    logger.info(f"艺术品总数: {stats.get('total_artworks', 0)}")
    logger.info(f"下载记录总数: {stats.get('total_downloads', 0)}")
    logger.info(f"成功下载数: {stats.get('success_downloads', 0)}")

    if stats.get("role_stats"):
        logger.info("\n各角色URL统计:")
        for role_name, count in stats["role_stats"][:10]:
            logger.info(f"  {role_name}: {count} 条")

    logger.info("\n============================================================")
    logger.info("数据采集相关表创建完成")
    logger.info("============================================================")


if __name__ == "__main__":
    main()
