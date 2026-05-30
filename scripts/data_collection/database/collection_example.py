#!/usr/bin/env python3
"""
数据采集示例脚本
演示如何使用数据库存储原始URL、角色信息、艺术品和下载记录
支持SQLite和MySQL
"""

import os
import sys
import json
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_collection.database.database_functions import DatabaseManager


def test_collection(db_manager, db_type):
    """测试数据采集功能"""
    print(f"\n测试{db_type}数据库数据采集功能")

    # ==================== 示例1: 存储原始URL ====================
    print("\n1. 存储原始URL地址")

    # 模拟从API获取的URL列表
    raw_urls = [
        {
            "url": "https://example.com/images/character1_001.jpg",
            "source": "pixiv",
            "role_name": "可莉",
            "priority": 1,
            "metadata": {"width": 1024, "height": 1024, "type": "illustration"},
        },
        {
            "url": "https://example.com/images/character1_002.jpg",
            "source": "pixiv",
            "role_name": "可莉",
            "priority": 1,
            "metadata": {"width": 1200, "height": 800, "type": "illustration"},
        },
        {
            "url": "https://example.com/images/character2_001.jpg",
            "source": "danbooru",
            "role_name": "纳西妲",
            "priority": 2,
            "metadata": {"width": 1500, "height": 1000, "type": "fanart"},
        },
    ]

    # 批量添加URL
    for url_info in raw_urls:
        url_id = db_manager.add_raw_url(
            url=url_info["url"],
            source=url_info["source"],
            role_name=url_info["role_name"],
            priority=url_info["priority"],
            metadata=url_info["metadata"],
        )
        print(f"  添加URL: {url_info['url'][:50]}... -> ID: {url_id}")

    # ==================== 示例2: 存储艺术品信息 ====================
    print("\n2. 存储艺术品信息")

    artworks = [
        {
            "title": "可莉的冒险",
            "artist": "画师A",
            "source": "pixiv",
            "source_url": "https://pixiv.net/artworks/12345678",
            "original_url": "https://example.com/images/character1_001.jpg",
            "thumbnail_url": "https://example.com/thumbnails/character1_001.jpg",
            "tags": ["可莉", "原神", "loli", "fire"],
            "resolution": "1024x1024",
            "file_size": 1024000,
            "file_format": "jpg",
            "rating": "safe",
            "favorites": 1234,
            "views": 5678,
            "published_at": "2024-01-15",
        },
        {
            "title": "纳西妲的花园",
            "artist": "画师B",
            "source": "danbooru",
            "source_url": "https://danbooru.donmai.us/posts/1234567",
            "original_url": "https://example.com/images/character2_001.jpg",
            "thumbnail_url": "https://example.com/thumbnails/character2_001.jpg",
            "tags": ["纳西妲", "原神", "loli", "flower"],
            "resolution": "1500x1000",
            "file_size": 2048000,
            "file_format": "png",
            "rating": "safe",
            "favorites": 890,
            "views": 3456,
            "published_at": "2024-02-20",
        },
    ]

    artwork_ids = []
    for artwork in artworks:
        artwork_id = db_manager.add_artwork(
            title=artwork["title"],
            artist=artwork["artist"],
            source=artwork["source"],
            source_url=artwork["source_url"],
            original_url=artwork["original_url"],
            thumbnail_url=artwork["thumbnail_url"],
            tags=artwork["tags"],
            resolution=artwork["resolution"],
            file_size=artwork["file_size"],
            file_format=artwork["file_format"],
            rating=artwork["rating"],
            favorites=artwork["favorites"],
            views=artwork["views"],
            published_at=artwork["published_at"],
        )
        artwork_ids.append(artwork_id)
        print(f"  添加艺术品: {artwork['title']} -> ID: {artwork_id}")

    # ==================== 示例3: 更新URL关联艺术品 ====================
    print("\n3. 更新URL关联艺术品")

    # 获取之前添加的URL
    url1 = db_manager.get_url_by_url("https://example.com/images/character1_001.jpg")
    if url1:
        print(f"  URL已存在: {url1[1][:50]}...")

    # ==================== 示例4: 存储下载记录 ====================
    print("\n4. 存储下载记录")

    download_info = {
        "url_id": 1,  # 对应第一个URL
        "artwork_id": artwork_ids[0],  # 对应第一个艺术品
        "role_name": "可莉",
        "save_path": "/data/role_images/可莉/123456.jpg",
        "file_name": "123456.jpg",
        "download_status": "success",
        "http_status": 200,
        "file_size": 1024000,
        "download_time": 2.5,
    }

    record_id = db_manager.add_download_record(
        url_id=download_info["url_id"],
        artwork_id=download_info["artwork_id"],
        role_name=download_info["role_name"],
        save_path=download_info["save_path"],
        file_name=download_info["file_name"],
        download_status=download_info["download_status"],
        http_status=download_info["http_status"],
        file_size=download_info["file_size"],
        download_time=download_info["download_time"],
    )
    print(f"  添加下载记录 -> ID: {record_id}")

    # 更新URL状态为已下载
    db_manager.update_url_status(1, "downloaded")
    print(f"  更新URL状态为已下载")

    # ==================== 示例5: 查询统计信息 ====================
    print("\n5. 查询采集统计信息")
    stats = db_manager.get_collection_statistics()

    print(f"  总URL数: {stats.get('total_urls', 0)}")
    print(f"  待处理URL数: {stats.get('pending_urls', 0)}")
    print(f"  已下载URL数: {stats.get('downloaded_urls', 0)}")
    print(f"  失败URL数: {stats.get('failed_urls', 0)}")
    print(f"  艺术品总数: {stats.get('total_artworks', 0)}")
    print(f"  下载记录总数: {stats.get('total_downloads', 0)}")
    print(f"  成功下载数: {stats.get('success_downloads', 0)}")

    print("\n  各角色URL统计:")
    for role_name, count in stats.get("role_stats", [])[:5]:
        print(f"    {role_name}: {count} 条")

    # ==================== 示例6: 获取待处理URL ====================
    print("\n6. 获取待处理URL")
    pending_urls = db_manager.get_pending_urls(limit=5)
    print(f"  待处理URL数量: {len(pending_urls)}")
    for url in pending_urls:
        print(f"    ID: {url[0]}, URL: {url[1][:50]}..., 角色: {url[3]}, 来源: {url[2]}")


def main():
    """主函数"""
    print("=" * 60)
    print("数据采集示例脚本")
    print("=" * 60)

    # 测试SQLite
    print("\n" + "=" * 40)
    print("测试SQLite数据库")
    print("=" * 40)
    sqlite_db = DatabaseManager(db_type="sqlite")
    try:
        test_collection(sqlite_db, "SQLite")
    except Exception as e:
        print(f"SQLite测试失败: {e}")

    # 测试MySQL
    print("\n" + "=" * 40)
    print("测试MySQL数据库")
    print("=" * 40)
    mysql_db = DatabaseManager(db_type="mysql")

    if mysql_db.connect():
        try:
            test_collection(mysql_db, "MySQL")
        except Exception as e:
            print(f"MySQL测试失败: {e}")
        mysql_db.close()
    else:
        print("MySQL连接失败，可能是数据库未配置或mysql-connector-python未安装")

    print("\n" + "=" * 60)
    print("数据采集示例完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
