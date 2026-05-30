#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将spider_image_system数据导入数据库
支持数据源追溯
"""

import os
import sys
import json
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

# 配置
SPIDER_DATA_DIR = (
    "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/src/run/data"
)
ROLE_LIST_PATH = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt"
DATABASE_FILE = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/spider_data.db"


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
    with open(ROLE_LIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                roles.append(
                    {
                        "cn_name": parts[0],
                        "game": parts[1],
                        "en_name": parts[2],
                        "jp_name": parts[3] if len(parts) > 3 else "",
                    }
                )
    return roles


def normalize_role_name(name):
    """标准化角色名称"""
    name = name.replace("_zip", "").replace(".txt", "")
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
        if (
            normalized == role["en_name"].lower()
            or normalized == role["cn_name"]
            or normalized == role["jp_name"]
        ):
            return role

    for role in roles:
        if (
            role["en_name"].lower() in normalized
            or role["cn_name"] in normalized
            or role["jp_name"] in normalized
        ):
            return role

    return None


def calculate_file_hash(file_path):
    """计算文件哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def import_roles(db_manager, roles):
    """导入角色数据"""
    print("=" * 60)
    print("导入角色数据")
    print("=" * 60)

    imported = 0
    for role in roles:
        try:
            existing = db_manager.execute_query(
                "SELECT id FROM roles WHERE name = ?", (role["en_name"],)
            )

            if not existing:
                db_manager.execute_update(
                    """INSERT INTO roles (name, display_name, origin) 
                       VALUES (?, ?, ?)""",
                    (role["en_name"], role["cn_name"], role["game"]),
                )
                imported += 1
                print(f"  ✓ {role['cn_name']} ({role['en_name']})")
        except Exception as e:
            print(f"  ✗ 导入角色失败 {role['cn_name']}: {e}")

    print(f"\n导入角色数: {imported}")
    return imported


def import_url_files(db_manager, roles):
    """导入URL文件数据"""
    print("\n" + "=" * 60)
    print("导入URL文件数据")
    print("=" * 60)

    href_url_dir = os.path.join(SPIDER_DATA_DIR, "href_url")
    if not os.path.exists(href_url_dir):
        print("URL文件目录不存在")
        return 0

    total_urls = 0
    imported_files = 0

    for filename in os.listdir(href_url_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(href_url_dir, filename)
        role_name = filename.replace("_zip.txt", "").replace(".txt", "")

        matched_role = find_matching_role(role_name, roles)
        if not matched_role:
            print(f"  ⚠ 跳过未匹配角色: {filename}")
            continue

        role_id = db_manager.execute_query(
            "SELECT id FROM roles WHERE name = ?", (matched_role["en_name"],)
        )

        if not role_id:
            print(f"  ⚠ 角色未在数据库中: {matched_role['en_name']}")
            continue

        role_id = role_id[0][0]

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                urls = [line.strip() for line in f if line.strip()]

            for url in urls:
                try:
                    existing = db_manager.execute_query(
                        "SELECT id FROM raw_urls WHERE url = ?", (url,)
                    )

                    if not existing:
                        db_manager.execute_update(
                            """INSERT INTO raw_urls (url, source, role_name, status, metadata) 
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                url,
                                "href_url",
                                matched_role["en_name"],
                                "pending",
                                json.dumps(
                                    {
                                        "source_file": filename,
                                        "imported_at": datetime.now().isoformat(),
                                        "cn_name": matched_role["cn_name"],
                                        "game": matched_role["game"],
                                    },
                                    ensure_ascii=False,
                                ),
                            ),
                        )
                        total_urls += 1
                except Exception as e:
                    print(f"    ✗ 插入URL失败: {e}")

            imported_files += 1
            print(f"  ✓ {filename}: {len(urls)} 个URL")

        except Exception as e:
            print(f"  ✗ 读取文件失败 {filename}: {e}")

    print(f"\n导入文件数: {imported_files}")
    print(f"导入URL总数: {total_urls}")
    return total_urls


def import_downloaded_images(db_manager, roles):
    """导入下载的图片数据"""
    print("\n" + "=" * 60)
    print("导入下载的图片数据")
    print("=" * 60)

    downloaded_dir = os.path.join(SPIDER_DATA_DIR, "downloaded_images")
    if not os.path.exists(downloaded_dir):
        print("下载图片目录不存在")
        return 0

    total_images = 0
    imported_dirs = 0

    for dirname in os.listdir(downloaded_dir):
        dir_path = os.path.join(downloaded_dir, dirname)
        if not os.path.isdir(dir_path):
            continue

        matched_role = find_matching_role(dirname, roles)
        if not matched_role:
            print(f"  ⚠ 跳过未匹配角色: {dirname}")
            continue

        role_id = db_manager.execute_query(
            "SELECT id FROM roles WHERE name = ?", (matched_role["en_name"],)
        )

        if not role_id:
            print(f"  ⚠ 角色未在数据库中: {matched_role['en_name']}")
            continue

        role_id = role_id[0][0]

        image_count = 0
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if not os.path.isfile(file_path):
                continue

            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            try:
                file_size = os.path.getsize(file_path)
                file_hash = calculate_file_hash(file_path)

                existing = db_manager.execute_query(
                    "SELECT id FROM images WHERE image_hash = ?", (file_hash,)
                )

                if not existing:
                    from PIL import Image

                    with Image.open(file_path) as img:
                        width, height = img.size
                        format = img.format

                    db_manager.execute_update(
                        """INSERT INTO images (file_path, file_name, width, height, format, image_hash, size) 
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (file_path, filename, width, height, format, file_hash, file_size),
                    )

                    image_id = db_manager.execute_query("SELECT last_insert_rowid()")[0][0]

                    db_manager.execute_update(
                        """INSERT INTO annotations (image_id, role_id, annotation_json) 
                           VALUES (?, ?, ?)""",
                        (
                            image_id,
                            role_id,
                            json.dumps(
                                {
                                    "source": "downloaded_images",
                                    "source_dir": dirname,
                                    "imported_at": datetime.now().isoformat(),
                                    "cn_name": matched_role["cn_name"],
                                    "game": matched_role["game"],
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    )

                    total_images += 1
                    image_count += 1
            except Exception as e:
                print(f"    ✗ 处理图片失败 {filename}: {e}")

        if image_count > 0:
            imported_dirs += 1
            print(f"  ✓ {dirname}: {image_count} 张图片")

    print(f"\n导入目录数: {imported_dirs}")
    print(f"导入图片总数: {total_images}")
    return total_images


def import_download_records(db_manager):
    """导入下载记录"""
    print("\n" + "=" * 60)
    print("导入下载记录")
    print("=" * 60)

    fail_file = os.path.join(SPIDER_DATA_DIR, "download_fail_image.txt")
    if os.path.exists(fail_file):
        try:
            with open(fail_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    url = line.strip()
                    if url:
                        try:
                            db_manager.execute_update(
                                """UPDATE raw_urls SET status = 'failed', updated_at = ? 
                                   WHERE url = ?""",
                                (datetime.now().isoformat(), url),
                            )
                        except Exception as e:
                            print(f"    ✗ 更新失败记录: {e}")
            print(f"  ✓ 导入失败记录")
        except Exception as e:
            print(f"  ✗ 读取失败记录文件: {e}")

    finished_file = os.path.join(SPIDER_DATA_DIR, "download_finished_txt.txt")
    if os.path.exists(finished_file):
        try:
            with open(finished_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    url = line.strip()
                    if url:
                        try:
                            db_manager.execute_update(
                                """UPDATE raw_urls SET status = 'completed', updated_at = ? 
                                   WHERE url = ?""",
                                (datetime.now().isoformat(), url),
                            )
                        except Exception as e:
                            print(f"    ✗ 更新完成记录: {e}")
            print(f"  ✓ 导入完成记录")
        except Exception as e:
            print(f"  ✗ 读取完成记录文件: {e}")

    return 0


def generate_import_report(db_manager):
    """生成导入报告"""
    print("\n" + "=" * 60)
    print("导入统计报告")
    print("=" * 60)

    role_count = db_manager.execute_query("SELECT COUNT(*) FROM roles")[0][0]
    print(f"角色总数: {role_count}")

    url_stats = db_manager.execute_query("SELECT status, COUNT(*) FROM raw_urls GROUP BY status")
    print("\nURL统计:")
    for status, count in url_stats:
        print(f"  {status}: {count}")

    image_count = db_manager.execute_query("SELECT COUNT(*) FROM images")[0][0]
    print(f"\n图片总数: {image_count}")

    annotation_count = db_manager.execute_query("SELECT COUNT(*) FROM annotations")[0][0]
    print(f"标注总数: {annotation_count}")

    print("\n各角色图片数量:")
    role_image_stats = db_manager.execute_query(
        """SELECT r.name, r.display_name, COUNT(a.id) as image_count
           FROM roles r
           LEFT JOIN annotations a ON r.id = a.role_id
           GROUP BY r.id
           ORDER BY image_count DESC
           LIMIT 10"""
    )
    for name, display_name, count in role_image_stats:
        print(f"  {display_name}({name}): {count}")


def main():
    print("=" * 60)
    print("Spider数据导入数据库工具")
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
        print("\n创建数据库表...")
        db_manager.execute_update(
            """
            CREATE TABLE IF NOT EXISTS roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                display_name TEXT,
                origin TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        db_manager.execute_update(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                format TEXT,
                image_hash TEXT UNIQUE,
                size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        db_manager.execute_update(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                role_id INTEGER,
                features TEXT,
                annotation_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE,
                FOREIGN KEY (role_id) REFERENCES roles (id) ON DELETE CASCADE,
                UNIQUE (image_id, role_id)
            )
        """
        )

        db_manager.execute_update(
            """
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
        """
        )

        db_manager.execute_update(
            """
            CREATE TABLE IF NOT EXISTS download_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER,
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
                FOREIGN KEY (url_id) REFERENCES raw_urls (id) ON DELETE CASCADE
            )
        """
        )

        print("✓ 数据库表创建完成")

        print("\n加载角色列表...")
        roles = load_role_list()
        print(f"✓ 加载 {len(roles)} 个角色")

        import_roles(db_manager, roles)
        import_url_files(db_manager, roles)
        import_downloaded_images(db_manager, roles)
        import_download_records(db_manager)

        generate_import_report(db_manager)

        db_manager.commit()

        print("\n" + "=" * 60)
        print("✓ 数据导入完成")
        print("=" * 60)
        print(f"数据库位置: {DATABASE_FILE}")

    except Exception as e:
        print(f"\n✗ 导入失败: {e}")
        db_manager.rollback()
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
