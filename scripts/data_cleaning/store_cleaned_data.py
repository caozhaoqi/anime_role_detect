#!/usr/bin/env python3
"""
数据入库脚本
处理重复数据并将清洗后的数据存储到数据库
"""
import os
import sys
import json
import sqlite3
import hashlib
from PIL import Image
from datetime import datetime


class DataStore:
    """数据存储管理器"""

    def __init__(self, db_path="./cleaned_data.db"):
        """初始化数据库连接"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """创建数据库表"""
        cursor = self.conn.cursor()

        # 角色表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                source TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """
        )

        # 图片表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role_id INTEGER,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                hash_value TEXT,
                width INTEGER,
                height INTEGER,
                file_size INTEGER,
                quality_score REAL,
                is_valid INTEGER DEFAULT 1,
                created_at TEXT,
                FOREIGN KEY (role_id) REFERENCES roles(id)
            )
        """
        )

        # 清洗记录表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cleaning_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT,
                total_images INTEGER,
                valid_images INTEGER,
                invalid_images INTEGER,
                duplicate_count INTEGER,
                low_resolution_count INTEGER,
                small_file_count INTEGER,
                start_time TEXT,
                end_time TEXT,
                duration REAL
            )
        """
        )

        self.conn.commit()

    def get_or_create_role(self, role_name):
        """获取或创建角色记录"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM roles WHERE name = ?", (role_name,))
        result = cursor.fetchone()

        if result:
            return result[0]

        cursor.execute(
            """
            INSERT INTO roles (name, source, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """,
            (role_name, "collection", datetime.now().isoformat(), datetime.now().isoformat()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def calculate_image_hash(self, image_path):
        """计算图片哈希值"""
        try:
            with open(image_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"计算哈希失败 {image_path}: {e}")
            return None

    def get_image_info(self, image_path):
        """获取图片信息"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                file_size = os.path.getsize(image_path)

                # 计算质量分数（基于分辨率和文件大小）
                quality_score = min(width * height / (1024 * 1024), 10.0)

                return {
                    "width": width,
                    "height": height,
                    "file_size": file_size,
                    "quality_score": round(quality_score, 2),
                }
        except Exception as e:
            print(f"获取图片信息失败 {image_path}: {e}")
            return None

    def check_duplicate(self, role_id, hash_value):
        """检查同一角色内是否重复"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM images WHERE role_id = ? AND hash_value = ?", (role_id, hash_value)
        )
        return cursor.fetchone() is not None

    def insert_image(self, role_id, file_path, role_hashes=None):
        """插入图片记录"""
        filename = os.path.basename(file_path)
        img_hash = self.calculate_image_hash(file_path)

        if not img_hash:
            return False, "计算哈希失败"

        # 检查同一角色内重复
        if role_hashes is not None and img_hash in role_hashes:
            return False, "重复图片"

        img_info = self.get_image_info(file_path)
        if not img_info:
            return False, "获取图片信息失败"

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO images (role_id, filename, file_path, hash_value, 
                                  width, height, file_size, quality_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    role_id,
                    filename,
                    file_path,
                    img_hash,
                    img_info["width"],
                    img_info["height"],
                    img_info["file_size"],
                    img_info["quality_score"],
                    datetime.now().isoformat(),
                ),
            )
            self.conn.commit()

            if role_hashes is not None:
                role_hashes.add(img_hash)

            return True, "成功"
        except sqlite3.IntegrityError:
            return False, "路径重复"
        except Exception as e:
            return False, str(e)

    def insert_cleaning_record(self, batch_id, stats):
        """插入清洗记录"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO cleaning_records (batch_id, total_images, valid_images, 
                                         invalid_images, duplicate_count, 
                                         low_resolution_count, small_file_count,
                                         start_time, end_time, duration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                batch_id,
                stats["total"],
                stats["valid"],
                stats["invalid"],
                stats.get("duplicate", 0),
                stats.get("low_resolution", 0),
                stats.get("small_file", 0),
                stats["start_time"],
                stats["end_time"],
                stats["duration"],
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_role_stats(self):
        """获取角色统计"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT r.name, COUNT(i.id) as image_count
            FROM roles r
            LEFT JOIN images i ON r.id = i.role_id
            GROUP BY r.id
            ORDER BY image_count DESC
        """
        )
        return cursor.fetchall()

    def get_total_stats(self):
        """获取总体统计"""
        cursor = self.conn.cursor()

        # 总角色数
        cursor.execute("SELECT COUNT(*) FROM roles")
        total_roles = cursor.fetchone()[0]

        # 总图片数
        cursor.execute("SELECT COUNT(*) FROM images")
        total_images = cursor.fetchone()[0]

        # 平均质量分数
        cursor.execute("SELECT AVG(quality_score) FROM images")
        avg_quality = cursor.fetchone()[0]

        return {
            "total_roles": total_roles,
            "total_images": total_images,
            "avg_quality": round(avg_quality, 2) if avg_quality else 0,
        }

    def close(self):
        """关闭数据库连接"""
        self.conn.close()


def process_cleaned_data(cleaned_dir, db_path="./cleaned_data.db"):
    """处理清洗后的数据并入库"""
    print(f"开始处理清洗后的数据: {cleaned_dir}")
    print(f"数据库路径: {db_path}")

    start_time = datetime.now()
    store = DataStore(db_path)

    stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "duplicate": 0,
        "low_resolution": 0,
        "small_file": 0,
        "start_time": start_time.isoformat(),
    }

    # 遍历所有角色目录
    role_dirs = []
    for item in os.listdir(cleaned_dir):
        item_path = os.path.join(cleaned_dir, item)
        if os.path.isdir(item_path):
            role_dirs.append((item, item_path))

    print(f"发现 {len(role_dirs)} 个角色")

    for role_name, role_dir in role_dirs:
        print(f"\n处理角色: {role_name}")

        # 获取或创建角色
        role_id = store.get_or_create_role(role_name)

        # 统计角色图片
        role_success = 0
        role_failed = 0

        # 角色内部去重集合
        role_hashes = set()

        # 遍历角色目录中的图片
        for filename in os.listdir(role_dir):
            if filename.endswith(".jpg"):
                file_path = os.path.join(role_dir, filename)
                stats["total"] += 1

                success, msg = store.insert_image(role_id, file_path, role_hashes)
                if success:
                    stats["valid"] += 1
                    role_success += 1
                else:
                    stats["invalid"] += 1
                    role_failed += 1
                    if msg == "重复图片":
                        stats["duplicate"] += 1

        print(f"  成功: {role_success}, 失败: {role_failed}")

    end_time = datetime.now()
    stats["end_time"] = end_time.isoformat()
    stats["duration"] = round((end_time - start_time).total_seconds(), 2)

    # 插入清洗记录
    batch_id = f"clean_{start_time.strftime('%Y%m%d_%H%M%S')}"
    store.insert_cleaning_record(batch_id, stats)

    # 获取统计信息
    total_stats = store.get_total_stats()
    role_stats = store.get_role_stats()

    store.close()

    # 输出结果
    print("\n" + "=" * 50)
    print("数据入库完成")
    print("=" * 50)
    print(f"总处理图片: {stats['total']}")
    print(f"成功入库: {stats['valid']}")
    print(f"入库失败: {stats['invalid']}")
    print(f"重复图片: {stats['duplicate']}")
    print(f"处理时长: {stats['duration']}秒")
    print("\n总体统计:")
    print(f"  总角色数: {total_stats['total_roles']}")
    print(f"  总图片数: {total_stats['total_images']}")
    print(f"  平均质量分数: {total_stats['avg_quality']}")

    # 保存统计结果到JSON文件
    result = {
        "batch_id": batch_id,
        "stats": stats,
        "total_stats": total_stats,
        "role_stats": [{"role": r[0], "count": r[1]} for r in role_stats],
    }

    with open("cleaning_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: cleaning_result.json")
    print(f"数据库已保存到: {db_path}")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="数据入库脚本")
    parser.add_argument("--cleaned_dir", type=str, required=True, help="清洗后数据目录")
    parser.add_argument("--db_path", type=str, default="./cleaned_data.db", help="数据库路径")

    args = parser.parse_args()

    process_cleaned_data(args.cleaned_dir, args.db_path)


if __name__ == "__main__":
    main()
