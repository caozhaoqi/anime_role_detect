#!/usr/bin/env python3
"""
数据库存储方案脚本
- 设计并创建数据库表结构
- 将现有数据导入到数据库
- 提供数据库操作接口
"""

import os
import json
import sqlite3
from datetime import datetime

# 导入统一日志配置
from common.logging_config import get_logger

# 配置日志
logger = get_logger("data_collection.setup_database", "setup_database.log")

# 全局配置
GLOBAL_CONFIG = {
    "database_file": "../../data/role_images.db",
    "image_dir": "../../data/role_images",
    "annotation_dir": "../../data/annotations",
}


def ensure_directory(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def create_database():
    """创建数据库表结构"""
    logger.info("开始创建数据库")

    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_file = os.path.join(script_dir, GLOBAL_CONFIG["database_file"])

    # 确保数据库目录存在
    ensure_directory(os.path.dirname(database_file))

    # 连接数据库
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # 创建角色表
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS roles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        display_name TEXT,
        origin TEXT,
        gender TEXT,
        age TEXT,
        hair_color TEXT,
        eye_color TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    )

    # 创建图片表
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE NOT NULL,
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

    # 创建标注表
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER,
        role_id INTEGER,
        features TEXT,
        annotation_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (image_id) REFERENCES images (id),
        FOREIGN KEY (role_id) REFERENCES roles (id),
        UNIQUE (image_id, role_id)
    )
    """
    )

    # 创建索引
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_file_path ON images (file_path)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_image_hash ON images (image_hash)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotations_image_id ON annotations (image_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotations_role_id ON annotations (role_id)")

    # 提交更改
    conn.commit()
    conn.close()

    logger.info(f"数据库创建完成: {database_file}")
    return database_file


def get_role_id(cursor, role_name):
    """获取角色ID，如果不存在则创建"""
    cursor.execute("SELECT id FROM roles WHERE name = ?", (role_name,))
    result = cursor.fetchone()

    if result:
        return result[0]
    else:
        # 创建新角色
        cursor.execute(
            """
        INSERT INTO roles (name, display_name) VALUES (?, ?)
        """,
            (role_name, role_name),
        )
        return cursor.lastrowid


def get_image_id(cursor, image_path, image_info):
    """获取图片ID，如果不存在则创建"""
    cursor.execute("SELECT id FROM images WHERE file_path = ?", (image_path,))
    result = cursor.fetchone()

    if result:
        return result[0]
    else:
        # 获取文件大小
        try:
            size = os.path.getsize(image_path)
        except:
            size = 0

        # 创建新图片记录
        cursor.execute(
            """
        INSERT INTO images (file_path, file_name, width, height, format, size)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                image_path,
                os.path.basename(image_path),
                image_info.get("width", 0),
                image_info.get("height", 0),
                image_info.get("format", "unknown"),
                size,
            ),
        )
        return cursor.lastrowid


def import_data():
    """导入现有数据到数据库"""
    logger.info("开始导入数据到数据库")

    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_file = os.path.join(script_dir, GLOBAL_CONFIG["database_file"])
    image_dir = os.path.join(script_dir, GLOBAL_CONFIG["image_dir"])
    annotation_dir = os.path.join(script_dir, GLOBAL_CONFIG["annotation_dir"])

    # 连接数据库
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # 导入角色和图片数据
    total_roles = 0
    total_images = 0
    total_annotations = 0

    # 遍历角色目录
    for role_name in os.listdir(image_dir):
        role_dir = os.path.join(image_dir, role_name)
        if not os.path.isdir(role_dir):
            continue

        total_roles += 1
        logger.info(f"处理角色: {role_name}")

        # 获取角色ID
        role_id = get_role_id(cursor, role_name)

        # 遍历角色目录下的图片
        for file_name in os.listdir(role_dir):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            image_path = os.path.join(role_dir, file_name)
            total_images += 1

            # 尝试读取标注文件
            annotation_file = os.path.join(
                annotation_dir, role_name, f"{os.path.splitext(file_name)[0]}.json"
            )
            image_info = {}
            features = []
            annotation_json = {}

            if os.path.exists(annotation_file):
                try:
                    with open(annotation_file, "r", encoding="utf-8") as f:
                        annotation_data = json.load(f)
                        image_info = annotation_data.get("image_info", {})
                        features = annotation_data.get("features", [])
                        annotation_json = annotation_data
                except Exception as e:
                    logger.warning(f"读取标注文件失败: {annotation_file} - {str(e)}")

            # 获取图片ID
            image_id = get_image_id(cursor, image_path, image_info)

            # 创建标注记录
            try:
                cursor.execute(
                    """
                INSERT OR IGNORE INTO annotations (image_id, role_id, features, annotation_json)
                VALUES (?, ?, ?, ?)
                """,
                    (image_id, role_id, json.dumps(features), json.dumps(annotation_json)),
                )
                if cursor.rowcount > 0:
                    total_annotations += 1
            except Exception as e:
                logger.warning(f"创建标注记录失败: {image_path} - {str(e)}")

        # 每处理完一个角色，提交一次更改
        conn.commit()

    # 关闭数据库连接
    conn.close()

    logger.info("\n============================================================")
    logger.info("数据导入完成")
    logger.info(f"总角色数: {total_roles}")
    logger.info(f"总图片数: {total_images}")
    logger.info(f"总标注数: {total_annotations}")
    logger.info("============================================================")


def create_database_functions():
    """创建数据库操作函数"""
    logger.info("数据库操作函数已单独创建")
    return "database_functions.py"


def main():
    """主函数"""
    logger.info("============================================================")
    logger.info("开始优化数据存储")
    logger.info("============================================================")

    # 创建数据库
    database_file = create_database()

    # 导入数据
    import_data()

    # 创建数据库操作函数
    create_database_functions()

    logger.info("\n============================================================")
    logger.info("数据存储优化完成")
    logger.info(f"数据库文件: {database_file}")
    logger.info("============================================================")


if __name__ == "__main__":
    main()
