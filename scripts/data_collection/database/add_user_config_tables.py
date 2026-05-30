#!/usr/bin/env python3
"""
添加用户信息、分类图片记录信息和配置信息表
- 在数据库中创建users、image_records和configs表
- 实现相关的CRUD操作
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
    filename="add_user_config_tables.log",
    filemode="a",
)
logger = logging.getLogger(__name__)

# 全局配置
GLOBAL_CONFIG = {"database_file": "../../data/role_images.db"}


def create_tables():
    """创建用户信息、分类图片记录信息和配置信息表"""
    logger.info("开始创建用户信息、分类图片记录信息和配置信息表")

    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_file = os.path.join(script_dir, GLOBAL_CONFIG["database_file"])

    # 连接数据库
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # 创建用户表
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT UNIQUE,
        full_name TEXT,
        role TEXT DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    """
    )
    logger.info("创建用户表完成")

    # 创建分类图片记录表
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS image_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        image_id INTEGER,
        role_id INTEGER,
        classification TEXT,
        confidence REAL,
        tags TEXT,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (image_id) REFERENCES images (id),
        FOREIGN KEY (role_id) REFERENCES roles (id)
    )
    """
    )
    logger.info("创建分类图片记录表完成")

    # 创建配置表
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE NOT NULL,
        value TEXT NOT NULL,
        description TEXT,
        category TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    )
    logger.info("创建配置表完成")

    # 创建索引
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_image_records_user_id ON image_records (user_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_image_records_image_id ON image_records (image_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_image_records_role_id ON image_records (role_id)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_configs_key ON configs (key)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_configs_category ON configs (category)")
    logger.info("创建索引完成")

    # 提交更改
    conn.commit()
    conn.close()

    logger.info("表创建完成")


def add_initial_data():
    """添加初始数据"""
    logger.info("开始添加初始数据")

    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_file = os.path.join(script_dir, GLOBAL_CONFIG["database_file"])

    # 连接数据库
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # 添加默认配置
    default_configs = [
        # 系统配置
        ("system_name", "Anime Role Detect", "系统名称", "system"),
        ("system_version", "1.0.0", "系统版本", "system"),
        ("max_upload_size", "10485760", "最大上传大小（字节）", "system"),
        ("allowed_extensions", "jpg,jpeg,png,gif", "允许的文件扩展名", "system"),
        # 模型配置
        ("model_name", "EfficientNet-B3", "使用的模型名称", "model"),
        ("confidence_threshold", "0.5", "置信度阈值", "model"),
        ("max_tags", "10", "最大标签数量", "model"),
        # 数据配置
        ("data_directory", "../../data", "数据目录", "data"),
        ("image_directory", "../../data/role_images", "图片目录", "data"),
        ("annotation_directory", "../../data/annotations", "标注目录", "data"),
        # 训练配置
        ("batch_size", "32", "批量大小", "training"),
        ("epochs", "50", "训练轮数", "training"),
        ("learning_rate", "0.001", "学习率", "training"),
    ]

    for key, value, description, category in default_configs:
        try:
            cursor.execute(
                """
            INSERT OR IGNORE INTO configs (key, value, description, category)
            VALUES (?, ?, ?, ?)
            """,
                (key, value, description, category),
            )
        except Exception as e:
            logger.warning(f"添加配置失败: {key} - {str(e)}")

    # 添加默认用户
    try:
        # 密码哈希：123456（示例）
        cursor.execute(
            """
        INSERT OR IGNORE INTO users (username, password_hash, email, full_name, role)
        VALUES (?, ?, ?, ?, ?)
        """,
            (
                "admin",
                "pbkdf2:sha256:260000$examplehash",
                "admin@example.com",
                "Admin User",
                "admin",
            ),
        )

        cursor.execute(
            """
        INSERT OR IGNORE INTO users (username, password_hash, email, full_name, role)
        VALUES (?, ?, ?, ?, ?)
        """,
            ("user", "pbkdf2:sha256:260000$examplehash", "user@example.com", "Test User", "user"),
        )
    except Exception as e:
        logger.warning(f"添加默认用户失败: {str(e)}")

    # 提交更改
    conn.commit()
    conn.close()

    logger.info("初始数据添加完成")


def update_database_functions():
    """更新数据库操作函数，添加新表的操作方法"""
    logger.info("开始更新数据库操作函数")

    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    functions_file = os.path.join(script_dir, "database_functions.py")

    # 读取现有文件内容
    with open(functions_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查是否已经添加了新的方法
    if "def add_user" in content:
        logger.info("数据库操作函数已经包含新表的操作方法")
        return

    # 在DatabaseManager类中添加新的方法
    # 找到类的末尾
    class_end = content.rfind("    def get_statistics(self):")
    if class_end == -1:
        logger.error("找不到DatabaseManager类的get_statistics方法")
        return

    # 找到get_statistics方法的末尾
    method_end = content.find("        return {", class_end)
    if method_end == -1:
        logger.error("找不到get_statistics方法的开始")
        return

    # 找到方法的结束
    method_end = content.find("    }", method_end)
    if method_end == -1:
        logger.error("找不到get_statistics方法的结束")
        return
    method_end += 5  # 包含方法结束的大括号

    # 添加新的方法
    new_methods = '''
    def add_user(self, username, password_hash, email=None, full_name=None, role='user'):
        """添加用户"""
        self.connect()
        try:
            self.cursor.execute('INSERT INTO users (username, password_hash, email, full_name, role) VALUES (?, ?, ?, ?, ?)', (username, password_hash, email, full_name, role))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"添加用户失败: {e}")
            return False
        finally:
            self.close()
    
    def get_user_by_username(self, username):
        """根据用户名获取用户"""
        self.connect()
        self.cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = self.cursor.fetchone()
        self.close()
        return user
    
    def add_image_record(self, user_id, image_id, role_id, classification, confidence, tags=None, notes=None):
        """添加分类图片记录"""
        self.connect()
        try:
            self.cursor.execute('INSERT INTO image_records (user_id, image_id, role_id, classification, confidence, tags, notes) VALUES (?, ?, ?, ?, ?, ?, ?)', (user_id, image_id, role_id, classification, confidence, tags, notes))
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            print(f"添加分类图片记录失败: {e}")
            return None
        finally:
            self.close()
    
    def get_image_records_by_user(self, user_id):
        """获取用户的分类图片记录"""
        self.connect()
        self.cursor.execute('SELECT ir.*, i.file_path, r.name as role_name FROM image_records ir JOIN images i ON ir.image_id = i.id JOIN roles r ON ir.role_id = r.id WHERE ir.user_id = ? ORDER BY ir.created_at DESC', (user_id,))
        records = self.cursor.fetchall()
        self.close()
        return records
    
    def get_config(self, key):
        """获取配置"""
        self.connect()
        self.cursor.execute('SELECT value FROM configs WHERE key = ?', (key,))
        result = self.cursor.fetchone()
        self.close()
        return result[0] if result else None
    
    def set_config(self, key, value, description=None, category=None):
        """设置配置"""
        self.connect()
        try:
            self.cursor.execute('INSERT OR REPLACE INTO configs (key, value, description, category) VALUES (?, ?, ?, ?)', (key, value, description, category))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"设置配置失败: {e}")
            return False
        finally:
            self.close()
    
    def get_all_configs(self, category=None):
        """获取所有配置"""
        self.connect()
        if category:
            self.cursor.execute('SELECT * FROM configs WHERE category = ? ORDER BY key', (category,))
        else:
            self.cursor.execute('SELECT * FROM configs ORDER BY category, key')
        configs = self.cursor.fetchall()
        self.close()
        return configs
'''

    # 插入新方法
    new_content = content[:method_end] + new_methods + content[method_end:]

    # 保存更新后的文件
    with open(functions_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    logger.info("数据库操作函数更新完成")


def main():
    """主函数"""
    logger.info("============================================================")
    logger.info("开始添加用户信息、分类图片记录信息和配置信息表")
    logger.info("============================================================")

    # 创建表
    create_tables()

    # 添加初始数据
    add_initial_data()

    # 更新数据库操作函数
    update_database_functions()

    logger.info("\n============================================================")
    logger.info("用户信息、分类图片记录信息和配置信息表添加完成")
    logger.info("============================================================")


if __name__ == "__main__":
    main()
