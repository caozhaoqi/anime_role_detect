#!/usr/bin/env python3
"""
MySQL数据库表初始化脚本
- 创建MySQL数据库表结构
- 支持从.env文件读取MySQL配置
"""

import os
import sys

try:
    import mysql.connector
    from mysql.connector import Error

    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False


def load_mysql_config():
    """从.env文件加载MySQL配置"""
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    env_file = os.path.join(project_root, ".env")

    config = {
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "",
        "database": "anime_role_db",
    }

    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("mysql_host"):
                    config["host"] = line.split("=", 1)[1].strip()
                elif line.startswith("mysql_port"):
                    config["port"] = int(line.split("=", 1)[1].strip())
                elif line.startswith("mysql_user"):
                    config["user"] = line.split("=", 1)[1].strip()
                elif line.startswith("mysql_password"):
                    config["password"] = line.split("=", 1)[1].strip()
                elif line.startswith("mysql_db"):
                    config["database"] = line.split("=", 1)[1].strip()

    return config


def create_mysql_tables():
    """创建MySQL数据库表结构"""
    if not MYSQL_AVAILABLE:
        print("错误: mysql-connector-python 未安装")
        return False

    config = load_mysql_config()

    try:
        # 连接数据库
        conn = mysql.connector.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database=config["database"],
        )

        if conn.is_connected():
            cursor = conn.cursor()
            print(f"成功连接到MySQL数据库: {config['host']}:{config['port']}/{config['database']}")

            # 创建角色表
            print("\n创建角色表 (roles)...")
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS roles (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                display_name VARCHAR(255),
                origin VARCHAR(100),
                gender VARCHAR(20),
                age VARCHAR(20),
                hair_color VARCHAR(50),
                eye_color VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            )

            # 创建图片表
            print("创建图片表 (images)...")
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS images (
                id INT AUTO_INCREMENT PRIMARY KEY,
                file_path VARCHAR(512) NOT NULL,
                file_name VARCHAR(255) NOT NULL,
                width INT,
                height INT,
                format VARCHAR(20),
                image_hash VARCHAR(64),
                size BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_file_path (file_path(512)),
                UNIQUE KEY uk_image_hash (image_hash)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            )

            # 创建标注表
            print("创建标注表 (annotations)...")
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS annotations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_id INT,
                role_id INT,
                features TEXT,
                annotation_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE,
                FOREIGN KEY (role_id) REFERENCES roles (id) ON DELETE CASCADE,
                UNIQUE (image_id, role_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            )

            # 创建艺术品表
            print("创建艺术品表 (artworks)...")
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS artworks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(512),
                artist VARCHAR(255),
                source VARCHAR(100),
                source_url VARCHAR(2048),
                original_url VARCHAR(2048),
                thumbnail_url VARCHAR(2048),
                tags TEXT,
                resolution VARCHAR(50),
                file_size BIGINT,
                file_format VARCHAR(20),
                rating VARCHAR(20),
                favorites INT DEFAULT 0,
                views INT DEFAULT 0,
                published_at VARCHAR(50),
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_source_url (source(50), source_url(255))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            )

            # 创建原始URL表
            print("创建原始URL表 (raw_urls)...")
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS raw_urls (
                id INT AUTO_INCREMENT PRIMARY KEY,
                url VARCHAR(2048) NOT NULL,
                source VARCHAR(100),
                role_name VARCHAR(255),
                artwork_id INT,
                status VARCHAR(20) DEFAULT 'pending',
                priority INT DEFAULT 1,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (artwork_id) REFERENCES artworks (id) ON DELETE SET NULL,
                UNIQUE KEY uk_url (url(767))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            )

            # 创建下载记录表
            print("创建下载记录表 (download_records)...")
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS download_records (
                id INT AUTO_INCREMENT PRIMARY KEY,
                url_id INT,
                artwork_id INT,
                role_name VARCHAR(255),
                save_path VARCHAR(512),
                file_name VARCHAR(255),
                download_status VARCHAR(20) DEFAULT 'pending',
                error_message TEXT,
                http_status INT,
                file_size BIGINT,
                download_time FLOAT,
                retry_count INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (url_id) REFERENCES raw_urls (id) ON DELETE CASCADE,
                FOREIGN KEY (artwork_id) REFERENCES artworks (id) ON DELETE SET NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            )

            # 创建用户表
            print("创建用户表 (users)...")
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(512) NOT NULL,
                email VARCHAR(255) UNIQUE,
                full_name VARCHAR(255),
                role VARCHAR(20) DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            )

            # 创建分类图片记录表
            print("创建分类图片记录表 (image_records)...")
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS image_records (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                image_id INT,
                role_id INT,
                classification VARCHAR(100),
                confidence FLOAT,
                tags TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE,
                FOREIGN KEY (role_id) REFERENCES roles (id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            )

            # 创建配置表
            print("创建配置表 (configs)...")
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS configs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                `key` VARCHAR(100) UNIQUE NOT NULL,
                value TEXT NOT NULL,
                description TEXT,
                category VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            )

            # 创建模型表
            print("创建模型表 (models)...")
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS models (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                path VARCHAR(512) NOT NULL,
                type VARCHAR(50) DEFAULT 'classification',
                architecture VARCHAR(100) DEFAULT 'unknown',
                version VARCHAR(20) DEFAULT '1.0',
                accuracy FLOAT,
                `precision` FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            )

            # 创建训练记录表
            print("创建训练记录表 (training_records)...")
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS training_records (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_id INT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration FLOAT,
                epochs INT,
                batch_size INT,
                learning_rate FLOAT,
                train_loss FLOAT,
                val_loss FLOAT,
                train_accuracy FLOAT,
                val_accuracy FLOAT,
                best_epoch INT,
                best_val_accuracy FLOAT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            )

            # 创建索引（忽略已存在的索引错误）
            print("\n创建索引...")
            indexes = [
                "CREATE INDEX idx_images_file_path ON images (file_path)",
                "CREATE INDEX idx_images_image_hash ON images (image_hash)",
                "CREATE INDEX idx_annotations_image_id ON annotations (image_id)",
                "CREATE INDEX idx_annotations_role_id ON annotations (role_id)",
                "CREATE INDEX idx_raw_urls_url ON raw_urls (url(255))",
                "CREATE INDEX idx_raw_urls_source ON raw_urls (source)",
                "CREATE INDEX idx_raw_urls_role_name ON raw_urls (role_name)",
                "CREATE INDEX idx_raw_urls_status ON raw_urls (status)",
                "CREATE INDEX idx_artworks_source ON artworks (source)",
                "CREATE INDEX idx_download_records_url_id ON download_records (url_id)",
                "CREATE INDEX idx_download_records_role_name ON download_records (role_name)",
                "CREATE INDEX idx_download_records_download_status ON download_records (download_status)",
                "CREATE INDEX idx_image_records_user_id ON image_records (user_id)",
                "CREATE INDEX idx_image_records_image_id ON image_records (image_id)",
                "CREATE INDEX idx_image_records_role_id ON image_records (role_id)",
                "CREATE INDEX idx_configs_key ON configs (`key`)",
                "CREATE INDEX idx_configs_category ON configs (category)",
            ]

            for idx_sql in indexes:
                try:
                    cursor.execute(idx_sql)
                except Error as e:
                    # 忽略索引已存在的错误
                    if "Duplicate key" not in str(e) and "already exists" not in str(e).lower():
                        print(f"  创建索引失败: {idx_sql[:50]}... - {e}")

            conn.commit()
            print("\n所有表创建完成!")

            return True

    except Error as e:
        print(f"MySQL错误: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def add_initial_data():
    """添加初始数据"""
    if not MYSQL_AVAILABLE:
        print("错误: mysql-connector-python 未安装")
        return False

    config = load_mysql_config()

    try:
        conn = mysql.connector.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database=config["database"],
        )

        if conn.is_connected():
            cursor = conn.cursor()

            # 添加默认配置
            print("\n添加默认配置...")
            default_configs = [
                ("system_name", "Anime Role Detect", "系统名称", "system"),
                ("system_version", "1.0.0", "系统版本", "system"),
                ("max_upload_size", "10485760", "最大上传大小（字节）", "system"),
                ("allowed_extensions", "jpg,jpeg,png,gif", "允许的文件扩展名", "system"),
                ("model_name", "EfficientNet-B3", "使用的模型名称", "model"),
                ("confidence_threshold", "0.5", "置信度阈值", "model"),
                ("max_tags", "10", "最大标签数量", "model"),
                ("data_directory", "../../data", "数据目录", "data"),
                ("image_directory", "../../data/role_images", "图片目录", "data"),
                ("annotation_directory", "../../data/annotations", "标注目录", "data"),
                ("batch_size", "32", "批量大小", "training"),
                ("epochs", "50", "训练轮数", "training"),
                ("learning_rate", "0.001", "学习率", "training"),
            ]

            for key_val, value, description, category in default_configs:
                try:
                    cursor.execute(
                        """
                    INSERT INTO configs (`key`, value, description, category)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE value = %s, description = %s, category = %s
                    """,
                        (key_val, value, description, category, value, description, category),
                    )
                except Exception as e:
                    print(f"  添加配置失败: {key_val} - {e}")

            # 添加默认用户
            print("添加默认用户...")
            try:
                cursor.execute(
                    """
                INSERT INTO users (username, password_hash, email, full_name, role)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE password_hash = %s, email = %s, full_name = %s, role = %s
                """,
                    (
                        "admin",
                        "pbkdf2:sha256:260000$examplehash",
                        "admin@example.com",
                        "Admin User",
                        "admin",
                        "pbkdf2:sha256:260000$examplehash",
                        "admin@example.com",
                        "Admin User",
                        "admin",
                    ),
                )

                cursor.execute(
                    """
                INSERT INTO users (username, password_hash, email, full_name, role)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE password_hash = %s, email = %s, full_name = %s, role = %s
                """,
                    (
                        "user",
                        "pbkdf2:sha256:260000$examplehash",
                        "user@example.com",
                        "Test User",
                        "user",
                        "pbkdf2:sha256:260000$examplehash",
                        "user@example.com",
                        "Test User",
                        "user",
                    ),
                )
            except Exception as e:
                print(f"  添加默认用户失败: {e}")

            conn.commit()
            print("初始数据添加完成!")

            return True

    except Error as e:
        print(f"MySQL错误: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def main():
    """主函数"""
    print("=" * 60)
    print("MySQL数据库表初始化")
    print("=" * 60)

    # 创建表
    if create_mysql_tables():
        # 添加初始数据
        add_initial_data()

    print("\n" + "=" * 60)
    print("MySQL数据库表初始化完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
