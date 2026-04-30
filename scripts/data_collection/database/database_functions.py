#!/usr/bin/env python3
"""
数据库操作函数
- 支持SQLite和MySQL数据库
- 提供数据库的增删改查操作
"""

import os
import json
import sqlite3
from datetime import datetime

try:
    import mysql.connector
    from mysql.connector import Error
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

class DatabaseManager:
    def __init__(self, db_type='sqlite', database_file=None, mysql_config=None):
        """初始化数据库管理器
        
        Args:
            db_type: 'sqlite' 或 'mysql'
            database_file: SQLite数据库文件路径（仅当db_type为'sqlite'时使用）
            mysql_config: MySQL配置字典（仅当db_type为'mysql'时使用）
        """
        self.db_type = db_type
        self.conn = None
        self.cursor = None
        
        if db_type == 'mysql':
            if mysql_config is None:
                # 从.env文件读取配置
                self.mysql_config = self._load_mysql_config()
            else:
                self.mysql_config = mysql_config
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if database_file is None:
                # 从database目录向上三级到项目根目录，然后进入data目录
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
                self.database_file = os.path.join(project_root, 'data', 'role_images.db')
            else:
                self.database_file = os.path.join(script_dir, database_file)
    
    def _load_mysql_config(self):
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
    
    def connect(self):
        """连接数据库"""
        try:
            if self.db_type == 'mysql':
                if not MYSQL_AVAILABLE:
                    raise ImportError("mysql-connector-python not installed")
                
                self.conn = mysql.connector.connect(
                    host=self.mysql_config['host'],
                    port=self.mysql_config['port'],
                    user=self.mysql_config['user'],
                    password=self.mysql_config['password'],
                    database=self.mysql_config['database']
                )
                self.cursor = self.conn.cursor()
            else:
                # SQLite
                # 确保目录存在
                os.makedirs(os.path.dirname(self.database_file), exist_ok=True)
                self.conn = sqlite3.connect(self.database_file)
                self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return False
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
    
    def _execute(self, sql, params=None):
        """执行SQL语句"""
        try:
            if params is None:
                self.cursor.execute(sql)
            else:
                self.cursor.execute(sql, params)
            return True
        except Exception as e:
            print(f"SQL执行失败: {e}")
            print(f"SQL: {sql}")
            print(f"Params: {params}")
            return False
    
    def _fetchone(self):
        """获取单行结果"""
        try:
            return self.cursor.fetchone()
        except Exception as e:
            print(f"获取结果失败: {e}")
            return None
    
    def _fetchall(self):
        """获取所有结果"""
        try:
            return self.cursor.fetchall()
        except Exception as e:
            print(f"获取结果失败: {e}")
            return []
    
    def get_all_roles(self):
        """获取所有角色"""
        self.connect()
        self._execute('SELECT * FROM roles')
        roles = self._fetchall()
        self.close()
        return roles
    
    def get_role_by_name(self, name):
        """根据名称获取角色"""
        self.connect()
        self._execute('SELECT * FROM roles WHERE name = %s', (name,))
        role = self._fetchone()
        self.close()
        return role
    
    def get_all_images(self):
        """获取所有图片"""
        self.connect()
        self._execute('SELECT * FROM images')
        images = self._fetchall()
        self.close()
        return images
    
    def get_images_by_role(self, role_name):
        """获取指定角色的所有图片"""
        self.connect()
        sql = "SELECT i.* FROM images i JOIN annotations a ON i.id = a.image_id JOIN roles r ON a.role_id = r.id WHERE r.name = %s"
        self._execute(sql, (role_name,))
        images = self._fetchall()
        self.close()
        return images
    
    def get_annotations_by_image(self, image_path):
        """获取指定图片的标注"""
        self.connect()
        sql = "SELECT a.*, r.name as role_name FROM annotations a JOIN images i ON a.image_id = i.id JOIN roles r ON a.role_id = r.id WHERE i.file_path = %s"
        self._execute(sql, (image_path,))
        annotations = self._fetchall()
        self.close()
        return annotations
    
    def add_role(self, name, display_name=None, origin=None, gender=None, age=None, hair_color=None, eye_color=None):
        """添加角色"""
        self.connect()
        try:
            sql = "INSERT INTO roles (name, display_name, origin, gender, age, hair_color, eye_color) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            self.cursor.execute(sql, (name, display_name or name, origin, gender, age, hair_color, eye_color))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"添加角色失败: {e}")
            return False
        finally:
            self.close()
    
    def add_image(self, file_path, file_name, width, height, format, size):
        """添加图片"""
        self.connect()
        try:
            sql = "INSERT INTO images (file_path, file_name, width, height, format, size) VALUES (%s, %s, %s, %s, %s, %s)"
            self.cursor.execute(sql, (file_path, file_name, width, height, format, size))
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            print(f"添加图片失败: {e}")
            return None
        finally:
            self.close()
    
    def add_annotation(self, image_id, role_id, features, annotation_json):
        """添加标注"""
        self.connect()
        try:
            sql = "INSERT IGNORE INTO annotations (image_id, role_id, features, annotation_json) VALUES (%s, %s, %s, %s)"
            self.cursor.execute(sql, (image_id, role_id, json.dumps(features), json.dumps(annotation_json)))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"添加标注失败: {e}")
            return False
        finally:
            self.close()
    
    def update_role(self, role_id, **kwargs):
        """更新角色信息"""
        self.connect()
        try:
            set_clause = ', '.join([f"{key} = %s" for key in kwargs])
            values = list(kwargs.values()) + [role_id]
            sql = f"UPDATE roles SET {set_clause} WHERE id = %s"
            self.cursor.execute(sql, values)
            self.conn.commit()
            return True
        except Exception as e:
            print(f"更新角色失败: {e}")
            return False
        finally:
            self.close()
    
    def delete_role(self, role_id):
        """删除角色"""
        self.connect()
        try:
            # 先删除相关的标注
            self.cursor.execute('DELETE FROM annotations WHERE role_id = %s', (role_id,))
            # 再删除角色
            self.cursor.execute('DELETE FROM roles WHERE id = %s', (role_id,))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"删除角色失败: {e}")
            return False
        finally:
            self.close()
    
    def get_statistics(self):
        """获取数据库统计信息"""
        self.connect()
        
        try:
            # 角色数量
            self.cursor.execute('SELECT COUNT(*) FROM roles')
            role_count = self.cursor.fetchone()[0]
            
            # 图片数量
            self.cursor.execute('SELECT COUNT(*) FROM images')
            image_count = self.cursor.fetchone()[0]
            
            # 标注数量
            self.cursor.execute('SELECT COUNT(*) FROM annotations')
            annotation_count = self.cursor.fetchone()[0]
            
            # 每个角色的图片数量
            sql = "SELECT r.name, COUNT(i.id) as image_count FROM roles r LEFT JOIN annotations a ON r.id = a.role_id LEFT JOIN images i ON a.image_id = i.id GROUP BY r.name ORDER BY image_count DESC"
            self.cursor.execute(sql)
            role_image_counts = self.cursor.fetchall()
            
            self.close()
            
            return {
                'role_count': role_count,
                'image_count': image_count,
                'annotation_count': annotation_count,
                'role_image_counts': role_image_counts
            }
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            self.close()
            return {}
    
    def add_user(self, username, password_hash, email=None, full_name=None, role='user'):
        """添加用户"""
        self.connect()
        try:
            self.cursor.execute('INSERT INTO users (username, password_hash, email, full_name, role) VALUES (%s, %s, %s, %s, %s)', (username, password_hash, email, full_name, role))
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
        self.cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = self.cursor.fetchone()
        self.close()
        return user
    
    def add_image_record(self, user_id, image_id, role_id, classification, confidence, tags=None, notes=None):
        """添加分类图片记录"""
        self.connect()
        try:
            self.cursor.execute('INSERT INTO image_records (user_id, image_id, role_id, classification, confidence, tags, notes) VALUES (%s, %s, %s, %s, %s, %s, %s)', (user_id, image_id, role_id, classification, confidence, tags, notes))
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
        self.cursor.execute('SELECT ir.*, i.file_path, r.name as role_name FROM image_records ir JOIN images i ON ir.image_id = i.id JOIN roles r ON ir.role_id = r.id WHERE ir.user_id = %s ORDER BY ir.created_at DESC', (user_id,))
        records = self.cursor.fetchall()
        self.close()
        return records
    
    def get_config(self, key):
        """获取配置"""
        self.connect()
        self.cursor.execute('SELECT value FROM configs WHERE key = %s', (key,))
        result = self.cursor.fetchone()
        self.close()
        return result[0] if result else None
    
    def set_config(self, key, value, description=None, category=None):
        """设置配置"""
        self.connect()
        try:
            self.cursor.execute('INSERT INTO configs (key, value, description, category) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE value = %s, description = %s, category = %s', (key, value, description, category, value, description, category))
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
            self.cursor.execute('SELECT * FROM configs WHERE category = %s ORDER BY key', (category,))
        else:
            self.cursor.execute('SELECT * FROM configs ORDER BY category, key')
        configs = self.cursor.fetchall()
        self.close()
        return configs
    
    def add_model(self, name, path, type='classification', architecture='unknown', version='1.0', accuracy=None, precision=None, recall=None, f1_score=None):
        """添加模型"""
        self.connect()
        try:
            self.cursor.execute('INSERT INTO models (name, path, type, architecture, version, accuracy, precision, recall, f1_score) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE path = %s, type = %s, architecture = %s, version = %s, accuracy = %s, precision = %s, recall = %s, f1_score = %s', (name, path, type, architecture, version, accuracy, precision, recall, f1_score, path, type, architecture, version, accuracy, precision, recall, f1_score))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"添加模型失败: {e}")
            return False
        finally:
            self.close()
    
    def get_model_by_name(self, name):
        """根据名称获取模型"""
        self.connect()
        self.cursor.execute('SELECT * FROM models WHERE name = %s', (name,))
        model = self.cursor.fetchone()
        self.close()
        return model
    
    def get_all_models(self):
        """获取所有模型"""
        self.connect()
        self.cursor.execute('SELECT * FROM models ORDER BY created_at DESC')
        models = self.cursor.fetchall()
        self.close()
        return models
    
    def add_training_record(self, model_id, start_time, end_time, duration, epochs, batch_size, learning_rate, train_loss, val_loss, train_accuracy, val_accuracy, best_epoch, best_val_accuracy, notes=None):
        """添加训练记录"""
        self.connect()
        try:
            self.cursor.execute('INSERT INTO training_records (model_id, start_time, end_time, duration, epochs, batch_size, learning_rate, train_loss, val_loss, train_accuracy, val_accuracy, best_epoch, best_val_accuracy, notes) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', (model_id, start_time, end_time, duration, epochs, batch_size, learning_rate, train_loss, val_loss, train_accuracy, val_accuracy, best_epoch, best_val_accuracy, notes))
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            print(f"添加训练记录失败: {e}")
            return None
        finally:
            self.close()
    
    def get_training_records_by_model(self, model_id):
        """获取模型的训练记录"""
        self.connect()
        self.cursor.execute('SELECT * FROM training_records WHERE model_id = %s ORDER BY start_time DESC', (model_id,))
        records = self.cursor.fetchall()
        self.close()
        return records
    
    def get_all_training_records(self):
        """获取所有训练记录"""
        self.connect()
        self.cursor.execute('SELECT tr.*, m.name as model_name FROM training_records tr JOIN models m ON tr.model_id = m.id ORDER BY tr.start_time DESC')
        records = self.cursor.fetchall()
        self.close()
        return records
    
    # ==================== 数据采集相关方法 ====================
    
    def add_raw_url(self, url, source=None, role_name=None, artwork_id=None, priority=1, metadata=None):
        """添加原始URL地址"""
        self.connect()
        try:
            self.cursor.execute('''
            INSERT INTO raw_urls (url, source, role_name, artwork_id, priority, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE source = %s, role_name = %s, artwork_id = %s, priority = %s, metadata = %s
            ''', (url, source, role_name, artwork_id, priority, json.dumps(metadata) if metadata else None,
                  source, role_name, artwork_id, priority, json.dumps(metadata) if metadata else None))
            self.conn.commit()
            
            # 获取插入的ID或已存在的ID
            self.cursor.execute('SELECT id FROM raw_urls WHERE url = %s', (url,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"添加原始URL失败: {e}")
            return None
        finally:
            self.close()
    
    def get_url_by_url(self, url):
        """根据URL获取记录"""
        self.connect()
        try:
            self.cursor.execute('SELECT * FROM raw_urls WHERE url = %s', (url,))
            return self.cursor.fetchone()
        except Exception as e:
            print(f"获取URL记录失败: {e}")
            return None
        finally:
            self.close()
    
    def update_url_status(self, url_id, status):
        """更新URL状态"""
        self.connect()
        try:
            self.cursor.execute('UPDATE raw_urls SET status = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s', (status, url_id))
            self.conn.commit()
            return self.cursor.rowcount > 0
        except Exception as e:
            print(f"更新URL状态失败: {e}")
            return False
        finally:
            self.close()
    
    def get_pending_urls(self, limit=100):
        """获取待处理的URL列表"""
        self.connect()
        try:
            self.cursor.execute('''
            SELECT * FROM raw_urls 
            WHERE status = 'pending' 
            ORDER BY priority DESC, created_at ASC 
            LIMIT %s
            ''', (limit,))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"获取待处理URL失败: {e}")
            return []
        finally:
            self.close()
    
    def add_artwork(self, title=None, artist=None, source=None, source_url=None, original_url=None, 
                    thumbnail_url=None, tags=None, resolution=None, file_size=None, 
                    file_format=None, rating=None, favorites=0, views=0, published_at=None, metadata=None):
        """添加艺术品信息"""
        self.connect()
        try:
            self.cursor.execute('''
            INSERT INTO artworks (title, artist, source, source_url, original_url, 
                                   thumbnail_url, tags, resolution, file_size, 
                                   file_format, rating, favorites, views, published_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE title = %s, artist = %s, original_url = %s, 
                                   thumbnail_url = %s, tags = %s, resolution = %s, 
                                   file_size = %s, file_format = %s, rating = %s, 
                                   favorites = %s, views = %s, published_at = %s, metadata = %s
            ''', (title, artist, source, source_url, original_url, thumbnail_url, 
                  json.dumps(tags) if tags else None, resolution, file_size, 
                  file_format, rating, favorites, views, published_at, 
                  json.dumps(metadata) if metadata else None,
                  title, artist, original_url, thumbnail_url, 
                  json.dumps(tags) if tags else None, resolution, 
                  file_size, file_format, rating, favorites, views, 
                  published_at, json.dumps(metadata) if metadata else None))
            self.conn.commit()
            
            # 获取插入的ID或已存在的ID
            self.cursor.execute('SELECT id FROM artworks WHERE source = %s AND source_url = %s', (source, source_url))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"添加艺术品失败: {e}")
            return None
        finally:
            self.close()
    
    def add_download_record(self, url_id, artwork_id=None, role_name=None, save_path=None, 
                            file_name=None, download_status='pending', error_message=None, 
                            http_status=None, file_size=None, download_time=None, retry_count=0):
        """添加下载记录"""
        self.connect()
        try:
            self.cursor.execute('''
            INSERT INTO download_records (url_id, artwork_id, role_name, save_path, file_name, 
                                          download_status, error_message, http_status, 
                                          file_size, download_time, retry_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (url_id, artwork_id, role_name, save_path, file_name, 
                  download_status, error_message, http_status, file_size, download_time, retry_count))
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            print(f"添加下载记录失败: {e}")
            return None
        finally:
            self.close()
    
    def update_download_record(self, record_id, **kwargs):
        """更新下载记录"""
        self.connect()
        try:
            set_clause = ', '.join([f"{key} = %s" for key in kwargs])
            values = list(kwargs.values()) + [record_id]
            sql = f"UPDATE download_records SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = %s"
            self.cursor.execute(sql, values)
            self.conn.commit()
            return self.cursor.rowcount > 0
        except Exception as e:
            print(f"更新下载记录失败: {e}")
            return False
        finally:
            self.close()
    
    def get_download_records_by_role(self, role_name):
        """获取角色的下载记录"""
        self.connect()
        try:
            self.cursor.execute('''
            SELECT dr.*, ru.url, a.title 
            FROM download_records dr 
            LEFT JOIN raw_urls ru ON dr.url_id = ru.id 
            LEFT JOIN artworks a ON dr.artwork_id = a.id 
            WHERE dr.role_name = %s 
            ORDER BY dr.created_at DESC
            ''', (role_name,))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"获取角色下载记录失败: {e}")
            return []
        finally:
            self.close()
    
    def get_collection_statistics(self):
        """获取采集统计信息"""
        self.connect()
        try:
            # URL统计
            self.cursor.execute('SELECT COUNT(*) FROM raw_urls')
            total_urls = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT COUNT(*) FROM raw_urls WHERE status = "pending"')
            pending_urls = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT COUNT(*) FROM raw_urls WHERE status = "downloaded"')
            downloaded_urls = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT COUNT(*) FROM raw_urls WHERE status = "failed"')
            failed_urls = self.cursor.fetchone()[0]
            
            # Artwork统计
            self.cursor.execute('SELECT COUNT(*) FROM artworks')
            total_artworks = self.cursor.fetchone()[0]
            
            # 下载记录统计
            self.cursor.execute('SELECT COUNT(*) FROM download_records')
            total_downloads = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT COUNT(*) FROM download_records WHERE download_status = "success"')
            success_downloads = self.cursor.fetchone()[0]
            
            # 按角色统计
            self.cursor.execute('''
            SELECT role_name, COUNT(*) as count 
            FROM raw_urls 
            WHERE role_name IS NOT NULL 
            GROUP BY role_name 
            ORDER BY count DESC
            ''')
            role_stats = self.cursor.fetchall()
            
            self.close()
            
            return {
                'total_urls': total_urls,
                'pending_urls': pending_urls,
                'downloaded_urls': downloaded_urls,
                'failed_urls': failed_urls,
                'total_artworks': total_artworks,
                'total_downloads': total_downloads,
                'success_downloads': success_downloads,
                'role_stats': role_stats
            }
        except Exception as e:
            print(f"获取采集统计失败: {e}")
            self.close()
            return {}
    
    def add_batch_urls(self, urls, source=None, role_name=None):
        """批量添加URL"""
        self.connect()
        success_count = 0
        fail_count = 0
        
        try:
            for url in urls:
                try:
                    self.cursor.execute('''
                    INSERT INTO raw_urls (url, source, role_name)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE source = %s, role_name = %s
                    ''', (url, source, role_name, source, role_name))
                    if self.cursor.rowcount > 0:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    print(f"添加URL失败: {url} - {e}")
            
            self.conn.commit()
            print(f"批量添加URL完成: 成功 {success_count} 条，失败 {fail_count} 条")
            return success_count, fail_count
        except Exception as e:
            print(f"批量添加URL失败: {e}")
            return 0, len(urls)
        finally:
            self.close()

if __name__ == "__main__":
    # 测试数据库操作
    print("测试SQLite数据库...")
    sqlite_db = DatabaseManager(db_type='sqlite')
    
    try:
        stats = sqlite_db.get_statistics()
        print("SQLite数据库统计信息:")
        print(f"  角色数量: {stats.get('role_count', 0)}")
        print(f"  图片数量: {stats.get('image_count', 0)}")
        print(f"  标注数量: {stats.get('annotation_count', 0)}")
    except Exception as e:
        print(f"SQLite测试失败: {e}")
    
    print("\n测试MySQL数据库...")
    mysql_db = DatabaseManager(db_type='mysql')
    
    if mysql_db.connect():
        print("MySQL连接成功!")
        
        try:
            stats = mysql_db.get_statistics()
            print("MySQL数据库统计信息:")
            print(f"  角色数量: {stats.get('role_count', 0)}")
            print(f"  图片数量: {stats.get('image_count', 0)}")
            print(f"  标注数量: {stats.get('annotation_count', 0)}")
        except Exception as e:
            print(f"MySQL查询失败: {e}")
        
        mysql_db.close()
    else:
        print("MySQL连接失败，可能是数据库未配置或mysql-connector-python未安装")