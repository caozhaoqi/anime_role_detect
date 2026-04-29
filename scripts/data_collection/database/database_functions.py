#!/usr/bin/env python3
"""
数据库操作函数
- 提供数据库的增删改查操作
"""

import os
import json
import sqlite3
from datetime import datetime

class DatabaseManager:
    def __init__(self, database_file='../../data/role_images.db'):
        """初始化数据库管理器"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.database_file = os.path.join(script_dir, database_file)
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """连接数据库"""
        self.conn = sqlite3.connect(self.database_file)
        self.cursor = self.conn.cursor()
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
    
    def get_all_roles(self):
        """获取所有角色"""
        self.connect()
        self.cursor.execute('SELECT * FROM roles')
        roles = self.cursor.fetchall()
        self.close()
        return roles
    
    def get_role_by_name(self, name):
        """根据名称获取角色"""
        self.connect()
        self.cursor.execute('SELECT * FROM roles WHERE name = ?', (name,))
        role = self.cursor.fetchone()
        self.close()
        return role
    
    def get_all_images(self):
        """获取所有图片"""
        self.connect()
        self.cursor.execute('SELECT * FROM images')
        images = self.cursor.fetchall()
        self.close()
        return images
    
    def get_images_by_role(self, role_name):
        """获取指定角色的所有图片"""
        self.connect()
        sql = "SELECT i.* FROM images i JOIN annotations a ON i.id = a.image_id JOIN roles r ON a.role_id = r.id WHERE r.name = ?"
        self.cursor.execute(sql, (role_name,))
        images = self.cursor.fetchall()
        self.close()
        return images
    
    def get_annotations_by_image(self, image_path):
        """获取指定图片的标注"""
        self.connect()
        sql = "SELECT a.*, r.name as role_name FROM annotations a JOIN images i ON a.image_id = i.id JOIN roles r ON a.role_id = r.id WHERE i.file_path = ?"
        self.cursor.execute(sql, (image_path,))
        annotations = self.cursor.fetchall()
        self.close()
        return annotations
    
    def add_role(self, name, display_name=None, origin=None, gender=None, age=None, hair_color=None, eye_color=None):
        """添加角色"""
        self.connect()
        try:
            sql = "INSERT INTO roles (name, display_name, origin, gender, age, hair_color, eye_color) VALUES (?, ?, ?, ?, ?, ?, ?)"
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
            sql = "INSERT INTO images (file_path, file_name, width, height, format, size) VALUES (?, ?, ?, ?, ?, ?)"
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
            sql = "INSERT OR IGNORE INTO annotations (image_id, role_id, features, annotation_json) VALUES (?, ?, ?, ?)"
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
            set_clause = ', '.join([f"{key} = ?" for key in kwargs])
            values = list(kwargs.values()) + [role_id]
            sql = f"UPDATE roles SET {set_clause} WHERE id = ?"
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
            self.cursor.execute('DELETE FROM annotations WHERE role_id = ?', (role_id,))
            # 再删除角色
            self.cursor.execute('DELETE FROM roles WHERE id = ?', (role_id,))
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

    def add_model(self, name, path, type='classification', architecture='unknown', version='1.0', accuracy=None, precision=None, recall=None, f1_score=None):
        """添加模型"""
        self.connect()
        try:
            self.cursor.execute('INSERT OR REPLACE INTO models (name, path, type, architecture, version, accuracy, precision, recall, f1_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (name, path, type, architecture, version, accuracy, precision, recall, f1_score))
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
        self.cursor.execute('SELECT * FROM models WHERE name = ?', (name,))
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
            self.cursor.execute('INSERT INTO training_records (model_id, start_time, end_time, duration, epochs, batch_size, learning_rate, train_loss, val_loss, train_accuracy, val_accuracy, best_epoch, best_val_accuracy, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (model_id, start_time, end_time, duration, epochs, batch_size, learning_rate, train_loss, val_loss, train_accuracy, val_accuracy, best_epoch, best_val_accuracy, notes))
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
        self.cursor.execute('SELECT * FROM training_records WHERE model_id = ? ORDER BY start_time DESC', (model_id,))
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

if __name__ == "__main__":
    # 测试数据库操作
    db_manager = DatabaseManager()
    
    # 获取统计信息
    stats = db_manager.get_statistics()
    print("数据库统计信息:")
    print(f"角色数量: {stats['role_count']}")
    print(f"图片数量: {stats['image_count']}")
    print(f"标注数量: {stats['annotation_count']}")
    print("\n每个角色的图片数量:")
    for role_name, image_count in stats['role_image_counts']:
        print(f"{role_name}: {image_count} 张")
