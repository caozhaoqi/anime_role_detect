#!/usr/bin/env python3
"""
同步模型和模型训练记录到数据库脚本
- 创建models和training_records表
- 扫描模型目录，将模型信息添加到数据库
- 扫描训练日志，将训练记录添加到数据库
- 更新数据库操作函数
"""

import os
import json
import sqlite3
import logging
import glob
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sync_models_to_db.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# 全局配置
GLOBAL_CONFIG = {
    'database_file': '../../data/role_images.db',
    'model_dir': '../../models',
    'training_log_dir': '../../logs/training'
}

def create_model_tables():
    """创建模型和训练记录表"""
    logger.info("开始创建模型和训练记录表")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_file = os.path.join(script_dir, GLOBAL_CONFIG['database_file'])
    
    # 连接数据库
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    
    # 创建模型表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        path TEXT UNIQUE NOT NULL,
        type TEXT,
        architecture TEXT,
        version TEXT,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    logger.info("创建模型表完成")
    
    # 创建训练记录表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS training_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER,
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        duration INTEGER,
        epochs INTEGER,
        batch_size INTEGER,
        learning_rate REAL,
        train_loss REAL,
        val_loss REAL,
        train_accuracy REAL,
        val_accuracy REAL,
        best_epoch INTEGER,
        best_val_accuracy REAL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (model_id) REFERENCES models (id)
    )
    ''')
    logger.info("创建训练记录表完成")
    
    # 创建索引
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_name ON models (name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_records_model_id ON training_records (model_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_records_start_time ON training_records (start_time)')
    logger.info("创建索引完成")
    
    # 提交更改
    conn.commit()
    conn.close()
    
    logger.info("表创建完成")

def scan_models():
    """扫描模型目录，将模型信息添加到数据库"""
    logger.info("开始扫描模型目录")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_file = os.path.join(script_dir, GLOBAL_CONFIG['database_file'])
    model_dir = os.path.join(script_dir, GLOBAL_CONFIG['model_dir'])
    
    # 确保模型目录存在
    if not os.path.exists(model_dir):
        logger.warning(f"模型目录不存在: {model_dir}")
        return []
    
    # 连接数据库
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    
    # 扫描模型文件
    model_files = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.h5') or file.endswith('.keras') or file.endswith('.pth'):
                model_path = os.path.join(root, file)
                model_name = os.path.splitext(file)[0]
                model_files.append((model_name, model_path))
    
    logger.info(f"找到 {len(model_files)} 个模型文件")
    
    # 将模型信息添加到数据库
    added_models = 0
    for model_name, model_path in model_files:
        # 相对路径
        relative_path = os.path.relpath(model_path, script_dir)
        
        # 推断模型类型和架构
        model_type = 'classification'
        architecture = 'unknown'
        
        if 'efficientnet' in model_name.lower():
            architecture = 'EfficientNet'
        elif 'resnet' in model_name.lower():
            architecture = 'ResNet'
        elif 'vgg' in model_name.lower():
            architecture = 'VGG'
        elif 'mobilenet' in model_name.lower():
            architecture = 'MobileNet'
        
        # 提取版本号
        version = '1.0'
        
        try:
            cursor.execute('''
            INSERT OR REPLACE INTO models (name, path, type, architecture, version)
            VALUES (?, ?, ?, ?, ?)
            ''', (model_name, relative_path, model_type, architecture, version))
            added_models += 1
            logger.info(f"添加模型: {model_name}")
        except Exception as e:
            logger.warning(f"添加模型失败: {model_name} - {str(e)}")
    
    # 提交更改
    conn.commit()
    conn.close()
    
    logger.info(f"添加了 {added_models} 个模型")
    return model_files

def scan_training_logs():
    """扫描训练日志，将训练记录添加到数据库"""
    logger.info("开始扫描训练日志")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_file = os.path.join(script_dir, GLOBAL_CONFIG['database_file'])
    training_log_dir = os.path.join(script_dir, GLOBAL_CONFIG['training_log_dir'])
    
    # 确保训练日志目录存在
    if not os.path.exists(training_log_dir):
        logger.warning(f"训练日志目录不存在: {training_log_dir}")
        return []
    
    # 连接数据库
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    
    # 扫描训练日志文件
    log_files = glob.glob(os.path.join(training_log_dir, '*.json'))
    logger.info(f"找到 {len(log_files)} 个训练日志文件")
    
    # 将训练记录添加到数据库
    added_records = 0
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # 获取模型名称
            model_name = log_data.get('model_name', os.path.splitext(os.path.basename(log_file))[0])
            
            # 获取模型ID
            cursor.execute('SELECT id FROM models WHERE name = ?', (model_name,))
            model_result = cursor.fetchone()
            if not model_result:
                # 如果模型不存在，先添加模型
                model_path = log_data.get('model_path', '')
                cursor.execute('''
                INSERT OR REPLACE INTO models (name, path, type, architecture, version)
                VALUES (?, ?, ?, ?, ?)
                ''', (model_name, model_path, 'classification', log_data.get('architecture', 'unknown'), '1.0'))
                model_id = cursor.lastrowid
            else:
                model_id = model_result[0]
            
            # 计算训练时长
            start_time = datetime.fromisoformat(log_data.get('start_time', datetime.now().isoformat()))
            end_time = datetime.fromisoformat(log_data.get('end_time', datetime.now().isoformat()))
            duration = int((end_time - start_time).total_seconds())
            
            # 提取训练信息
            epochs = log_data.get('epochs', 0)
            batch_size = log_data.get('batch_size', 0)
            learning_rate = log_data.get('learning_rate', 0.0)
            train_loss = log_data.get('train_loss', 0.0)
            val_loss = log_data.get('val_loss', 0.0)
            train_accuracy = log_data.get('train_accuracy', 0.0)
            val_accuracy = log_data.get('val_accuracy', 0.0)
            best_epoch = log_data.get('best_epoch', 0)
            best_val_accuracy = log_data.get('best_val_accuracy', 0.0)
            notes = log_data.get('notes', '')
            
            # 添加训练记录
            cursor.execute('''
            INSERT INTO training_records (
                model_id, start_time, end_time, duration, epochs, batch_size, 
                learning_rate, train_loss, val_loss, train_accuracy, val_accuracy, 
                best_epoch, best_val_accuracy, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id, start_time.isoformat(), end_time.isoformat(), duration, epochs, batch_size, 
                learning_rate, train_loss, val_loss, train_accuracy, val_accuracy, 
                best_epoch, best_val_accuracy, notes
            ))
            
            added_records += 1
            logger.info(f"添加训练记录: {model_name}")
        except Exception as e:
            logger.warning(f"添加训练记录失败: {log_file} - {str(e)}")
    
    # 提交更改
    conn.commit()
    conn.close()
    
    logger.info(f"添加了 {added_records} 个训练记录")
    return log_files

def update_database_functions():
    """更新数据库操作函数，添加模型和训练记录的操作方法"""
    logger.info("开始更新数据库操作函数")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    functions_file = os.path.join(script_dir, 'database_functions.py')
    
    # 读取现有文件内容
    with open(functions_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经添加了新的方法
    if 'def add_model' in content:
        logger.info("数据库操作函数已经包含模型和训练记录的操作方法")
        return
    
    # 在DatabaseManager类中添加新的方法
    # 找到类的末尾
    class_end = content.rfind('    def get_all_configs(self, category=None):')
    if class_end == -1:
        logger.error("找不到DatabaseManager类的get_all_configs方法")
        return
    
    # 找到get_all_configs方法的末尾
    method_end = content.find('    }', class_end)
    if method_end == -1:
        logger.error("找不到get_all_configs方法的结束")
        return
    method_end += 5  # 包含方法结束的大括号
    
    # 添加新的方法
    new_methods = '''
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
'''
    
    # 插入新方法
    new_content = content[:method_end] + new_methods + content[method_end:]
    
    # 保存更新后的文件
    with open(functions_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    logger.info("数据库操作函数更新完成")

def main():
    """主函数"""
    logger.info("============================================================")
    logger.info("开始同步模型和模型训练记录到数据库")
    logger.info("============================================================")
    
    # 创建模型和训练记录表
    create_model_tables()
    
    # 扫描模型目录
    scan_models()
    
    # 扫描训练日志
    scan_training_logs()
    
    # 更新数据库操作函数
    update_database_functions()
    
    logger.info("\n============================================================")
    logger.info("模型和模型训练记录同步完成")
    logger.info("============================================================")

if __name__ == "__main__":
    main()
