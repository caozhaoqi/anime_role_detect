#!/usr/bin/env python3
"""
更新数据库标注信息脚本
- 将扩展后的标注信息更新到数据库
"""

import os
import json
import sqlite3
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='update_database_annotations.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# 全局配置
GLOBAL_CONFIG = {
    'database_file': '../../data/role_images.db',
    'annotation_dir': '../../data/annotations'
}

def update_annotations():
    """更新数据库中的标注信息"""
    logger.info("开始更新数据库标注信息")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_file = os.path.join(script_dir, GLOBAL_CONFIG['database_file'])
    annotation_dir = os.path.join(script_dir, GLOBAL_CONFIG['annotation_dir'])
    
    # 连接数据库
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    
    # 遍历角色目录
    total_roles = 0
    total_annotations = 0
    
    for role_name in os.listdir(annotation_dir):
        role_dir = os.path.join(annotation_dir, role_name)
        if not os.path.isdir(role_dir):
            continue
        
        total_roles += 1
        logger.info(f"处理角色: {role_name}")
        
        # 获取角色ID
        cursor.execute('SELECT id FROM roles WHERE name = ?', (role_name,))
        role_result = cursor.fetchone()
        if not role_result:
            logger.warning(f"角色 {role_name} 在数据库中不存在")
            continue
        role_id = role_result[0]
        
        # 遍历角色目录下的标注文件
        for file_name in os.listdir(role_dir):
            if not file_name.endswith('.json'):
                continue
            
            annotation_file = os.path.join(role_dir, file_name)
            image_name = os.path.splitext(file_name)[0] + os.path.splitext(file_name)[1].replace('.json', '.jpg')
            
            # 读取标注文件
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
            except Exception as e:
                logger.warning(f"读取标注文件失败: {annotation_file} - {str(e)}")
                continue
            
            # 获取图片ID
            image_path = annotation_data.get('image_path')
            if not image_path:
                logger.warning(f"标注文件 {annotation_file} 缺少图片路径")
                continue
            
            cursor.execute('SELECT id FROM images WHERE file_path = ?', (image_path,))
            image_result = cursor.fetchone()
            if not image_result:
                logger.warning(f"图片 {image_path} 在数据库中不存在")
                continue
            image_id = image_result[0]
            
            # 更新标注信息
            try:
                cursor.execute('''
                UPDATE annotations SET annotation_json = ? WHERE image_id = ? AND role_id = ?
                ''', (json.dumps(annotation_data), image_id, role_id))
                
                if cursor.rowcount > 0:
                    total_annotations += 1
                    logger.info(f"更新标注信息: {file_name}")
            except Exception as e:
                logger.warning(f"更新标注信息失败: {annotation_file} - {str(e)}")
        
        # 每处理完一个角色，提交一次更改
        conn.commit()
    
    # 关闭数据库连接
    conn.close()
    
    logger.info("\n============================================================")
    logger.info("数据库标注信息更新完成")
    logger.info(f"总处理角色数: {total_roles}")
    logger.info(f"总更新标注数: {total_annotations}")
    logger.info("============================================================")

def main():
    """主函数"""
    logger.info("============================================================")
    logger.info("开始更新数据库标注信息")
    logger.info("============================================================")
    
    update_annotations()

if __name__ == "__main__":
    main()
