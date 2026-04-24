#!/usr/bin/env python3
"""
数据版本控制脚本
- 管理数据集的版本
- 创建版本快照
- 记录版本之间的差异
- 支持版本回滚
- 提供版本历史查询
"""

import os
import json
import shutil
import hashlib
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='version_control.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# 全局配置
GLOBAL_CONFIG = {
    'data_dir': '../../data',
    'version_dir': '../../data/versions',
    'database_file': '../../data/role_images.db',
    'metadata_file': 'version_metadata.json'
}

def ensure_directory(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def calculate_directory_hash(directory):
    """计算目录的哈希值"""
    hasher = hashlib.md5()
    
    for root, dirs, files in os.walk(directory):
        # 排序确保顺序一致
        dirs.sort()
        files.sort()
        
        for file in files:
            if file == GLOBAL_CONFIG['metadata_file']:
                continue
            
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'rb') as f:
                    while True:
                        data = f.read(65536)  # 64KB chunks
                        if not data:
                            break
                        hasher.update(data)
            except Exception as e:
                logger.warning(f"计算文件哈希值失败: {file_path} - {str(e)}")
    
    return hasher.hexdigest()

def create_version(version_name, description):
    """创建新版本"""
    logger.info(f"开始创建版本: {version_name}")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, GLOBAL_CONFIG['data_dir'])
    version_dir = os.path.join(script_dir, GLOBAL_CONFIG['version_dir'])
    
    # 确保版本目录存在
    ensure_directory(version_dir)
    
    # 计算当前数据的哈希值
    data_hash = calculate_directory_hash(data_dir)
    logger.info(f"当前数据哈希值: {data_hash}")
    
    # 检查是否已经存在相同哈希值的版本
    existing_versions = []
    for version_folder in os.listdir(version_dir):
        version_path = os.path.join(version_dir, version_folder)
        if os.path.isdir(version_path):
            metadata_file = os.path.join(version_path, GLOBAL_CONFIG['metadata_file'])
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        if metadata.get('data_hash') == data_hash:
                            existing_versions.append(version_folder)
                except Exception as e:
                    logger.warning(f"读取版本元数据失败: {metadata_file} - {str(e)}")
    
    if existing_versions:
        logger.info(f"已经存在相同数据的版本: {existing_versions}")
        return False, f"已经存在相同数据的版本: {existing_versions}"
    
    # 创建版本目录
    version_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_folder = f"{version_timestamp}_{version_name}"
    version_path = os.path.join(version_dir, version_folder)
    ensure_directory(version_path)
    
    # 复制数据到版本目录
    logger.info("开始复制数据到版本目录")
    
    # 复制role_images目录
    src_role_images = os.path.join(data_dir, 'role_images')
    dst_role_images = os.path.join(version_path, 'role_images')
    if os.path.exists(src_role_images):
        shutil.copytree(src_role_images, dst_role_images)
    
    # 复制annotations目录
    src_annotations = os.path.join(data_dir, 'annotations')
    dst_annotations = os.path.join(version_path, 'annotations')
    if os.path.exists(src_annotations):
        shutil.copytree(src_annotations, dst_annotations)
    
    # 复制数据库文件
    src_db = os.path.join(script_dir, GLOBAL_CONFIG['database_file'])
    dst_db = os.path.join(version_path, 'role_images.db')
    if os.path.exists(src_db):
        shutil.copy2(src_db, dst_db)
    
    # 生成版本元数据
    metadata = {
        'version_name': version_name,
        'version_timestamp': version_timestamp,
        'description': description,
        'data_hash': data_hash,
        'created_at': datetime.now().isoformat(),
        'file_counts': {
            'role_images': count_files(src_role_images) if os.path.exists(src_role_images) else 0,
            'annotations': count_files(src_annotations) if os.path.exists(src_annotations) else 0
        }
    }
    
    # 保存元数据
    metadata_file = os.path.join(version_path, GLOBAL_CONFIG['metadata_file'])
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"版本创建完成: {version_folder}")
    return True, version_folder

def count_files(directory):
    """统计目录中的文件数量"""
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

def list_versions():
    """列出所有版本"""
    logger.info("开始列出所有版本")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    version_dir = os.path.join(script_dir, GLOBAL_CONFIG['version_dir'])
    
    if not os.path.exists(version_dir):
        logger.info("版本目录不存在")
        return []
    
    versions = []
    for version_folder in os.listdir(version_dir):
        version_path = os.path.join(version_dir, version_folder)
        if os.path.isdir(version_path):
            metadata_file = os.path.join(version_path, GLOBAL_CONFIG['metadata_file'])
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        metadata['version_folder'] = version_folder
                        versions.append(metadata)
                except Exception as e:
                    logger.warning(f"读取版本元数据失败: {metadata_file} - {str(e)}")
    
    # 按时间戳排序
    versions.sort(key=lambda x: x.get('version_timestamp', ''), reverse=True)
    
    logger.info(f"找到 {len(versions)} 个版本")
    return versions

def rollback_to_version(version_folder):
    """回滚到指定版本"""
    logger.info(f"开始回滚到版本: {version_folder}")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, GLOBAL_CONFIG['data_dir'])
    version_dir = os.path.join(script_dir, GLOBAL_CONFIG['version_dir'])
    version_path = os.path.join(version_dir, version_folder)
    
    # 检查版本是否存在
    if not os.path.exists(version_path):
        logger.error(f"版本不存在: {version_folder}")
        return False, f"版本不存在: {version_folder}"
    
    # 备份当前数据
    backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(version_dir, f"backup_{backup_timestamp}")
    ensure_directory(backup_path)
    
    # 备份role_images目录
    src_role_images = os.path.join(data_dir, 'role_images')
    dst_role_images = os.path.join(backup_path, 'role_images')
    if os.path.exists(src_role_images):
        shutil.copytree(src_role_images, dst_role_images)
    
    # 备份annotations目录
    src_annotations = os.path.join(data_dir, 'annotations')
    dst_annotations = os.path.join(backup_path, 'annotations')
    if os.path.exists(src_annotations):
        shutil.copytree(src_annotations, dst_annotations)
    
    # 备份数据库文件
    src_db = os.path.join(script_dir, GLOBAL_CONFIG['database_file'])
    dst_db = os.path.join(backup_path, 'role_images.db')
    if os.path.exists(src_db):
        shutil.copy2(src_db, dst_db)
    
    logger.info(f"已备份当前数据到: {backup_path}")
    
    # 恢复版本数据
    # 恢复role_images目录
    src_role_images = os.path.join(version_path, 'role_images')
    dst_role_images = os.path.join(data_dir, 'role_images')
    if os.path.exists(src_role_images):
        if os.path.exists(dst_role_images):
            shutil.rmtree(dst_role_images)
        shutil.copytree(src_role_images, dst_role_images)
    
    # 恢复annotations目录
    src_annotations = os.path.join(version_path, 'annotations')
    dst_annotations = os.path.join(data_dir, 'annotations')
    if os.path.exists(src_annotations):
        if os.path.exists(dst_annotations):
            shutil.rmtree(dst_annotations)
        shutil.copytree(src_annotations, dst_annotations)
    
    # 恢复数据库文件
    src_db = os.path.join(version_path, 'role_images.db')
    dst_db = os.path.join(script_dir, GLOBAL_CONFIG['database_file'])
    if os.path.exists(src_db):
        if os.path.exists(dst_db):
            os.remove(dst_db)
        shutil.copy2(src_db, dst_db)
    
    logger.info(f"回滚到版本完成: {version_folder}")
    return True, f"回滚到版本 {version_folder} 成功，当前数据已备份到 {backup_path}"

def delete_version(version_folder):
    """删除指定版本"""
    logger.info(f"开始删除版本: {version_folder}")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    version_dir = os.path.join(script_dir, GLOBAL_CONFIG['version_dir'])
    version_path = os.path.join(version_dir, version_folder)
    
    # 检查版本是否存在
    if not os.path.exists(version_path):
        logger.error(f"版本不存在: {version_folder}")
        return False, f"版本不存在: {version_folder}"
    
    # 删除版本目录
    try:
        shutil.rmtree(version_path)
        logger.info(f"版本删除完成: {version_folder}")
        return True, f"版本 {version_folder} 删除成功"
    except Exception as e:
        logger.error(f"删除版本失败: {e}")
        return False, f"删除版本失败: {str(e)}"

def compare_versions(version1, version2):
    """比较两个版本的差异"""
    logger.info(f"开始比较版本: {version1} 和 {version2}")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    version_dir = os.path.join(script_dir, GLOBAL_CONFIG['version_dir'])
    version1_path = os.path.join(version_dir, version1)
    version2_path = os.path.join(version_dir, version2)
    
    # 检查版本是否存在
    if not os.path.exists(version1_path):
        return False, f"版本不存在: {version1}"
    if not os.path.exists(version2_path):
        return False, f"版本不存在: {version2}"
    
    # 读取版本元数据
    def read_metadata(version_path):
        metadata_file = os.path.join(version_path, GLOBAL_CONFIG['metadata_file'])
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"读取版本元数据失败: {metadata_file} - {str(e)}")
        return {}
    
    metadata1 = read_metadata(version1_path)
    metadata2 = read_metadata(version2_path)
    
    # 比较差异
    differences = {
        'version1': metadata1,
        'version2': metadata2,
        'changes': {
            'data_hash_changed': metadata1.get('data_hash') != metadata2.get('data_hash'),
            'file_counts': {
                'role_images': metadata2.get('file_counts', {}).get('role_images', 0) - metadata1.get('file_counts', {}).get('role_images', 0),
                'annotations': metadata2.get('file_counts', {}).get('annotations', 0) - metadata1.get('file_counts', {}).get('annotations', 0)
            }
        }
    }
    
    logger.info("版本比较完成")
    return True, differences

def main():
    """主函数"""
    logger.info("============================================================")
    logger.info("开始数据版本控制")
    logger.info("============================================================")
    
    # 示例用法
    # 1. 创建新版本
    # create_version("initial_version", "初始版本")
    
    # 2. 列出所有版本
    versions = list_versions()
    logger.info(f"当前版本数量: {len(versions)}")
    for version in versions:
        logger.info(f"版本: {version.get('version_folder')}, 名称: {version.get('version_name')}, 创建时间: {version.get('created_at')}")
    
    # 3. 回滚到指定版本
    # if versions:
    #     rollback_to_version(versions[-1].get('version_folder'))
    
    # 4. 比较版本
    # if len(versions) >= 2:
    #     compare_versions(versions[0].get('version_folder'), versions[1].get('version_folder'))
    
    logger.info("\n============================================================")
    logger.info("数据版本控制完成")
    logger.info("============================================================")

if __name__ == "__main__":
    main()
