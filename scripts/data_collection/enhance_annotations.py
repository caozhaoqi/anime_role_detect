#!/usr/bin/env python3
"""
扩展标注信息脚本
- 增加更多标注维度，如角色服装、场景等
"""

import os
import json
import shutil
from datetime import datetime

# 导入统一日志配置
from scripts.common.logging_config import get_logger

# 配置日志
logger = get_logger('data_collection.enhance_annotations', 'enhance_annotations.log')

# 全局配置
GLOBAL_CONFIG = {
    'annotation_dir': '../../data/annotations',
    'backup_dir': '../../data/annotations_backup'
}

# 扩展标注维度的可能值
ENHANCED_ANNOTATIONS = {
    'clothing': [
        'school uniform', 'uniform', 'casual', 'formal', 'sportswear',
        'swimsuit', 'costume', 'dress', 'pants', 'skirt',
        'jacket', 'sweater', 'shirt', 't-shirt', 'hoodie'
    ],
    'scene': [
        'indoor', 'outdoor', 'school', 'classroom', 'cafeteria',
        'library', 'park', 'street', 'beach', 'forest',
        'city', 'home', 'office', 'battlefield', 'studio'
    ],
    'pose': [
        'standing', 'sitting', 'kneeling', 'lying', 'jumping',
        'running', 'walking', 'dancing', 'fighting', 'waving',
        'pointing', 'thinking', 'smiling', 'laughing', 'crying'
    ],
    'expression': [
        'smile', 'serious', 'happy', 'sad', 'angry',
        'surprised', 'confused', 'worried', 'excited', 'calm',
        'embarrassed', 'determined', 'tired', 'sleepy', 'neutral'
    ],
    'accessories': [
        'glasses', 'hat', 'headphones', 'scarf', 'necklace',
        'bracelet', 'ring', 'backpack', 'bag', 'weapon',
        'book', 'phone', 'camera', 'umbrella', 'watch'
    ],
    'lighting': [
        'natural', 'artificial', 'bright', 'dim', 'sunny',
        'cloudy', 'night', 'day', 'indirect', 'direct'
    ],
    'angle': [
        'front', 'side', 'back', '3/4', 'top',
        'bottom', 'close-up', 'medium', 'full-body', 'extreme close-up'
    ],
    'style': [
        'anime', 'manga', 'realistic', 'chibi', 'cartoony',
        'stylized', 'semi-realistic', 'pixel', 'vector', '3D'
    ]
}

# 角色特定的标注映射
ROLE_SPECIFIC_ANNOTATIONS = {
    'a1luo2na4': {
        'clothing': ['school uniform', 'dress'],
        'scene': ['school', 'classroom', 'library'],
        'expression': ['smile', 'happy', 'calm'],
        'accessories': []
    },
    'hua1yin1': {
        'clothing': ['school uniform', 'formal'],
        'scene': ['school', 'library', 'indoor'],
        'expression': ['serious', 'calm', 'determined'],
        'accessories': ['glasses']
    },
    'shen1yue4': {
        'clothing': ['school uniform', 'casual'],
        'scene': ['school', 'park', 'outdoor'],
        'expression': ['smile', 'happy', 'excited'],
        'accessories': []
    },
    'li3shi4': {
        'clothing': ['school uniform', 'casual'],
        'scene': ['school', 'library', 'classroom'],
        'expression': ['serious', 'confused', 'determined'],
        'accessories': ['glasses', 'book']
    },
    'lei3bei4': {
        'clothing': ['school uniform', 'sportswear'],
        'scene': ['school', 'outdoor', 'gym'],
        'expression': ['smile', 'excited', 'happy'],
        'accessories': []
    }
}

def ensure_directory(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def backup_annotations():
    """备份现有标注文件"""
    logger.info("开始备份现有标注文件")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotation_dir = os.path.join(script_dir, GLOBAL_CONFIG['annotation_dir'])
    backup_dir = os.path.join(script_dir, GLOBAL_CONFIG['backup_dir'])
    
    # 确保备份目录存在
    ensure_directory(backup_dir)
    
    # 复制标注文件到备份目录
    import shutil
    for role_name in os.listdir(annotation_dir):
        role_dir = os.path.join(annotation_dir, role_name)
        if os.path.isdir(role_dir):
            backup_role_dir = os.path.join(backup_dir, role_name)
            if os.path.exists(backup_role_dir):
                shutil.rmtree(backup_role_dir)
            shutil.copytree(role_dir, backup_role_dir)
            logger.info(f"备份角色 {role_name} 的标注文件")
    
    logger.info("标注文件备份完成")

def enhance_annotation(annotation, role_name):
    """扩展标注信息"""
    # 获取角色特定的标注
    role_annotations = ROLE_SPECIFIC_ANNOTATIONS.get(role_name, {})
    
    # 扩展标注维度
    enhanced_annotation = annotation.copy()
    
    # 添加服装类型
    if 'clothing' not in enhanced_annotation:
        if 'clothing' in role_annotations:
            enhanced_annotation['clothing'] = random.choice(role_annotations['clothing'])
        else:
            enhanced_annotation['clothing'] = random.choice(ENHANCED_ANNOTATIONS['clothing'])
    
    # 添加场景类型
    if 'scene' not in enhanced_annotation:
        if 'scene' in role_annotations:
            enhanced_annotation['scene'] = random.choice(role_annotations['scene'])
        else:
            enhanced_annotation['scene'] = random.choice(ENHANCED_ANNOTATIONS['scene'])
    
    # 添加姿态
    if 'pose' not in enhanced_annotation:
        enhanced_annotation['pose'] = random.choice(ENHANCED_ANNOTATIONS['pose'])
    
    # 添加表情
    if 'expression' not in enhanced_annotation:
        if 'expression' in role_annotations:
            enhanced_annotation['expression'] = random.choice(role_annotations['expression'])
        else:
            enhanced_annotation['expression'] = random.choice(ENHANCED_ANNOTATIONS['expression'])
    
    # 添加配饰
    if 'accessories' not in enhanced_annotation:
        if 'accessories' in role_annotations:
            enhanced_annotation['accessories'] = role_annotations['accessories']
        else:
            # 随机选择0-2个配饰
            num_accessories = random.randint(0, 2)
            if num_accessories > 0:
                enhanced_annotation['accessories'] = random.sample(ENHANCED_ANNOTATIONS['accessories'], num_accessories)
            else:
                enhanced_annotation['accessories'] = []
    
    # 添加光照
    if 'lighting' not in enhanced_annotation:
        enhanced_annotation['lighting'] = random.choice(ENHANCED_ANNOTATIONS['lighting'])
    
    # 添加角度
    if 'angle' not in enhanced_annotation:
        enhanced_annotation['angle'] = random.choice(ENHANCED_ANNOTATIONS['angle'])
    
    # 添加风格
    if 'style' not in enhanced_annotation:
        enhanced_annotation['style'] = random.choice(ENHANCED_ANNOTATIONS['style'])
    
    return enhanced_annotation

def process_role_annotations(role_name, role_dir):
    """处理单个角色的标注文件"""
    logger.info(f"开始处理角色 {role_name} 的标注文件")
    
    # 获取角色目录下的所有标注文件
    annotation_files = []
    for file_name in os.listdir(role_dir):
        if file_name.endswith('.json'):
            annotation_files.append(os.path.join(role_dir, file_name))
    
    total_files = len(annotation_files)
    if total_files == 0:
        logger.info(f"角色 {role_name} 没有标注文件")
        return role_name, 0
    
    logger.info(f"角色 {role_name} 共有 {total_files} 个标注文件")
    
    # 处理每个标注文件
    processed_count = 0
    for annotation_file in annotation_files:
        try:
            # 读取标注文件
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            
            # 扩展标注信息
            enhanced_annotation = enhance_annotation(annotation, role_name)
            
            # 保存更新后的标注文件
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_annotation, f, indent=2, ensure_ascii=False)
            
            processed_count += 1
            logger.info(f"已扩展标注文件: {os.path.basename(annotation_file)}")
        except Exception as e:
            logger.error(f"处理标注文件失败: {annotation_file} - {str(e)}")
    
    logger.info(f"角色 {role_name} 标注扩展完成: {processed_count}/{total_files} 个文件")
    return role_name, processed_count

def main():
    """主函数"""
    logger.info("============================================================")
    logger.info("开始扩展标注信息")
    logger.info("============================================================")
    
    # 备份现有标注文件
    backup_annotations()
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotation_dir = os.path.join(script_dir, GLOBAL_CONFIG['annotation_dir'])
    
    # 获取所有角色目录
    role_dirs = []
    for item in os.listdir(annotation_dir):
        item_path = os.path.join(annotation_dir, item)
        if os.path.isdir(item_path):
            role_dirs.append((item, item_path))
    
    logger.info(f"发现 {len(role_dirs)} 个角色目录")
    
    # 处理每个角色的标注文件
    total_files = 0
    total_processed = 0
    
    for role_name, role_dir in role_dirs:
        # 获取角色标注文件数量
        role_files = len([f for f in os.listdir(role_dir) if f.endswith('.json')])
        total_files += role_files
        
        # 处理标注文件
        _, processed = process_role_annotations(role_name, role_dir)
        total_processed += processed
    
    logger.info("\n============================================================")
    logger.info("标注信息扩展完成")
    logger.info(f"总处理文件数: {total_files}")
    logger.info(f"总扩展文件数: {total_processed}")
    logger.info(f"扩展率: {total_processed / total_files * 100:.2f}%" if total_files > 0 else "无文件")
    logger.info("============================================================")

if __name__ == "__main__":
    main()
