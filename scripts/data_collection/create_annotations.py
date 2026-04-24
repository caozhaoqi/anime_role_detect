#!/usr/bin/env python3
"""
数据标注系统脚本
- 为图片添加角色名称、特征等标注信息
- 生成标注文件
"""

import os
import json
import logging
import hashlib
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='create_annotations.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# 全局配置
GLOBAL_CONFIG = {
    'image_dir': '../../data/role_images',
    'annotation_dir': '../../data/annotations',
    'annotation_format': 'json'  # 支持 json, xml, csv
}

# 角色特征映射
ROLE_FEATURES = {
    'a1luo2na4': {
        'name': '阿罗娜',
        'origin': '蔚蓝档案',
        'gender': 'female',
        'age': 'teen',
        'hair_color': 'pink',
        'eye_color': 'pink',
        'features': ['school uniform', 'pink hair', 'pink eyes', 'small', 'cute']
    },
    'hua1yin1': {
        'name': '华音',
        'origin': '蔚蓝档案',
        'gender': 'female',
        'age': 'teen',
        'hair_color': 'white',
        'eye_color': 'red',
        'features': ['school uniform', 'white hair', 'red eyes', 'serious', 'intelligent']
    },
    'shen1yue4': {
        'name': '神乐',
        'origin': '蔚蓝档案',
        'gender': 'female',
        'age': 'teen',
        'hair_color': 'black',
        'eye_color': 'purple',
        'features': ['school uniform', 'black hair', 'purple eyes', 'mischievous', 'energetic']
    },
    'li3shi4': {
        'name': '历史',
        'origin': '蔚蓝档案',
        'gender': 'female',
        'age': 'teen',
        'hair_color': 'brown',
        'eye_color': 'brown',
        'features': ['school uniform', 'brown hair', 'brown eyes', 'glasses', 'studious']
    },
    'lei3bei4': {
        'name': '雷贝',
        'origin': '蔚蓝档案',
        'gender': 'female',
        'age': 'teen',
        'hair_color': 'blonde',
        'eye_color': 'blue',
        'features': ['school uniform', 'blonde hair', 'blue eyes', 'athletic', 'confident']
    }
}

def ensure_directory(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def calculate_image_hash(image_path):
    """计算图片的哈希值"""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"计算图片哈希值失败: {image_path} - {str(e)}")
        return None

def get_image_info(image_path):
    """获取图片信息"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            format = img.format
            mode = img.mode
            return {
                'width': width,
                'height': height,
                'format': format,
                'mode': mode
            }
    except Exception as e:
        logger.warning(f"获取图片信息失败: {image_path} - {str(e)}")
        return {
            'width': 0,
            'height': 0,
            'format': 'unknown',
            'mode': 'unknown'
        }

def create_annotation(image_path, role_name, image_info):
    """创建图片标注"""
    # 基础标注信息
    annotation = {
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'role_name': role_name,
        'image_hash': calculate_image_hash(image_path),
        'image_info': image_info,
        'timestamp': os.path.getmtime(image_path),
        'annotated_at': os.path.getctime(image_path),
        'features': []
    }
    
    # 添加角色特征
    if role_name in ROLE_FEATURES:
        annotation.update(ROLE_FEATURES[role_name])
    else:
        # 为未配置的角色添加基础信息
        annotation.update({
            'name': role_name,
            'origin': 'unknown',
            'gender': 'unknown',
            'age': 'unknown',
            'hair_color': 'unknown',
            'eye_color': 'unknown',
            'features': [role_name]
        })
    
    return annotation

def save_annotation(annotation, annotation_dir, role_name, image_name):
    """保存标注文件"""
    # 确保角色标注目录存在
    role_annotation_dir = os.path.join(annotation_dir, role_name)
    ensure_directory(role_annotation_dir)
    
    # 生成标注文件名
    base_name = os.path.splitext(image_name)[0]
    if GLOBAL_CONFIG['annotation_format'] == 'json':
        annotation_file = os.path.join(role_annotation_dir, f"{base_name}.json")
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)
    elif GLOBAL_CONFIG['annotation_format'] == 'xml':
        # 生成XML格式标注
        xml_content = f'''
<annotation>
    <image_path>{annotation['image_path']}</image_path>
    <image_name>{annotation['image_name']}</image_name>
    <role_name>{annotation['role_name']}</role_name>
    <name>{annotation['name']}</name>
    <origin>{annotation['origin']}</origin>
    <gender>{annotation['gender']}</gender>
    <age>{annotation['age']}</age>
    <hair_color>{annotation['hair_color']}</hair_color>
    <eye_color>{annotation['eye_color']}</eye_color>
    <features>
'''
        for feature in annotation['features']:
            xml_content += f'        <feature>{feature}</feature>\n'
        xml_content += f'''
    </features>
    <image_info>
        <width>{annotation['image_info']['width']}</width>
        <height>{annotation['image_info']['height']}</height>
        <format>{annotation['image_info']['format']}</format>
        <mode>{annotation['image_info']['mode']}</mode>
    </image_info>
    <image_hash>{annotation['image_hash']}</image_hash>
    <timestamp>{annotation['timestamp']}</timestamp>
    <annotated_at>{annotation['annotated_at']}</annotated_at>
</annotation>
'''
        annotation_file = os.path.join(role_annotation_dir, f"{base_name}.xml")
        with open(annotation_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
    elif GLOBAL_CONFIG['annotation_format'] == 'csv':
        # 生成CSV格式标注
        import csv
        annotation_file = os.path.join(role_annotation_dir, f"{role_name}_annotations.csv")
        write_header = not os.path.exists(annotation_file)
        with open(annotation_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['image_path', 'image_name', 'role_name', 'name', 'origin', 'gender', 'age', 'hair_color', 'eye_color', 'features', 'width', 'height', 'format', 'mode', 'image_hash', 'timestamp', 'annotated_at'])
            writer.writerow([
                annotation['image_path'],
                annotation['image_name'],
                annotation['role_name'],
                annotation['name'],
                annotation['origin'],
                annotation['gender'],
                annotation['age'],
                annotation['hair_color'],
                annotation['eye_color'],
                ','.join(annotation['features']),
                annotation['image_info']['width'],
                annotation['image_info']['height'],
                annotation['image_info']['format'],
                annotation['image_info']['mode'],
                annotation['image_hash'],
                annotation['timestamp'],
                annotation['annotated_at']
            ])
    
    return annotation_file

def process_role_annotations(role_name, role_dir, annotation_dir):
    """处理单个角色的图片标注"""
    logger.info(f"开始处理角色 {role_name} 的图片标注")
    
    # 获取角色目录下的所有图片
    image_files = []
    for file_name in os.listdir(role_dir):
        file_path = os.path.join(role_dir, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            image_files.append(file_path)
    
    total_images = len(image_files)
    if total_images == 0:
        logger.info(f"角色 {role_name} 没有图片")
        return role_name, 0
    
    logger.info(f"角色 {role_name} 共有 {total_images} 张图片")
    
    # 为每张图片创建标注
    annotated_count = 0
    for image_path in image_files:
        try:
            # 获取图片信息
            image_info = get_image_info(image_path)
            
            # 创建标注
            annotation = create_annotation(image_path, role_name, image_info)
            
            # 保存标注
            image_name = os.path.basename(image_path)
            annotation_file = save_annotation(annotation, annotation_dir, role_name, image_name)
            
            annotated_count += 1
            logger.info(f"已为图片 {image_name} 创建标注: {annotation_file}")
        except Exception as e:
            logger.error(f"处理图片标注失败: {image_path} - {str(e)}")
    
    logger.info(f"角色 {role_name} 标注完成: {annotated_count}/{total_images} 张图片")
    return role_name, annotated_count

def main():
    """主函数"""
    logger.info("============================================================")
    logger.info("开始创建数据标注")
    logger.info("============================================================")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, GLOBAL_CONFIG['image_dir'])
    annotation_dir = os.path.join(script_dir, GLOBAL_CONFIG['annotation_dir'])
    
    # 确保目录存在
    ensure_directory(annotation_dir)
    
    # 获取所有角色目录
    role_dirs = []
    for item in os.listdir(image_dir):
        item_path = os.path.join(image_dir, item)
        if os.path.isdir(item_path):
            role_dirs.append((item, item_path))
    
    logger.info(f"发现 {len(role_dirs)} 个角色目录")
    
    # 处理每个角色的图片标注
    total_images = 0
    total_annotated = 0
    
    for role_name, role_dir in role_dirs:
        # 获取角色图片数量
        role_images = len([f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        total_images += role_images
        
        # 处理标注
        _, annotated = process_role_annotations(role_name, role_dir, annotation_dir)
        total_annotated += annotated
    
    logger.info("\n============================================================")
    logger.info("数据标注创建完成")
    logger.info(f"总处理图片数: {total_images}")
    logger.info(f"总标注图片数: {total_annotated}")
    logger.info(f"标注率: {total_annotated / total_images * 100:.2f}%" if total_images > 0 else "无图片")
    logger.info("============================================================")

if __name__ == "__main__":
    main()
