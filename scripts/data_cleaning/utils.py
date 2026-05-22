#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗工具库 - 整合重复功能
提供统一的：
- MD5计算
- 去重
- 文件扫描
- 质量检查
"""

import os
import hashlib
import json
from collections import defaultdict
from PIL import Image
from typing import List, Dict, Tuple, Optional

# ==================== 常量 ====================
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')
MIN_FILE_SIZE_KB = 10
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

# ==================== 文件扫描 ====================
def scan_images(data_dir: str) -> List[str]:
    """扫描目录中的所有图片文件"""
    image_files = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_files.append(os.path.join(root, filename))
    return image_files


def scan_role_images(data_dir: str) -> Dict[str, List[str]]:
    """按角色分组扫描图片"""
    role_images = defaultdict(list)
    for root, _, files in os.walk(data_dir):
        role = os.path.basename(root)
        parent_dir = os.path.basename(os.path.dirname(root))
        
        # 判断是否为角色目录
        if (os.path.dirname(root) == data_dir) or (parent_dir in ['expanded_dataset', 'merged_dataset', 'final_dataset']):
            for filename in files:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    role_images[role].append(os.path.join(root, filename))
    return dict(role_images)

# ==================== MD5计算 ====================
def calculate_md5(file_path: str, chunk_size: int = 8192) -> Optional[str]:
    """计算文件MD5哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"❌ 计算MD5失败: {file_path} - {str(e)}")
        return None


def batch_calculate_md5(file_list: List[str]) -> Dict[str, List[str]]:
    """批量计算MD5，返回hash到文件列表的映射"""
    hash_to_files = defaultdict(list)
    for file_path in file_list:
        file_hash = calculate_md5(file_path)
        if file_hash:
            hash_to_files[file_hash].append(file_path)
    return dict(hash_to_files)

# ==================== 去重 ====================
def find_duplicate_files(file_list: List[str]) -> Dict[str, List[str]]:
    """从文件列表中查找重复文件"""
    hash_to_files = batch_calculate_md5(file_list)
    duplicates = {h: fs for h, fs in hash_to_files.items() if len(fs) > 1}
    return duplicates


def get_deletion_candidates(duplicates: Dict[str, List[str]], 
                           prefer_shorter: bool = True, 
                           exclude_prefixes: List[str] = ['crop', 'temp', 'tmp']) -> List[str]:
    """根据策略确定要删除的文件列表
    
    Args:
        duplicates: hash到文件列表的映射
        prefer_shorter: 优先保留文件名较短的
        exclude_prefixes: 优先删除包含这些前缀的文件
        
    Returns:
        要删除的文件列表
    """
    to_delete = []
    
    for file_hash, files in duplicates.items():
        if len(files) <= 1:
            continue
        
        # 评分排序：分数越高越应该保留
        scored_files = []
        for file_path in files:
            score = 0
            filename = os.path.basename(file_path)
            
            # 优先删除包含排除前缀的
            has_excluded = any(prefix in filename.lower() for prefix in exclude_prefixes)
            if not has_excluded:
                score += 100
            
            # 优先保留较短的文件名
            if prefer_shorter:
                score -= len(filename)
            
            scored_files.append((score, file_path))
        
        # 按分数降序排序，分数高的保留
        scored_files.sort(key=lambda x: x[0], reverse=True)
        
        # 保留第一个，删除其余
        to_delete.extend([f for (_, f) in scored_files[1:]])
    
    return to_delete


def delete_files(file_list: List[str]) -> Tuple[int, int]:
    """批量删除文件
    
    Returns:
        (成功数, 失败数)
    """
    success = 0
    failed = 0
    
    for file_path in file_list:
        try:
            os.remove(file_path)
            success += 1
        except Exception as e:
            print(f"❌ 删除失败: {file_path} - {str(e)}")
            failed += 1
    
    return success, failed

# ==================== 质量检查 ====================
def check_image_quality(file_path: str) -> Tuple[bool, Optional[str]]:
    """检查图片质量
    
    Returns:
        (是否合格, 不合格原因)
    """
    try:
        # 文件大小检查
        file_size_kb = os.path.getsize(file_path) / 1024
        if file_size_kb < MIN_FILE_SIZE_KB:
            return False, f"文件过小 ({file_size_kb:.1f}KB)"
        
        with Image.open(file_path) as img:
            img.verify()  # 验证完整性
            
            # 重新打开获取尺寸
            img = Image.open(file_path)
            width, height = img.size
            
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                return False, f"尺寸过小 ({width}x{height})"
            
            # 纯色检查
            if len(set(img.convert('RGB').getdata())) < 10:
                return False, "纯色图片"
        
        return True, None
    except Exception as e:
        return False, f"文件损坏: {str(e)}"


def batch_quality_check(file_list: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """批量质量检查
    
    Returns:
        (合格文件列表, 不合格文件{路径:原因})
    """
    passed = []
    failed = {}
    
    for file_path in file_list:
        ok, reason = check_image_quality(file_path)
        if ok:
            passed.append(file_path)
        else:
            failed[file_path] = reason
    
    return passed, failed

# ==================== 保存/加载 ====================
def save_json(data, output_path: str):
    """保存数据到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(input_path: str):
    """从JSON文件加载数据"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==================== 报告生成 ====================
def generate_size_report(file_list: List[str]) -> Dict[str, int]:
    """生成文件大小分布报告"""
    size_buckets = defaultdict(int)
    
    for file_path in file_list:
        try:
            size_kb = os.path.getsize(file_path) / 1024
            if size_kb < 50:
                size_buckets['<50KB'] += 1
            elif size_kb < 100:
                size_buckets['50-100KB'] += 1
            elif size_kb < 500:
                size_buckets['100-500KB'] += 1
            elif size_kb < 1000:
                size_buckets['500KB-1MB'] += 1
            else:
                size_buckets['>1MB'] += 1
        except:
            size_buckets['error'] += 1
    
    return dict(size_buckets)
