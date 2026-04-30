#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测并删除无法打开的损坏图片
"""

import os
import sys
import logging

# 尝试导入图片处理库
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("警告: PIL库未安装，将使用简单的文件头检测")

# 配置
DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_image_corrupted(file_path):
    """检查图片是否损坏"""
    if not PIL_AVAILABLE:
        return is_corrupted_simple(file_path)
    
    try:
        # 尝试打开图片
        with Image.open(file_path) as img:
            # 尝试读取图片数据
            img.verify()
            # 重新打开并尝试加载像素数据
            img = Image.open(file_path)
            img.load()
            return False
    except Exception as e:
        logger.debug(f"图片损坏: {file_path} - {str(e)}")
        return True


def is_corrupted_simple(file_path):
    """简单的文件头检测（当PIL不可用时）"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 常见图片格式的文件头
    file_signatures = {
        '.jpg': b'\xFF\xD8\xFF',
        '.jpeg': b'\xFF\xD8\xFF',
        '.png': b'\x89PNG\r\n\x1a\n',
        '.webp': b'RIFF....WEBP',
        '.gif': b'GIF8',
        '.bmp': b'BM'
    }
    
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
        
        expected_sig = file_signatures.get(file_ext)
        if expected_sig:
            # 检查文件头是否匹配
            if not header.startswith(expected_sig[:len(header)]):
                return True
        
        # 检查文件大小
        if os.path.getsize(file_path) < 10:
            return True
        
        return False
    except Exception:
        return True


def scan_and_remove_corrupted(root_dir):
    """扫描并删除损坏的图片"""
    stats = {
        'total_scanned': 0,
        'total_deleted': 0,
        'total_skipped': 0,
        'total_failed': 0,
        'by_role': {}
    }
    
    logger.info(f"开始扫描 {root_dir} 目录...")
    logger.info(f"PIL可用: {PIL_AVAILABLE}")
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        role_name = os.path.basename(dirpath)
        
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                file_path = os.path.join(dirpath, filename)
                stats['total_scanned'] += 1
                
                try:
                    if is_image_corrupted(file_path):
                        os.remove(file_path)
                        stats['total_deleted'] += 1
                        
                        if role_name not in stats['by_role']:
                            stats['by_role'][role_name] = {'scanned': 0, 'deleted': 0}
                        stats['by_role'][role_name]['deleted'] += 1
                        
                        logger.info(f"删除损坏图片: {file_path}")
                    else:
                        if role_name not in stats['by_role']:
                            stats['by_role'][role_name] = {'scanned': 0, 'deleted': 0}
                        stats['by_role'][role_name]['scanned'] += 1
                        stats['total_skipped'] += 1
                        
                except Exception as e:
                    logger.error(f"处理失败 {file_path}: {str(e)}")
                    stats['total_failed'] += 1
        
        if stats['total_scanned'] % 100 == 0:
            logger.info(f"已扫描 {stats['total_scanned']} 张图片")
    
    return stats


def main():
    """主函数"""
    if not PIL_AVAILABLE:
        logger.warning("PIL库未安装，将使用简单的文件头检测方法")
    
    stats = scan_and_remove_corrupted(DATA_DIR)
    
    # 输出统计结果
    logger.info("\n=== 检测完成 ===")
    logger.info(f"总扫描: {stats['total_scanned']} 张")
    logger.info(f"已删除: {stats['total_deleted']} 张")
    logger.info(f"正常跳过: {stats['total_skipped']} 张")
    logger.info(f"处理失败: {stats['total_failed']} 张")
    
    # 保存检测报告
    report_path = os.path.join(DATA_DIR, 'corruption_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"检测时间: {__import__('datetime').datetime.now()}\n")
        f.write(f"总扫描: {stats['total_scanned']} 张\n")
        f.write(f"已删除: {stats['total_deleted']} 张\n")
        f.write(f"正常跳过: {stats['total_skipped']} 张\n")
        f.write(f"处理失败: {stats['total_failed']} 张\n")
        
        if stats['by_role']:
            f.write("\n=== 角色统计 ===")
            for role_name, role_stats in stats['by_role'].items():
                if role_stats['deleted'] > 0:
                    f.write(f"\n{role_name}: 扫描 {role_stats['scanned']} 张, 删除 {role_stats['deleted']} 张")
    
    logger.info(f"\n检测报告已保存: {report_path}")


if __name__ == '__main__':
    main()
