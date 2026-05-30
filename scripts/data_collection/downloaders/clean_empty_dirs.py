#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理空目录脚本
删除 data 目录下所有空目录
"""

import os
import logging

# 配置
DATA_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def is_empty_dir(dir_path):
    """检查目录是否为空"""
    try:
        return len(os.listdir(dir_path)) == 0
    except OSError:
        return False


def clean_empty_dirs(root_dir):
    """清理所有空目录"""
    empty_dirs = []

    # 先收集所有空目录（从深到浅）
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if is_empty_dir(full_path):
                empty_dirs.append(full_path)

    # 删除空目录
    deleted_count = 0
    for empty_dir in empty_dirs:
        try:
            os.rmdir(empty_dir)
            logger.info(f"删除空目录: {empty_dir}")
            deleted_count += 1
        except OSError as e:
            logger.warning(f"删除失败 {empty_dir}: {str(e)}")

    return deleted_count, len(empty_dirs) - deleted_count


def main():
    """主函数"""
    logger.info(f"开始清理 {DATA_DIR} 下的空目录...")

    deleted_count, failed_count = clean_empty_dirs(DATA_DIR)

    logger.info(f"\n=== 清理完成 ===")
    logger.info(f"删除空目录: {deleted_count} 个")
    logger.info(f"删除失败: {failed_count} 个")


if __name__ == "__main__":
    main()
