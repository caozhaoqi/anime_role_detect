#!/usr/bin/env python3
"""
构建 FAISS 索引脚本
从训练数据中提取特征并构建索引，用于 CLIP 模型的分类
"""
import os
import sys
import numpy as np
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('build_faiss_index')

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 导入通用分类模块
try:
    from src.core.classification.general_classification import GeneralClassification, get_classifier
    logger.info("成功导入通用分类模块")
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    sys.exit(1)

def build_index_from_data_dir(data_dir, index_path="role_index", max_images_per_class=50):
    """从数据目录构建索引
    
    Args:
        data_dir: 数据目录
        index_path: 索引保存路径
        max_images_per_class: 每个类别的最大图像数量
    
    Returns:
        bool: 是否构建成功
    """
    logger.info(f"开始从目录构建索引: {data_dir}")
    logger.info(f"索引保存路径: {index_path}")
    logger.info(f"每个类别的最大图像数量: {max_images_per_class}")
    
    # 初始化分类器
    classifier = get_classifier()
    
    # 构建索引
    success = classifier.build_index_from_directory(data_dir)
    
    if success:
        # 保存索引
        save_path = os.path.join(project_root, index_path)
        save_success = classifier.save_index(save_path)
        
        if save_success:
            logger.info(f"索引构建成功并保存到: {save_path}")
            return True
        else:
            logger.error("索引构建成功，但保存失败")
            return False
    else:
        logger.error("索引构建失败")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='构建 FAISS 索引脚本')
    parser.add_argument('--data_dir', type=str, default='data/train', help='数据目录')
    parser.add_argument('--index_path', type=str, default='role_index', help='索引保存路径')
    parser.add_argument('--max_images_per_class', type=int, default=50, help='每个类别的最大图像数量')
    
    args = parser.parse_args()
    
    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        logger.error(f"数据目录不存在: {args.data_dir}")
        sys.exit(1)
    
    # 构建索引
    start_time = datetime.now()
    logger.info(f"开始构建索引，时间: {start_time}")
    
    success = build_index_from_data_dir(
        args.data_dir,
        args.index_path,
        args.max_images_per_class
    )
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    logger.info(f"索引构建完成，时间: {end_time}")
    logger.info(f"耗时: {elapsed.total_seconds():.2f} 秒")
    
    if success:
        logger.info("索引构建成功！")
        logger.info(f"现在可以使用此索引进行 CLIP 模型的分类")
        return 0
    else:
        logger.error("索引构建失败！")
        return 1

if __name__ == "__main__":
    sys.exit(main())
