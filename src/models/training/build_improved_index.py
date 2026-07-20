#!/usr/bin/env python3
"""
构建改进的特征索引
使用增强后的数据构建更全面的特征索引，提高识别率
"""
import os
import sys
import argparse
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("build_improved_index")


from src.core.general_classification import build_index_from_directory


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构建改进的特征索引")
    parser.add_argument(
        "--data_dir", type=str, default="data/augmented_characters", help="数据目录"
    )
    parser.add_argument(
        "--output_index", type=str, default="models/improved_character_index", help="输出索引路径"
    )

    args = parser.parse_args()

    logger.info("开始构建改进的特征索引")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出索引路径: {args.output_index}")

    # 构建索引
    success = build_index_from_directory(args.data_dir)

    if success:
        logger.info("🎉 改进的特征索引构建成功！")
        print(f"\n🎉 改进的特征索引构建成功！")
        print(f"数据目录: {args.data_dir}")
        print(f"索引已构建并加载到系统中")
    else:
        logger.error("❌ 改进的特征索引构建失败！")
        print(f"\n❌ 改进的特征索引构建失败！")


if __name__ == "__main__":
    main()
