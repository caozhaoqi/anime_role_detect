#!/usr/bin/env python3
"""
运行日志融合脚本
从日志中融合特征并构建新模型
"""
import os
import sys
import argparse
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run_log_fusion')

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.log_fusion.log_fusion import LogFusion


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行日志融合脚本')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--model_dir', type=str, default='./models', help='模型目录')
    parser.add_argument('--max_logs', type=int, default=5, help='最大日志数量')
    parser.add_argument('--fusion_method', type=str, default='mean', help='融合方法')
    parser.add_argument('--model_name', type=str, default=None, help='模型名称')
    
    args = parser.parse_args()
    
    logger.info("开始运行日志融合脚本")
    logger.info(f"日志目录: {args.log_dir}")
    logger.info(f"模型目录: {args.model_dir}")
    logger.info(f"最大日志数量: {args.max_logs}")
    logger.info(f"融合方法: {args.fusion_method}")
    
    # 初始化日志融合模块
    fusion = LogFusion(log_dir=args.log_dir, model_dir=args.model_dir)
    
    # 运行融合流程
    model_path = fusion.run_fusion(
        max_logs=args.max_logs,
        fusion_method=args.fusion_method,
        model_name=args.model_name
    )
    
    if model_path:
        logger.info(f"🎉 日志融合成功！新模型路径: {model_path}")
        print(f"\n🎉 日志融合成功！")
        print(f"新模型路径: {model_path}")
        print(f"融合了 {args.max_logs} 条日志")
        print(f"使用融合方法: {args.fusion_method}")
    else:
        logger.error("❌ 日志融合失败！")
        print(f"\n❌ 日志融合失败！")


if __name__ == "__main__":
    main()
