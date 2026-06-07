#!/usr/bin/env python3
"""
数据流水线CLI工具
Data Pipeline CLI Tool
"""
import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.pipeline import DataPipeline


def cmd_import(args):
    """导入样本"""
    pipeline = DataPipeline()
    count = pipeline.import_samples(args.dir)
    print(f"\n✅ 成功导入 {count} 个样本")


def cmd_deduplicate(args):
    """去重样本"""
    pipeline = DataPipeline()
    retained, removed = pipeline.deduplicate_samples(
        phash_threshold=args.phash,
        clip_threshold=args.clip
    )
    print(f"\n✅ 保留 {retained} 个样本，去除 {removed} 个重复样本")


def cmd_annotate(args):
    """自动标注"""
    pipeline = DataPipeline()
    count = pipeline.annotate_samples(
        conf_threshold=args.conf,
        limit=args.limit
    )
    print(f"\n✅ 成功标注 {count} 个样本")


def cmd_filter(args):
    """筛选困难样本"""
    pipeline = DataPipeline()
    sample_ids = pipeline.filter_difficult_samples(
        batch_size=args.batch,
        strategy=args.strategy
    )
    print(f"\n✅ 找到 {len(sample_ids)} 个困难样本")
    if args.create_batch:
        batch_id = pipeline.create_review_batch(sample_ids, args.batch_name)
        print(f"✅ 审核批次已创建: {batch_id}")


def cmd_review(args):
    """创建审核批次"""
    pipeline = DataPipeline()
    batch_id = pipeline.create_review_batch(
        sample_ids=args.sample_ids,
        batch_name=args.batch_name
    )
    print(f"\n✅ 审核批次已创建: {batch_id}")


def cmd_run(args):
    """运行完整流水线"""
    pipeline = DataPipeline()
    stats = pipeline.run_full_pipeline(
        data_dir=args.dir,
        auto_review=args.auto_review
    )
    print(f"\n✅ 流水线执行完成！")


def cmd_stats(args):
    """查看统计信息"""
    pipeline = DataPipeline()
    pipeline.print_stats()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="数据流水线CLI工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 导入样本
  python scripts/pipeline_cli.py import -d data/final_dataset

  # 去重样本
  python scripts/pipeline_cli.py deduplicate --phash 5 --clip 0.98

  # 自动标注
  python scripts/pipeline_cli.py annotate --conf 0.5 --limit 100

  # 筛选困难样本
  python scripts/pipeline_cli.py filter --batch 10 --strategy confidence

  # 运行完整流水线
  python scripts/pipeline_cli.py run -d data/final_dataset
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 导入命令
    import_parser = subparsers.add_parser('import', help='导入样本图片')
    import_parser.add_argument('-d', '--dir', default='data/final_dataset', help='数据目录')
    import_parser.set_defaults(func=cmd_import)

    # 去重命令
    dedup_parser = subparsers.add_parser('deduplicate', help='去重样本')
    dedup_parser.add_argument('--phash', type=int, default=5, help='感知哈希阈值')
    dedup_parser.add_argument('--clip', type=float, default=0.98, help='CLIP相似度阈值')
    dedup_parser.set_defaults(func=cmd_deduplicate)

    # 标注命令
    annotate_parser = subparsers.add_parser('annotate', help='自动标注样本')
    annotate_parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    annotate_parser.add_argument('--limit', type=int, default=None, help='处理数量限制')
    annotate_parser.set_defaults(func=cmd_annotate)

    # 筛选命令
    filter_parser = subparsers.add_parser('filter', help='筛选困难样本')
    filter_parser.add_argument('--batch', type=int, default=10, help='批次大小')
    filter_parser.add_argument('--strategy', default='confidence', choices=['confidence', 'entropy', 'margin'], help='筛选策略')
    filter_parser.add_argument('--create-batch', action='store_true', help='创建审核批次')
    filter_parser.add_argument('--batch-name', default=None, help='批次名称')
    filter_parser.set_defaults(func=cmd_filter)

    # 审核命令
    review_parser = subparsers.add_parser('review', help='创建审核批次')
    review_parser.add_argument('sample_ids', nargs='+', type=int, help='样本ID列表')
    review_parser.add_argument('--batch-name', default=None, help='批次名称')
    review_parser.set_defaults(func=cmd_review)

    # 运行命令
    run_parser = subparsers.add_parser('run', help='运行完整流水线')
    run_parser.add_argument('-d', '--dir', default='data/final_dataset', help='数据目录')
    run_parser.add_argument('--auto-review', action='store_true', help='自动审核')
    run_parser.set_defaults(func=cmd_run)

    # 统计命令
    stats_parser = subparsers.add_parser('stats', help='查看统计信息')
    stats_parser.set_defaults(func=cmd_stats)

    # 解析参数
    args = parser.parse_args()

    # 执行命令
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()