#!/usr/bin/env python3
"""测试数据流水线"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.pipeline import DataPipeline


def test_pipeline():
    """测试流水线功能"""
    print("🧪 开始测试数据流水线\n")

    # 创建流水线
    pipeline = DataPipeline()

    # 测试1: 筛选困难样本
    print("\n" + "=" * 60)
    print("测试1: 筛选困难样本")
    print("=" * 60)
    difficult_samples = pipeline.filter_difficult_samples(batch_size=999999, strategy='confidence')
    print(f"✅ 找到 {len(difficult_samples)} 个困难样本")

    # 测试2: 创建审核批次
    print("\n" + "=" * 60)
    print("测试2: 创建审核批次")
    print("=" * 60)
    if difficult_samples:
        batch_id = pipeline.create_review_batch(difficult_samples, "test_batch")
        print(f"✅ 审核批次创建成功: {batch_id}")
    else:
        print("⚠️ 没有困难样本，跳过")

    # 测试3: 查看统计信息
    print("\n" + "=" * 60)
    print("测试3: 查看统计信息")
    print("=" * 60)
    pipeline.print_stats()

    print("\n✅ 流水线测试完成！")


if __name__ == "__main__":
    test_pipeline()