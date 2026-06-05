#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据流水线集成测试
Integration Tests for Data Pipeline
"""
import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.pipeline import DataPipeline


class TestDataPipelineIntegration(unittest.TestCase):
    """数据流水线集成测试类"""

    @classmethod
    def setUpClass(cls):
        """在所有测试之前执行一次"""
        print("📥 初始化数据流水线...")
        cls.pipeline = DataPipeline()

    def test_initialization(self):
        """测试流水线初始化"""
        self.assertIsNotNone(self.pipeline.session)
        self.assertIsNotNone(self.pipeline.stats)
        self.assertEqual(self.pipeline.stats['total_samples'], 0)

    def test_retry_decorator(self):
        """测试重试机制"""
        call_count = 0

        @self.pipeline.retry_on_error(max_retries=3, delay=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("模拟错误")
            return "success"

        result = flaky_function()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)

    def test_retry_decorator_failure(self):
        """测试重试机制最终失败"""
        @self.pipeline.retry_on_error(max_retries=2, delay=0.1)
        def always_fail():
            raise Exception("始终失败")

        with self.assertRaises(Exception):
            always_fail()

    def test_monitor_performance(self):
        """测试性能监控"""
        @self.pipeline.monitor_performance("test_func")
        def test_function():
            return "completed"

        result = test_function()
        self.assertEqual(result, "completed")
        self.assertIn("test_func", self.pipeline.performance_metrics)
        self.assertEqual(self.pipeline.performance_metrics["test_func"]["status"], "success")

    def test_import_samples(self):
        """测试样本导入"""
        # 使用测试数据目录
        test_dir = project_root / "data" / "final_dataset"
        if test_dir.exists():
            count = self.pipeline.import_samples(str(test_dir))
            self.assertIsInstance(count, int)
            print(f"✅ 导入测试完成，导入 {count} 个样本")

    def test_stats_tracking(self):
        """测试统计信息跟踪"""
        # 验证统计信息结构
        self.assertIn('start_time', self.pipeline.stats)
        self.assertIn('end_time', self.pipeline.stats)
        self.assertIn('errors', self.pipeline.stats)
        self.assertIsInstance(self.pipeline.stats['errors'], list)

    def test_performance_metrics(self):
        """测试性能指标"""
        self.assertIsInstance(self.pipeline.performance_metrics, dict)

    @classmethod
    def tearDownClass(cls):
        """在所有测试之后执行一次"""
        # 关闭数据库连接
        if hasattr(cls.pipeline, 'session'):
            cls.pipeline.session.close()
        if hasattr(cls.pipeline, 'engine'):
            cls.pipeline.engine.dispose()
        print("✅ 集成测试完成")


class TestDataPipelinePerformance(unittest.TestCase):
    """数据流水线性能测试类"""

    @classmethod
    def setUpClass(cls):
        """在所有测试之前执行一次"""
        cls.pipeline = DataPipeline()

    def test_parallel_processing(self):
        """测试并行处理能力"""
        import time

        # 模拟并行处理任务
        def task(n):
            time.sleep(0.1)
            return n * 2

        start = time.time()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(task, range(10)))
        elapsed = time.time() - start

        # 验证结果
        self.assertEqual(len(results), 10)
        self.assertEqual(results[0], 0)
        self.assertEqual(results[5], 10)

        # 并行处理应该比串行快（串行需要1秒）
        self.assertLess(elapsed, 0.5)
        print(f"✅ 并行处理测试完成，耗时: {elapsed:.3f}秒")

    def test_memory_usage(self):
        """测试内存使用情况"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # 执行一些操作
        _ = self.pipeline.stats.copy()

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_diff = mem_after - mem_before

        # 内存增长应该合理（小于50MB）
        self.assertLess(mem_diff, 50)
        print(f"✅ 内存使用测试完成，增长: {mem_diff:.2f}MB")

    @classmethod
    def tearDownClass(cls):
        """在所有测试之后执行一次"""
        if hasattr(cls.pipeline, 'session'):
            cls.pipeline.session.close()
        if hasattr(cls.pipeline, 'engine'):
            cls.pipeline.engine.dispose()


if __name__ == "__main__":
    unittest.main()
