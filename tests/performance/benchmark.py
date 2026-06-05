#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
压力测试工具
测试系统性能和并发处理能力
"""

import os
import sys
import time
import json
import random
import statistics
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Callable
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.logging.global_logger import get_logger

logger = get_logger("benchmark")


class PerformanceBenchmark:
    """
    性能基准测试类
    """
    
    def __init__(self, output_dir: str = "tests/performance/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def benchmark_single_inference(
        self,
        inference_func: Callable,
        test_images: List[str],
        warmup: int = 3,
        iterations: int = 50,
    ) -> Dict:
        """
        测试单次推理性能
        
        Args:
            inference_func: 推理函数
            test_images: 测试图片列表
            warmup: 预热次数
            iterations: 测试次数
            
        Returns:
            性能统计结果
        """
        logger.info(f"开始单次推理测试 (预热{warmup}次, 测试{iterations}次)")
        
        # 预热
        for i in range(warmup):
            img = random.choice(test_images)
            inference_func(img)
            logger.info(f"预热 {i+1}/{warmup}")
        
        # 正式测试
        latencies = []
        for i in range(iterations):
            img = random.choice(test_images)
            
            start = time.time()
            result = inference_func(img)
            end = time.time()
            
            latency = (end - start) * 1000  # 转为毫秒
            latencies.append(latency)
            
            if (i + 1) % 10 == 0:
                logger.info(f"测试进度 {i+1}/{iterations}, 当前延迟: {latency:.2f}ms")
        
        # 统计
        stats = {
            "test_type": "single_inference",
            "total_iterations": iterations,
            "avg_latency_ms": statistics.mean(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": self._percentile(latencies, 95),
            "p99_latency_ms": self._percentile(latencies, 99),
            "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "throughput_qps": 1000 / statistics.mean(latencies),
        }
        
        self._print_stats(stats)
        return stats
    
    def benchmark_batch_inference(
        self,
        inference_func: Callable,
        test_images: List[str],
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        iterations_per_batch: int = 10,
    ) -> List[Dict]:
        """
        测试批量推理性能
        
        Args:
            inference_func: 批量推理函数
            test_images: 测试图片列表
            batch_sizes: 批处理大小列表
            iterations_per_batch: 每个批大小的测试次数
            
        Returns:
            各批大小的性能统计
        """
        logger.info(f"开始批量推理测试，批大小: {batch_sizes}")
        
        all_stats = []
        
        for batch_size in batch_sizes:
            logger.info(f"测试批大小: {batch_size}")
            
            latencies = []
            for i in range(iterations_per_batch):
                # 随机选择图片
                batch = random.choices(test_images, k=batch_size)
                
                start = time.time()
                results = inference_func(batch)
                end = time.time()
                
                latency = (end - start) * 1000
                latencies.append(latency)
            
            stats = {
                "test_type": "batch_inference",
                "batch_size": batch_size,
                "avg_latency_ms": statistics.mean(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "median_latency_ms": statistics.median(latencies),
                "p95_latency_ms": self._percentile(latencies, 95),
                "throughput_qps": (batch_size * 1000) / statistics.mean(latencies),
                "latency_per_image_ms": statistics.mean(latencies) / batch_size,
            }
            
            all_stats.append(stats)
            self._print_stats(stats)
        
        return all_stats
    
    def benchmark_concurrent_inference(
        self,
        inference_func: Callable,
        test_images: List[str],
        concurrency_levels: List[int] = [1, 5, 10, 20, 50],
        requests_per_level: int = 100,
    ) -> List[Dict]:
        """
        测试并发推理性能
        
        Args:
            inference_func: 推理函数
            test_images: 测试图片列表
            concurrency_levels: 并发级别列表
            requests_per_level: 每个并发级别的请求数
            
        Returns:
            各并发级别的性能统计
        """
        logger.info(f"开始并发推理测试，并发级别: {concurrency_levels}")
        
        all_stats = []
        
        for concurrency in concurrency_levels:
            logger.info(f"测试并发数: {concurrency}")
            
            latencies = []
            errors = 0
            
            def worker():
                img = random.choice(test_images)
                start = time.time()
                try:
                    inference_func(img)
                    return (time.time() - start) * 1000
                except Exception as e:
                    logger.error(f"推理错误: {e}")
                    return None
            
            start_total = time.time()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(worker) for _ in range(requests_per_level)]
                
                for future in as_completed(futures):
                    latency = future.result()
                    if latency is not None:
                        latencies.append(latency)
                    else:
                        errors += 1
            
            total_time = (time.time() - start_total) * 1000
            
            stats = {
                "test_type": "concurrent_inference",
                "concurrency": concurrency,
                "total_requests": requests_per_level,
                "successful_requests": len(latencies),
                "failed_requests": errors,
                "error_rate": errors / requests_per_level * 100,
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                "min_latency_ms": min(latencies) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0,
                "p95_latency_ms": self._percentile(latencies, 95) if latencies else 0,
                "p99_latency_ms": self._percentile(latencies, 99) if latencies else 0,
                "total_time_ms": total_time,
                "throughput_qps": (len(latencies) * 1000) / total_time if total_time > 0 else 0,
            }
            
            all_stats.append(stats)
            self._print_stats(stats)
        
        return all_stats
    
    def benchmark_feature_store(
        self,
        feature_store,
        num_features: int = 10000,
        num_queries: int = 100,
        dimensions: int = 512,
    ) -> Dict:
        """
        测试特征库性能
        
        Args:
            feature_store: 特征库实例
            num_features: 特征数量
            num_queries: 查询次数
            dimensions: 特征维度
            
        Returns:
            性能统计
        """
        logger.info(f"开始特征库性能测试 (特征数: {num_features}, 查询: {num_queries})")
        
        import numpy as np
        
        # 生成随机特征
        features = np.random.randn(num_features, dimensions).astype(np.float32)
        # 归一化
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        # 添加特征到库中
        logger.info("添加特征到库中...")
        start = time.time()
        for i in range(num_features):
            feature_store.add_features_to_character(f"character_{i%100}", [features[i]])
        add_time = (time.time() - start) * 1000
        
        # 查询测试
        query_features = np.random.randn(num_queries, dimensions).astype(np.float32)
        query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
        
        logger.info("开始查询测试...")
        latencies = []
        for i in range(num_queries):
            start = time.time()
            results = feature_store.search(query_features[i], top_k=10)
            end = time.time()
            latencies.append((end - start) * 1000)
        
        stats = {
            "test_type": "feature_store",
            "num_features": num_features,
            "num_queries": num_queries,
            "dimensions": dimensions,
            "add_time_total_ms": add_time,
            "add_time_per_feature_ms": add_time / num_features,
            "avg_query_latency_ms": statistics.mean(latencies),
            "min_query_latency_ms": min(latencies),
            "max_query_latency_ms": max(latencies),
            "p95_query_latency_ms": self._percentile(latencies, 95),
            "throughput_qps": (num_queries * 1000) / sum(latencies),
        }
        
        self._print_stats(stats)
        return stats
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _print_stats(self, stats: Dict):
        """打印统计信息"""
        logger.info("=" * 50)
        logger.info(f"测试类型: {stats.get('test_type', 'unknown')}")
        for key, value in stats.items():
            if key != "test_type":
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.2f}")
                else:
                    logger.info(f"  {key}: {value}")
        logger.info("=" * 50)
    
    def save_results(self, results: List[Dict], filename: Optional[str] = None):
        """
        保存测试结果
        
        Args:
            results: 结果列表
            filename: 文件名，None则使用默认名称
        """
        if filename is None:
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到: {output_path}")


def run_full_benchmark(test_images_dir: str, model_name: str = "ViT-B/32"):
    """
    运行完整基准测试
    
    Args:
        test_images_dir: 测试图片目录
        model_name: CLIP模型名称
    """
    from src.core.recognition.clip_embedder import CLIPEmbedder
    from src.core.recognition.feature_store import FeatureStore
    
    # 获取测试图片
    test_images = list(Path(test_images_dir).glob("*.jpg"))
    test_images.extend(Path(test_images_dir).glob("*.png"))
    test_images = [str(p) for p in test_images]
    
    if not test_images:
        logger.error(f"未找到测试图片: {test_images_dir}")
        return
    
    logger.info(f"找到 {len(test_images)} 张测试图片")
    
    # 初始化组件
    embedder = CLIPEmbedder(model_name=model_name)
    feature_store = FeatureStore(dim=embedder.embedding_dim)
    
    benchmark = PerformanceBenchmark()
    all_results = []
    
    # 1. 单次推理测试
    logger.info("\n" + "="*60)
    logger.info("测试1: 单次推理性能")
    logger.info("="*60)
    result = benchmark.benchmark_single_inference(
        embedder.embed_image,
        test_images,
        warmup=3,
        iterations=50,
    )
    all_results.append(result)
    
    # 2. 批量推理测试
    logger.info("\n" + "="*60)
    logger.info("测试2: 批量推理性能")
    logger.info("="*60)
    batch_results = benchmark.benchmark_batch_inference(
        embedder.embed_images,
        test_images,
        batch_sizes=[1, 4, 8, 16],
        iterations_per_batch=10,
    )
    all_results.extend(batch_results)
    
    # 3. 并发推理测试
    logger.info("\n" + "="*60)
    logger.info("测试3: 并发推理性能")
    logger.info("="*60)
    concurrent_results = benchmark.benchmark_concurrent_inference(
        embedder.embed_image,
        test_images,
        concurrency_levels=[1, 5, 10, 20],
        requests_per_level=100,
    )
    all_results.extend(concurrent_results)
    
    # 4. 特征库测试
    logger.info("\n" + "="*60)
    logger.info("测试4: 特征库性能")
    logger.info("="*60)
    feature_result = benchmark.benchmark_feature_store(
        feature_store,
        num_features=5000,
        num_queries=100,
        dimensions=embedder.embedding_dim,
    )
    all_results.append(feature_result)
    
    # 保存结果
    benchmark.save_results(all_results)
    
    logger.info("\n" + "="*60)
    logger.info("所有测试完成！")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="性能基准测试工具")
    parser.add_argument("--test-images", type=str, required=True,
                        help="测试图片目录")
    parser.add_argument("--model", type=str, default="ViT-B/32",
                        help="CLIP模型名称")
    
    args = parser.parse_args()
    
    run_full_benchmark(args.test_images, args.model)
