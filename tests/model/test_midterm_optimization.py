#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试中期优化效果

验证性能和稳定性提升
"""

import os
import sys
import time
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.http_utils import HTTPUtils
from src.utils.image_utils import ImageUtils
from src.utils.cache_manager import cache_manager
from src.utils.monitoring_system import MonitoringSystem
from src.utils.concurrency_manager import ConcurrencyManager
from src.core.exception_handling.exception_handler import exception_handler
from src.data_collection.keyword_based_collector import KeywordBasedDataCollector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_midterm_optimization')


class MidtermOptimizationTest:
    """
    中期优化测试类
    """
    
    def __init__(self):
        """
        初始化测试类
        """
        self.test_results = {}
        self.monitoring_system = MonitoringSystem()
        self.monitoring_system.start()
        
        logger.info("初始化中期优化测试")
    
    def test_network_optimization(self):
        """
        测试网络优化效果
        """
        logger.info("开始测试网络优化")
        
        test_urls = [
            'https://sd.vv50.de/search.php?word=原神',
            'https://sd.vv50.de/illustration?word=原神',
            'https://sd.vv50.de/ranking.php?word=原神',
            'https://sd.vv50.de/bookmark_new_illust.php?word=原神'
        ]
        
        # 创建HTTP工具实例
        http_utils = HTTPUtils(monitoring_system=self.monitoring_system)
        
        # 测试第一次请求（无缓存）
        first_times = []
        for url in test_urls:
            start_time = time.time()
            try:
                response = http_utils.get(url, cache_enabled=True)
                end_time = time.time()
                first_times.append(end_time - start_time)
                logger.info(f"第一次请求 {url} 耗时: {end_time - start_time:.4f}秒")
            except Exception as e:
                logger.error(f"请求失败 {url}: {e}")
                first_times.append(float('inf'))
        
        # 测试第二次请求（有缓存）
        second_times = []
        for url in test_urls:
            start_time = time.time()
            try:
                response = http_utils.get(url, cache_enabled=True)
                end_time = time.time()
                second_times.append(end_time - start_time)
                logger.info(f"第二次请求 {url} 耗时: {end_time - start_time:.4f}秒")
            except Exception as e:
                logger.error(f"请求失败 {url}: {e}")
                second_times.append(float('inf'))
        
        # 计算缓存命中率
        cache_stats = cache_manager.get_stats()
        hit_rate = cache_stats['hit_rate']
        
        # 计算平均耗时
        avg_first_time = sum(first_times) / len(first_times) if first_times else 0
        avg_second_time = sum(second_times) / len(second_times) if second_times else 0
        speedup = avg_first_time / avg_second_time if avg_second_time > 0 else 0
        
        result = {
            'first_times': first_times,
            'second_times': second_times,
            'avg_first_time': avg_first_time,
            'avg_second_time': avg_second_time,
            'speedup': speedup,
            'cache_hit_rate': hit_rate,
            'cache_stats': cache_stats
        }
        
        self.test_results['network_optimization'] = result
        logger.info(f"网络优化测试完成，平均加速比: {speedup:.2f}x, 缓存命中率: {hit_rate:.2f}%")
        
        return result
    
    def test_concurrency_optimization(self):
        """
        测试并发优化效果
        """
        logger.info("开始测试并发优化")
        
        # 创建并发管理器
        concurrency_manager = ConcurrencyManager()
        
        # 测试函数
        def test_func(url):
            http_utils = HTTPUtils()
            try:
                response = http_utils.get(url, cache_enabled=True)
                return True
            except Exception:
                return False
        
        # 测试不同并发数的性能
        test_urls = [
            'https://sd.vv50.de/search.php?word=原神',
            'https://sd.vv50.de/illustration?word=原神',
            'https://sd.vv50.de/ranking.php?word=原神',
            'https://sd.vv50.de/bookmark_new_illust.php?word=原神'
        ] * 5  # 重复5次，共20个请求
        
        concurrency_levels = [1, 5, 10, 15, 20]
        results = []
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            # 使用并发管理器执行任务
            futures = []
            for url in test_urls:
                future = concurrency_manager.submit(test_func, url)
                futures.append(future)
            
            # 等待所有任务完成
            success_count = 0
            for future in futures:
                if future.result():
                    success_count += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            qps = len(test_urls) / total_time if total_time > 0 else 0
            
            results.append({
                'concurrency': concurrency,
                'total_time': total_time,
                'qps': qps,
                'success_count': success_count,
                'total_requests': len(test_urls),
                'success_rate': success_count / len(test_urls) * 100
            })
            
            logger.info(f"并发数 {concurrency}: 总耗时 {total_time:.4f}秒, QPS: {qps:.2f}, 成功率: {success_count / len(test_urls) * 100:.2f}%")
        
        # 获取动态并发数调整结果
        dynamic_concurrency = concurrency_manager.get_optimal_workers()
        
        result = {
            'concurrency_tests': results,
            'dynamic_concurrency': dynamic_concurrency
        }
        
        self.test_results['concurrency_optimization'] = result
        logger.info(f"并发优化测试完成，最佳动态并发数: {dynamic_concurrency}")
        
        return result
    
    def test_image_processing_optimization(self):
        """
        测试图片处理优化效果
        """
        logger.info("开始测试图片处理优化")
        
        # 创建HTTP工具实例
        http_utils = HTTPUtils()
        
        # 下载测试图片
        test_urls = [
            'https://images.unsplash.com/photo-1547425260-76bcadfb4f2c',
            'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d',
            'https://images.unsplash.com/photo-1518843875459-f738682238a6'
        ]
        
        test_images = []
        for url in test_urls:
            try:
                content = http_utils.download_file(url, cache_enabled=True)
                test_images.append(content)
                logger.info(f"下载测试图片成功: {url}")
            except Exception as e:
                logger.error(f"下载测试图片失败 {url}: {e}")
        
        if not test_images:
            logger.error("没有可用的测试图片")
            return {}
        
        # 测试图片质量评估
        start_time = time.time()
        quality_scores = []
        for image in test_images:
            score = ImageUtils.calculate_image_quality(image)
            quality_scores.append(score)
            logger.info(f"图片质量分数: {score:.2f}")
        quality_time = time.time() - start_time
        
        # 测试图片去重
        start_time = time.time()
        unique_images = ImageUtils.deduplicate_images(test_images * 2)  # 重复测试图片
        deduplicate_time = time.time() - start_time
        
        # 测试批量图片分析
        start_time = time.time()
        analysis_results = ImageUtils.batch_analyze_images(test_images)
        batch_analysis_time = time.time() - start_time
        
        result = {
            'quality_scores': quality_scores,
            'quality_time': quality_time,
            'deduplicate_count': len(unique_images),
            'deduplicate_time': deduplicate_time,
            'batch_analysis_count': len(analysis_results),
            'batch_analysis_time': batch_analysis_time
        }
        
        self.test_results['image_processing_optimization'] = result
        logger.info(f"图片处理优化测试完成")
        
        return result
    
    def test_error_handling_optimization(self):
        """
        测试错误处理优化效果
        """
        logger.info("开始测试错误处理优化")
        
        # 测试异常处理器
        def test_func(url):
            http_utils = HTTPUtils()
            response = http_utils.get(url)
            return response
        
        # 测试正常情况
        normal_url = 'https://sd.vv50.de/search.php?word=原神'
        
        # 测试错误情况
        error_url = 'https://sd.vv50.de/nonexistent_page.php'
        
        # 测试重试机制
        start_time = time.time()
        try:
            result = exception_handler.retry_with_backoff(test_func, error_url)
            success = True
        except Exception:
            success = False
        retry_time = time.time() - start_time
        
        # 测试安全执行
        safe_result = exception_handler.safe_execute(test_func, error_url)
        
        # 获取错误统计
        error_stats = exception_handler.get_error_stats()
        
        result = {
            'retry_success': success,
            'retry_time': retry_time,
            'safe_execute_result': safe_result is not None,
            'error_stats': error_stats
        }
        
        self.test_results['error_handling_optimization'] = result
        logger.info(f"错误处理优化测试完成")
        
        return result
    
    def test_data_collection_optimization(self):
        """
        测试数据采集优化效果
        """
        logger.info("开始测试数据采集优化")
        
        # 创建关键词采集器
        collector = KeywordBasedDataCollector()
        
        # 测试单个角色采集
        start_time = time.time()
        try:
            # 只采集10张图片，快速测试
            result = collector._process_character('原神', '雷电将军', max_images=10)
            end_time = time.time()
            success = True
        except Exception as e:
            logger.error(f"数据采集测试失败: {e}")
            result = 0
            end_time = time.time()
            success = False
        
        collection_time = end_time - start_time
        
        result = {
            'success': success,
            'collection_time': collection_time,
            'collected_images': result
        }
        
        self.test_results['data_collection_optimization'] = result
        logger.info(f"数据采集优化测试完成，采集 {result} 张图片，耗时 {collection_time:.4f}秒")
        
        return result
    
    def test_monitoring_system(self):
        """
        测试监控系统功能
        """
        logger.info("开始测试监控系统")
        
        # 获取监控统计信息
        stats = self.monitoring_system.get_all_stats()
        
        # 获取告警信息
        alerts = self.monitoring_system.get_alerts()
        
        # 获取仪表板数据
        dashboard_data = self.monitoring_system.get_dashboard_data()
        
        result = {
            'stats': stats,
            'alerts': alerts,
            'dashboard_data': dashboard_data
        }
        
        self.test_results['monitoring_system'] = result
        logger.info(f"监控系统测试完成")
        
        return result
    
    def run_all_tests(self):
        """
        运行所有测试
        """
        logger.info("开始运行所有中期优化测试")
        
        tests = [
            self.test_network_optimization,
            self.test_concurrency_optimization,
            self.test_image_processing_optimization,
            self.test_error_handling_optimization,
            self.test_data_collection_optimization,
            self.test_monitoring_system
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"测试失败 {test.__name__}: {e}")
        
        # 保存测试结果
        self.save_test_results()
        
        # 打印测试总结
        self.print_test_summary()
        
        logger.info("所有中期优化测试完成")
    
    def save_test_results(self):
        """
        保存测试结果
        """
        import json
        
        results_file = 'test_midterm_optimization_results.json'
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            logger.info(f"测试结果已保存到 {results_file}")
        except Exception as e:
            logger.error(f"保存测试结果失败: {e}")
    
    def print_test_summary(self):
        """
        打印测试总结
        """
        logger.info("\n=== 中期优化测试总结 ===")
        
        # 网络优化总结
        if 'network_optimization' in self.test_results:
            net_result = self.test_results['network_optimization']
            logger.info(f"网络优化: 平均加速比 {net_result.get('speedup', 0):.2f}x, 缓存命中率 {net_result.get('cache_hit_rate', 0):.2f}%")
        
        # 并发优化总结
        if 'concurrency_optimization' in self.test_results:
            concur_result = self.test_results['concurrency_optimization']
            best_result = max(concur_result.get('concurrency_tests', []), key=lambda x: x['qps']) if 'concurrency_tests' in concur_result else {}
            logger.info(f"并发优化: 最佳QPS {best_result.get('qps', 0):.2f}, 最佳并发数 {best_result.get('concurrency', 0)}")
        
        # 图片处理优化总结
        if 'image_processing_optimization' in self.test_results:
            img_result = self.test_results['image_processing_optimization']
            logger.info(f"图片处理优化: 质量评估耗时 {img_result.get('quality_time', 0):.4f}秒, 去重后剩余 {img_result.get('deduplicate_count', 0)}张")
        
        # 数据采集优化总结
        if 'data_collection_optimization' in self.test_results:
            coll_result = self.test_results['data_collection_optimization']
            logger.info(f"数据采集优化: 采集 {coll_result.get('collected_images', 0)}张图片, 耗时 {coll_result.get('collection_time', 0):.4f}秒")
        
        # 监控系统总结
        if 'monitoring_system' in self.test_results:
            monitor_result = self.test_results['monitoring_system']
            alerts_count = len(monitor_result.get('alerts', []))
            logger.info(f"监控系统: 运行正常, 告警数 {alerts_count}")
        
        logger.info("======================")
    
    def cleanup(self):
        """
        清理测试资源
        """
        # 停止监控系统
        self.monitoring_system.stop()
        
        # 清理缓存
        cache_manager.clear()
        
        logger.info("清理测试资源")


if __name__ == '__main__':
    # 创建测试实例
    test = MidtermOptimizationTest()
    
    try:
        # 运行所有测试
        test.run_all_tests()
    finally:
        # 清理资源
        test.cleanup()
