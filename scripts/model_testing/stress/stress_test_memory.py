#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存压力测试脚本

测试系统在高负载下的内存使用情况和稳定性
"""

import os
import sys
import time
import requests
import concurrent.futures
import random
from PIL import Image
import io
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("stress_test_memory")

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MemoryStressTester:
    """
    内存压力测试器
    """
    
    def __init__(self, api_url="http://localhost:8000/api/classify", test_duration=300, concurrent_users=10):
        """
        初始化内存压力测试器
        
        Args:
            api_url: API接口地址
            test_duration: 测试持续时间（秒）
            concurrent_users: 并发用户数
        """
        self.api_url = api_url
        self.test_duration = test_duration
        self.concurrent_users = concurrent_users
        self.results = []
        self.error_count = 0
        self.success_count = 0
        self.start_time = None
        
        # 准备测试图像
        self.test_images = self._prepare_test_images()
    
    def _prepare_test_images(self):
        """
        准备测试图像
        
        Returns:
            测试图像列表
        """
        test_images = []
        
        # 生成不同大小的测试图像
        for size in [(224, 224), (512, 512), (1024, 1024)]:
            for i in range(3):
                # 创建随机图像
                img = Image.new('RGB', size, color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                buffer.seek(0)
                test_images.append((f"test_{size[0]}x{size[1]}_{i}.jpg", buffer.getvalue()))
        
        # 尝试加载真实图像（如果存在）
        test_data_path = os.path.join(project_root, "data", "train")
        if os.path.exists(test_data_path):
            for role_dir in os.listdir(test_data_path):
                role_path = os.path.join(test_data_path, role_dir)
                if os.path.isdir(role_path):
                    for img_file in os.listdir(role_path)[:2]:  # 每个角色取2张图片
                        if img_file.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(role_path, img_file)
                            try:
                                with open(img_path, 'rb') as f:
                                    img_data = f.read()
                                test_images.append((img_file, img_data))
                            except Exception as e:
                                logger.warning(f"加载测试图像 {img_path} 失败: {e}")
        
        logger.info(f"准备了 {len(test_images)} 张测试图像")
        return test_images
    
    def _send_request(self, user_id):
        """
        发送测试请求
        
        Args:
            user_id: 用户ID
        """
        try:
            # 随机选择一张测试图像
            img_name, img_data = random.choice(self.test_images)
            
            # 构建请求
            files = {'file': (img_name, img_data, 'image/jpeg')}
            data = {'model_name': 'default'}
            
            start_time = time.time()
            response = requests.post(self.api_url, files=files, data=data, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                self.success_count += 1
                self.results.append({
                    'user_id': user_id,
                    'status': 'success',
                    'response_time': response_time,
                    'role': result.get('role', 'unknown'),
                    'similarity': result.get('similarity', 0.0)
                })
                if user_id == 0 and random.random() < 0.1:  # 每10次请求打印一次
                    logger.info(f"请求成功: 角色={result.get('role')}, 相似度={result.get('similarity'):.4f}, 响应时间={response_time:.2f}s")
            else:
                self.error_count += 1
                self.results.append({
                    'user_id': user_id,
                    'status': 'error',
                    'response_time': response_time,
                    'error': f"状态码: {response.status_code}"
                })
                if user_id == 0 and random.random() < 0.1:
                    logger.error(f"请求失败: 状态码={response.status_code}")
                    
        except Exception as e:
            self.error_count += 1
            self.results.append({
                'user_id': user_id,
                'status': 'exception',
                'response_time': time.time() - start_time if 'start_time' in locals() else 0,
                'error': str(e)
            })
            if user_id == 0 and random.random() < 0.1:
                logger.error(f"请求异常: {e}")
    
    def run_test(self):
        """
        运行压力测试
        """
        logger.info(f"开始内存压力测试")
        logger.info(f"测试配置: 并发用户数={self.concurrent_users}, 测试时长={self.test_duration}秒")
        
        self.start_time = time.time()
        end_time = self.start_time + self.test_duration
        
        # 使用线程池进行并发测试
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrent_users) as executor:
            while time.time() < end_time:
                # 提交并发请求
                futures = []
                for user_id in range(self.concurrent_users):
                    future = executor.submit(self._send_request, user_id)
                    futures.append(future)
                
                # 等待所有请求完成
                concurrent.futures.wait(futures)
                
                # 短暂休息，避免过于密集的请求
                time.sleep(0.1)
        
        self._print_results()
    
    def _print_results(self):
        """
        打印测试结果
        """
        total_requests = self.success_count + self.error_count
        test_duration = time.time() - self.start_time
        
        logger.info(f"\n===== 内存压力测试结果 =====")
        logger.info(f"测试时长: {test_duration:.2f}秒")
        logger.info(f"总请求数: {total_requests}")
        logger.info(f"成功请求: {self.success_count} ({self.success_count/total_requests*100:.2f}%)")
        logger.info(f"失败请求: {self.error_count} ({self.error_count/total_requests*100:.2f}%)")
        
        if self.results:
            response_times = [r['response_time'] for r in self.results if r['status'] == 'success']
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                logger.info(f"平均响应时间: {avg_response_time:.2f}秒")
                logger.info(f"最大响应时间: {max_response_time:.2f}秒")
                logger.info(f"最小响应时间: {min_response_time:.2f}秒")
        
        logger.info("===== 测试完成 =====")


if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description='内存压力测试脚本')
    parser.add_argument('--duration', type=int, default=300, help='测试持续时间（秒）')
    parser.add_argument('--users', type=int, default=10, help='并发用户数')
    parser.add_argument('--url', type=str, default='http://localhost:8000/api/classify', help='API接口地址')
    
    args = parser.parse_args()
    
    # 运行测试
    tester = MemoryStressTester(
        api_url=args.url,
        test_duration=args.duration,
        concurrent_users=args.users
    )
    
    try:
        tester.run_test()
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    finally:
        logger.info("内存压力测试结束")
