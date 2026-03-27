#!/usr/bin/env python3
"""
模型推理性能测试脚本
测试API接口的响应时间和吞吐量
"""

import time
import requests
import json
import concurrent.futures
import argparse
from PIL import Image
import io

class PerformanceTester:
    def __init__(self, api_url, test_image_path, concurrent_users=10, total_requests=100):
        """初始化性能测试器
        
        Args:
            api_url: API接口地址
            test_image_path: 测试图片路径
            concurrent_users: 并发用户数
            total_requests: 总请求数
        """
        self.api_url = api_url
        self.test_image_path = test_image_path
        self.concurrent_users = concurrent_users
        self.total_requests = total_requests
        self.results = []
    
    def load_test_image(self):
        """加载测试图片"""
        try:
            with open(self.test_image_path, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"加载图片失败: {e}")
            return None
    
    def test_request(self, image_data, request_id):
        """发送单个测试请求"""
        start_time = time.time()
        try:
            files = {'file': ('test_image.jpg', image_data, 'image/jpeg')}
            response = requests.post(self.api_url, files=files, timeout=30)
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                self.results.append({
                    'request_id': request_id,
                    'status': 'success',
                    'response_time': response_time,
                    'status_code': response.status_code,
                    'predictions': result.get('predictions', [])
                })
            else:
                self.results.append({
                    'request_id': request_id,
                    'status': 'error',
                    'response_time': response_time,
                    'status_code': response.status_code,
                    'error': response.text
                })
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            self.results.append({
                'request_id': request_id,
                'status': 'exception',
                'response_time': response_time,
                'error': str(e)
            })
    
    def run_test(self):
        """运行性能测试"""
        print(f"开始性能测试")
        print(f"测试配置: 并发用户数={self.concurrent_users}, 总请求数={self.total_requests}")
        print(f"API地址: {self.api_url}")
        
        # 加载测试图片
        image_data = self.load_test_image()
        if not image_data:
            print("测试图片加载失败，退出测试")
            return
        
        # 开始测试
        start_time = time.time()
        
        # 使用线程池进行并发测试
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrent_users) as executor:
            futures = []
            for i in range(self.total_requests):
                future = executor.submit(self.test_request, image_data, i+1)
                futures.append(future)
            
            # 等待所有请求完成
            concurrent.futures.wait(futures)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 分析结果
        self.analyze_results(total_time)
    
    def analyze_results(self, total_time):
        """分析测试结果"""
        if not self.results:
            print("没有测试结果")
            return
        
        # 统计成功和失败的请求
        success_count = sum(1 for r in self.results if r['status'] == 'success')
        error_count = sum(1 for r in self.results if r['status'] == 'error')
        exception_count = sum(1 for r in self.results if r['status'] == 'exception')
        
        # 计算响应时间
        response_times = [r['response_time'] for r in self.results if r['status'] == 'success']
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = 0
            min_response_time = 0
            max_response_time = 0
        
        # 计算吞吐量
        throughput = len(self.results) / total_time
        success_throughput = success_count / total_time
        
        # 打印结果
        print("\n===== 性能测试结果 =====")
        print(f"总请求数: {len(self.results)}")
        print(f"成功请求: {success_count} ({success_count/len(self.results)*100:.1f}%)")
        print(f"失败请求: {error_count} ({error_count/len(self.results)*100:.1f}%)")
        print(f"异常请求: {exception_count} ({exception_count/len(self.results)*100:.1f}%)")
        print(f"总测试时间: {total_time:.2f} 秒")
        print(f"平均响应时间: {avg_response_time:.3f} 秒")
        print(f"最小响应时间: {min_response_time:.3f} 秒")
        print(f"最大响应时间: {max_response_time:.3f} 秒")
        print(f"总吞吐量: {throughput:.2f} 请求/秒")
        print(f"成功吞吐量: {success_throughput:.2f} 请求/秒")
        
        # 保存详细结果
        with open('performance_test_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'test_config': {
                    'api_url': self.api_url,
                    'concurrent_users': self.concurrent_users,
                    'total_requests': self.total_requests
                },
                'summary': {
                    'total_requests': len(self.results),
                    'success_count': success_count,
                    'error_count': error_count,
                    'exception_count': exception_count,
                    'total_time': total_time,
                    'avg_response_time': avg_response_time,
                    'min_response_time': min_response_time,
                    'max_response_time': max_response_time,
                    'throughput': throughput,
                    'success_throughput': success_throughput
                },
                'details': self.results
            }, f, ensure_ascii=False, indent=2)
        
        print("\n详细测试结果已保存到 performance_test_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型推理性能测试脚本')
    parser.add_argument('--api-url', default='http://localhost:8000/api/classify', help='API接口地址')
    parser.add_argument('--image', default='test_images/sample.jpg', help='测试图片路径')
    parser.add_argument('--concurrent', type=int, default=10, help='并发用户数')
    parser.add_argument('--requests', type=int, default=100, help='总请求数')
    
    args = parser.parse_args()
    
    tester = PerformanceTester(
        api_url=args.api_url,
        test_image_path=args.image,
        concurrent_users=args.concurrent,
        total_requests=args.requests
    )
    
    tester.run_test()
