#!/usr/bin/env python3
"""
性能测试脚本
测试系统的性能稳定性和效果
"""

import os
import sys
import time
import json
import requests
import psutil
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('performance_test')

class PerformanceTester:
    """
    性能测试器
    """
    
    def __init__(self):
        """初始化性能测试器"""
        logger.info("初始化性能测试器")
        self.api_url = "http://localhost:8000/api/classify"
        self.test_images = self._get_test_images()
        logger.info(f"收集到 {len(self.test_images)} 张测试图片")
    
    def _get_test_images(self):
        """
        获取测试图片
        
        Returns:
            测试图片路径列表
        """
        test_images = []
        data_dir = 'data/train'
        
        if not os.path.exists(data_dir):
            logger.warning(f"测试数据目录不存在: {data_dir}")
            return test_images
        
        role_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        for role_dir in role_dirs:
            role_path = os.path.join(data_dir, role_dir)
            image_files = [f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            
            # 每个角色取前3张图片
            for img_file in image_files[:3]:
                img_path = os.path.join(role_path, img_file)
                test_images.append((img_path, role_dir))
        
        return test_images
    
    def test_response_time(self):
        """
        测试API响应时间
        
        Returns:
            响应时间列表
        """
        logger.info("开始测试API响应时间")
        response_times = []
        
        for img_path, true_role in self.test_images[:5]:  # 只测试前5张图片
            start_time = time.time()
            
            try:
                with open(img_path, 'rb') as f:
                    files = {'file': (os.path.basename(img_path), f, 'image/jpeg')}
                    response = requests.post(self.api_url, files=files, timeout=60)
                
                if response.status_code == 200:
                    end_time = time.time()
                    response_time = end_time - start_time
                    response_times.append(response_time)
                    logger.info(f"图片 {os.path.basename(img_path)} 响应时间: {response_time:.2f} 秒")
                else:
                    logger.error(f"API请求失败，状态码: {response.status_code}")
            except Exception as e:
                logger.error(f"测试响应时间时出错: {e}")
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            logger.info(f"响应时间统计: 平均 {avg_time:.2f} 秒, 最大 {max_time:.2f} 秒, 最小 {min_time:.2f} 秒")
        
        return response_times
    
    def test_resource_usage(self):
        """
        测试系统资源使用情况
        
        Returns:
            资源使用情况字典
        """
        logger.info("开始测试系统资源使用情况")
        
        # 获取当前进程
        process = psutil.Process()
        
        # 测试前的资源使用
        before_cpu = process.cpu_percent(interval=1)
        before_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行测试
        for img_path, true_role in self.test_images[:3]:  # 只测试前3张图片
            try:
                with open(img_path, 'rb') as f:
                    files = {'file': (os.path.basename(img_path), f, 'image/jpeg')}
                    response = requests.post(self.api_url, files=files, timeout=60)
                
                if response.status_code == 200:
                    logger.info(f"处理图片 {os.path.basename(img_path)} 成功")
                else:
                    logger.error(f"API请求失败，状态码: {response.status_code}")
            except Exception as e:
                logger.error(f"测试资源使用时出错: {e}")
        
        # 测试后的资源使用
        after_cpu = process.cpu_percent(interval=1)
        after_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        resource_usage = {
            'cpu_before': before_cpu,
            'cpu_after': after_cpu,
            'memory_before': before_memory,
            'memory_after': after_memory,
            'memory_increase': after_memory - before_memory
        }
        
        logger.info(f"资源使用情况: CPU 使用率从 {before_cpu:.2f}% 变为 {after_cpu:.2f}%")
        logger.info(f"内存使用从 {before_memory:.2f} MB 变为 {after_memory:.2f} MB，增加了 {resource_usage['memory_increase']:.2f} MB")
        
        return resource_usage
    
    def test_classification_accuracy(self):
        """
        测试分类准确率
        
        Returns:
            准确率
        """
        logger.info("开始测试分类准确率")
        
        correct = 0
        total = 0
        results = []
        
        for img_path, true_role in self.test_images[:10]:  # 只测试前10张图片
            try:
                with open(img_path, 'rb') as f:
                    files = {'file': (os.path.basename(img_path), f, 'image/jpeg')}
                    response = requests.post(self.api_url, files=files, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    predicted_role = data.get('role', 'unknown')
                    ai_predicted_role = data.get('ai_predicted_role', 'unknown')
                    similarity = data.get('similarity', 0.0)
                    
                    # 检查预测结果是否正确
                    is_correct = true_role in predicted_role or predicted_role in true_role
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    results.append({
                        'image_path': img_path,
                        'true_role': true_role,
                        'predicted_role': predicted_role,
                        'ai_predicted_role': ai_predicted_role,
                        'similarity': similarity,
                        'is_correct': is_correct
                    })
                    
                    logger.info(f"图片 {os.path.basename(img_path)}: 真实角色={true_role}, 预测角色={predicted_role}, AI预测={ai_predicted_role}, 相似度={similarity:.4f}, {'正确' if is_correct else '错误'}")
                else:
                    logger.error(f"API请求失败，状态码: {response.status_code}")
            except Exception as e:
                logger.error(f"测试分类准确率时出错: {e}")
        
        if total > 0:
            accuracy = correct / total
            logger.info(f"分类准确率: {accuracy * 100:.2f}% ({correct}/{total})")
        else:
            accuracy = 0.0
            logger.error("没有测试数据")
        
        return accuracy, results
    
    def test_tagging_quality(self):
        """
        测试标签生成质量
        
        Returns:
            标签质量评估
        """
        logger.info("开始测试标签生成质量")
        
        tag_stats = {
            'total_tags': 0,
            'total_images': 0,
            'avg_tags_per_image': 0
        }
        
        for img_path, true_role in self.test_images[:5]:  # 只测试前5张图片
            try:
                with open(img_path, 'rb') as f:
                    files = {'file': (os.path.basename(img_path), f, 'image/jpeg')}
                    response = requests.post(self.api_url, files=files, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    attributes = data.get('attributes', [])
                    tag_count = len(attributes)
                    
                    tag_stats['total_tags'] += tag_count
                    tag_stats['total_images'] += 1
                    
                    logger.info(f"图片 {os.path.basename(img_path)}: 生成 {tag_count} 个标签")
                    logger.info(f"前10个标签: {attributes[:10]}")
                else:
                    logger.error(f"API请求失败，状态码: {response.status_code}")
            except Exception as e:
                logger.error(f"测试标签生成质量时出错: {e}")
        
        if tag_stats['total_images'] > 0:
            tag_stats['avg_tags_per_image'] = tag_stats['total_tags'] / tag_stats['total_images']
            logger.info(f"标签生成统计: 平均每张图片生成 {tag_stats['avg_tags_per_image']:.2f} 个标签")
        
        return tag_stats
    
    def run_all_tests(self):
        """
        运行所有测试
        
        Returns:
            测试结果字典
        """
        logger.info("开始运行所有性能测试")
        
        results = {
            'response_times': self.test_response_time(),
            'resource_usage': self.test_resource_usage(),
            'classification_accuracy': self.test_classification_accuracy(),
            'tagging_quality': self.test_tagging_quality()
        }
        
        # 保存测试结果
        output_dir = 'test_results'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'performance_test_{int(time.time())}.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试结果已保存到: {output_path}")
        
        return results

def main():
    """主函数"""
    tester = PerformanceTester()
    results = tester.run_all_tests()
    
    # 打印测试总结
    logger.info("\n=== 性能测试总结 ===")
    
    # 响应时间
    if results['response_times']:
        avg_time = sum(results['response_times']) / len(results['response_times'])
        logger.info(f"响应时间: 平均 {avg_time:.2f} 秒")
    
    # 资源使用
    resource_usage = results['resource_usage']
    logger.info(f"CPU 使用率: 从 {resource_usage['cpu_before']:.2f}% 变为 {resource_usage['cpu_after']:.2f}%")
    logger.info(f"内存使用: 从 {resource_usage['memory_before']:.2f} MB 变为 {resource_usage['memory_after']:.2f} MB")
    
    # 分类准确率
    accuracy, _ = results['classification_accuracy']
    logger.info(f"分类准确率: {accuracy * 100:.2f}%")
    
    # 标签生成质量
    tag_stats = results['tagging_quality']
    if tag_stats['total_images'] > 0:
        logger.info(f"标签生成: 平均每张图片 {tag_stats['avg_tags_per_image']:.2f} 个标签")

if __name__ == '__main__':
    main()
