#!/usr/bin/env python3
"""
压力测试脚本
模拟多个并发请求，测试系统的稳定性和性能
"""

import os
import sys
import time
import requests
import concurrent.futures
import psutil
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='stress_test.log'
)
logger = logging.getLogger("stress_test")

def get_project_root():
    """
    获取项目根目录
    """
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root

def monitor_resources():
    """
    监控系统资源使用情况
    """
    while True:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        logger.info(f"资源使用: CPU={cpu_percent:.2f}%, 内存={memory.percent:.2f}%, 磁盘={disk.percent:.2f}%")
        time.sleep(1)

def test_classification(test_image_path, model_name="default"):
    """
    测试单个分类请求
    """
    url = "http://localhost:8000/api/classify"
    start_time = time.time()
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': (os.path.basename(test_image_path), f, 'image/jpeg')}
            data = {'model_name': model_name}
            response = requests.post(url, files=files, data=data, timeout=60)  # 增加超时时间
        
        response_time = time.time() - start_time
        status_code = response.status_code
        
        logger.info(f"请求状态码: {status_code}, 响应时间: {response_time:.2f}秒")
        logger.info(f"响应内容: {response.text[:500]}...")  # 记录响应内容的前500个字符
        
        if status_code == 200:
            try:
                result = response.json()
                role = result.get('role', 'unknown')
                similarity = result.get('similarity', 0.0)
                logger.info(f"请求成功: 角色={role}, 相似度={similarity:.4f}, 响应时间={response_time:.2f}秒")
                return True, response_time
            except Exception as json_error:
                logger.error(f"解析JSON失败: {json_error}")
                return False, response_time
        else:
            logger.error(f"请求失败: 状态码={status_code}, 响应时间={response_time:.2f}秒")
            return False, response_time
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"请求异常: {e}, 响应时间={response_time:.2f}秒")
        return False, response_time

def run_stress_test(test_image_path, num_requests=20, concurrent_workers=5):
    """
    运行压力测试
    """
    logger.info(f"开始压力测试: {num_requests}个请求, {concurrent_workers}个并发")
    
    # 启动资源监控
    import threading
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    # 等待1秒，确保监控线程启动
    time.sleep(1)
    
    # 执行并发请求
    success_count = 0
    total_response_time = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        futures = []
        for i in range(num_requests):
            future = executor.submit(test_classification, test_image_path)
            futures.append(future)
            # 每提交5个请求，暂停0.5秒，避免瞬间发送过多请求
            if (i + 1) % 5 == 0:
                time.sleep(0.5)
        
        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            success, response_time = future.result()
            if success:
                success_count += 1
            total_response_time += response_time
    
    # 计算统计信息
    success_rate = (success_count / num_requests) * 100
    avg_response_time = total_response_time / num_requests if num_requests > 0 else 0
    
    logger.info(f"压力测试完成: 成功={success_count}/{num_requests} ({success_rate:.2f}%), 平均响应时间={avg_response_time:.2f}秒")
    print(f"压力测试完成: 成功={success_count}/{num_requests} ({success_rate:.2f}%), 平均响应时间={avg_response_time:.2f}秒")

if __name__ == "__main__":
    project_root = get_project_root()
    
    # 查找测试图像
    test_image_path = None
    test_dir = os.path.join(project_root, "data", "test")
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                test_image_path = os.path.join(test_dir, file)
                break
    
    # 如果测试目录不存在，从训练数据中找一个图像
    if not test_image_path:
        train_dir = os.path.join(project_root, "data", "train")
        if os.path.exists(train_dir):
            for role_dir in os.listdir(train_dir):
                role_path = os.path.join(train_dir, role_dir)
                if os.path.isdir(role_path):
                    for file in os.listdir(role_path):
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            test_image_path = os.path.join(role_path, file)
                            break
                    if test_image_path:
                        break
    
    if not test_image_path:
        logger.error("找不到测试图像")
        print("找不到测试图像")
        sys.exit(1)
    
    logger.info(f"使用测试图像: {test_image_path}")
    print(f"使用测试图像: {test_image_path}")
    
    # 运行压力测试
    run_stress_test(test_image_path, num_requests=50, concurrent_workers=10)
