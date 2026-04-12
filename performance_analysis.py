#!/usr/bin/env python3
"""
性能分析脚本
用于分析从前端发送分类请求到后端返回检测结果期间的内存、CPU和GPU占用
"""

import os
import sys
import time
import psutil
import requests
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import random

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 配置
API_URL = "http://localhost:8000/api/classify"
DOWNLOADED_IMAGES_DIR = "./downloaded_images"  # 下载的图片目录
NUM_REQUESTS = 10  # 测试请求次数
SAMPLING_INTERVAL = 0.1  # 采样间隔（秒）
NUM_TESTS = 3  # 测试次数，取平均值

# 性能数据收集
performance_data = {
    "cpu_percent": [],
    "memory_percent": [],
    "memory_used": [],
    "gpu_memory_used": [],
    "gpu_utilization": [],
    "timestamps": []
}

# 尝试导入GPU监控库
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: pynvml not installed, GPU monitoring disabled")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"Warning: GPU initialization failed: {e}")


def get_gpu_metrics():
    """获取GPU指标"""
    if not GPU_AVAILABLE:
        return 0, 0
    
    try:
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        memory_used = memory_info.used / 1024 / 1024  # MB
        utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        return memory_used, utilization
    except Exception as e:
        print(f"Error getting GPU metrics: {e}")
        return 0, 0


def collect_performance_data():
    """收集性能数据"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_used = memory.used / 1024 / 1024  # MB
    gpu_memory, gpu_util = get_gpu_metrics()
    
    performance_data["cpu_percent"].append(cpu_percent)
    performance_data["memory_percent"].append(memory_percent)
    performance_data["memory_used"].append(memory_used)
    performance_data["gpu_memory_used"].append(gpu_memory)
    performance_data["gpu_utilization"].append(gpu_util)
    performance_data["timestamps"].append(time.time())


def get_downloaded_images():
    """获取下载的图片列表"""
    image_files = []
    for root, dirs, files in os.walk(DOWNLOADED_IMAGES_DIR):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    return image_files

def send_classification_request(image_path):
    """发送分类请求"""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data = {'model_name': 'default'}
            response = requests.post(API_URL, files=files, data=data, timeout=60)
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending request: {e}")
        return False


def run_performance_test():
    """运行性能测试"""
    # 获取下载的图片列表
    image_files = get_downloaded_images()
    if not image_files:
        print(f"Error: No images found in {DOWNLOADED_IMAGES_DIR}")
        return []
    
    print(f"找到 {len(image_files)} 张下载的图片")
    
    all_request_times = []
    
    for test_num in range(NUM_TESTS):
        print(f"\n=== 测试 {test_num+1}/{NUM_TESTS} ===")
        print(f"开始性能测试，共发送 {NUM_REQUESTS} 个请求...")
        
        # 预热请求
        print("发送预热请求...")
        test_image = random.choice(image_files)
        send_classification_request(test_image)
        time.sleep(2)
        
        # 清空之前的数据
        for key in performance_data:
            performance_data[key] = []
        
        # 开始监控
        start_time = time.time()
        
        # 启动数据收集线程
        import threading
        stop_event = threading.Event()
        
        def monitor_thread():
            while not stop_event.is_set():
                collect_performance_data()
                time.sleep(SAMPLING_INTERVAL)
        
        monitor = threading.Thread(target=monitor_thread)
        monitor.start()
        
        # 发送请求
        request_times = []
        for i in range(NUM_REQUESTS):
            # 随机选择一张图片
            test_image = random.choice(image_files)
            print(f"发送请求 {i+1}/{NUM_REQUESTS}...")
            print(f"使用图片: {os.path.basename(test_image)}")
            req_start = time.time()
            success = send_classification_request(test_image)
            req_end = time.time()
            request_times.append(req_end - req_start)
            print(f"请求 {i+1} {'成功' if success else '失败'}，耗时: {req_end - req_start:.2f}秒")
            if i < NUM_REQUESTS - 1:
                time.sleep(0.5)  # 避免请求过于密集
        
        # 停止监控
        time.sleep(2)  # 等待一些额外的数据点
        stop_event.set()
        monitor.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n测试 {test_num+1} 完成，总耗时: {total_time:.2f}秒")
        print(f"平均请求时间: {sum(request_times) / len(request_times):.2f}秒")
        
        all_request_times.extend(request_times)
        
        # 测试之间休息一下
        if test_num < NUM_TESTS - 1:
            print("\n休息 5 秒，准备下一次测试...")
            time.sleep(5)
    
    # 计算所有测试的平均值
    if all_request_times:
        avg_time = sum(all_request_times) / len(all_request_times)
        print(f"\n=== 所有测试完成 ===")
        print(f"总请求数: {len(all_request_times)}")
        print(f"平均请求时间: {avg_time:.2f}秒")
    
    return all_request_times


def generate_plots():
    """生成性能分析图表"""
    if not performance_data["timestamps"]:
        print("No performance data collected")
        return
    
    # 计算相对时间
    start_time = performance_data["timestamps"][0]
    relative_times = [t - start_time for t in performance_data["timestamps"]]
    
    # 创建图表
    plt.figure(figsize=(15, 12))
    
    # CPU使用率
    plt.subplot(3, 2, 1)
    plt.plot(relative_times, performance_data["cpu_percent"])
    plt.title('CPU使用率 (%)')
    plt.xlabel('时间 (秒)')
    plt.ylabel('使用率 (%)')
    plt.grid(True)
    
    # 内存使用率
    plt.subplot(3, 2, 2)
    plt.plot(relative_times, performance_data["memory_percent"])
    plt.title('内存使用率 (%)')
    plt.xlabel('时间 (秒)')
    plt.ylabel('使用率 (%)')
    plt.grid(True)
    
    # 内存使用量
    plt.subplot(3, 2, 3)
    plt.plot(relative_times, performance_data["memory_used"])
    plt.title('内存使用量 (MB)')
    plt.xlabel('时间 (秒)')
    plt.ylabel('使用量 (MB)')
    plt.grid(True)
    
    # GPU内存使用量
    if GPU_AVAILABLE:
        plt.subplot(3, 2, 4)
        plt.plot(relative_times, performance_data["gpu_memory_used"])
        plt.title('GPU内存使用量 (MB)')
        plt.xlabel('时间 (秒)')
        plt.ylabel('使用量 (MB)')
        plt.grid(True)
    
    # GPU使用率
    if GPU_AVAILABLE:
        plt.subplot(3, 2, 5)
        plt.plot(relative_times, performance_data["gpu_utilization"])
        plt.title('GPU使用率 (%)')
        plt.xlabel('时间 (秒)')
        plt.ylabel('使用率 (%)')
        plt.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'performance_analysis_{timestamp}.png'
    plt.savefig(plot_path)
    print(f"性能分析图表已保存到: {plot_path}")
    
    # 显示图表
    plt.show()


def analyze_bottlenecks():
    """分析性能瓶颈"""
    print("\n=== 性能瓶颈分析 ===")
    
    # 计算平均值
    avg_cpu = np.mean(performance_data["cpu_percent"])
    max_cpu = np.max(performance_data["cpu_percent"])
    avg_memory = np.mean(performance_data["memory_percent"])
    max_memory = np.max(performance_data["memory_percent"])
    
    print(f"CPU平均使用率: {avg_cpu:.2f}%")
    print(f"CPU最大使用率: {max_cpu:.2f}%")
    print(f"内存平均使用率: {avg_memory:.2f}%")
    print(f"内存最大使用率: {max_memory:.2f}%")
    
    if GPU_AVAILABLE:
        avg_gpu_mem = np.mean(performance_data["gpu_memory_used"])
        max_gpu_mem = np.max(performance_data["gpu_memory_used"])
        avg_gpu_util = np.mean(performance_data["gpu_utilization"])
        max_gpu_util = np.max(performance_data["gpu_utilization"])
        
        print(f"GPU平均内存使用: {avg_gpu_mem:.2f} MB")
        print(f"GPU最大内存使用: {max_gpu_mem:.2f} MB")
        print(f"GPU平均使用率: {avg_gpu_util:.2f}%")
        print(f"GPU最大使用率: {max_gpu_util:.2f}%")


def generate_optimization_recommendations():
    """生成优化建议"""
    print("\n=== 优化建议 ===")
    
    # 基于代码分析的优化建议
    recommendations = [
        "1. **内存优化**:",
        "   - 减少临时文件的创建和删除，使用内存缓存替代",
        "   - 及时释放不再使用的模型和张量内存",
        "   - 优化图像预处理流程，减少内存拷贝",
        "",
        "2. **CPU优化**:",
        "   - 并行处理多个请求，使用异步IO",
        "   - 优化模型推理过程，减少计算开销",
        "   - 合理设置线程池大小，避免过度并行",
        "",
        "3. **GPU优化**:",
        "   - 使用批处理推理，提高GPU利用率",
        "   - 优化模型精度，考虑使用半精度(fp16)",
        "   - 合理分配GPU内存，避免内存碎片化",
        "",
        "4. **缓存优化**:",
        "   - 扩大缓存容量，缓存更多的处理结果",
        "   - 优化缓存键生成策略，提高缓存命中率",
        "   - 实现缓存过期机制，避免缓存占用过多内存",
        "",
        "5. **代码优化**:",
        "   - 减少不必要的日志输出，特别是在处理过程中",
        "   - 优化文件IO操作，使用异步IO",
        "   - 减少模块导入开销，使用延迟导入",
        "",
        "6. **架构优化**:",
        "   - 考虑使用模型服务分离，减轻主API负担",
        "   - 实现请求队列，避免系统过载",
        "   - 优化网络传输，减少数据传输量"
    ]
    
    for recommendation in recommendations:
        print(recommendation)


def main():
    """主函数"""
    # 检查下载的图片目录是否存在
    if not os.path.exists(DOWNLOADED_IMAGES_DIR):
        print(f"Error: Downloaded images directory not found at {DOWNLOADED_IMAGES_DIR}")
        print("Please make sure images are downloaded first")
        return
    
    # 检查是否有图片
    image_files = get_downloaded_images()
    if not image_files:
        print(f"Error: No images found in {DOWNLOADED_IMAGES_DIR}")
        print("Please make sure images are downloaded first")
        return
    
    # 运行性能测试
    request_times = run_performance_test()
    
    # 分析瓶颈
    analyze_bottlenecks()
    
    # 生成图表
    generate_plots()
    
    # 生成优化建议
    generate_optimization_recommendations()


if __name__ == "__main__":
    main()
