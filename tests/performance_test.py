import time
import requests
import psutil
import os
import json
from concurrent.futures import ThreadPoolExecutor

# 后端API地址
API_URL = "http://localhost:8000/api/classify"

# 测试图片路径
TEST_IMAGE = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/downloaded_images/arona/140883369_p0_master1200.jpg"

# 确保测试图片存在
if not os.path.exists(TEST_IMAGE):
    # 如果不存在，创建一个简单的测试图片
    from PIL import Image
    img = Image.new('RGB', (200, 200), color='red')
    img.save(TEST_IMAGE)
    print(f"Created test image: {TEST_IMAGE}")

# 获取后端进程ID
def get_backend_pid():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.name().lower() and 'app.py' in ' '.join(proc.cmdline()):
                return proc.pid
        except:
            pass
    return None

# 监控进程性能
def monitor_process(pid, duration, interval=0.1):
    results = []
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        try:
            proc = psutil.Process(pid)
            cpu_percent = proc.cpu_percent(interval=interval)
            memory_percent = proc.memory_percent()
            memory_mb = proc.memory_info().rss / (1024 * 1024)
            results.append({
                'time': time.time() - start_time,
                'cpu': cpu_percent,
                'memory_percent': memory_percent,
                'memory_mb': memory_mb
            })
        except:
            break
        time.sleep(interval)
    
    return results

# 发送单个请求
def send_request():
    try:
        start_time = time.time()
        with open(TEST_IMAGE, 'rb') as f:
            files = {'file': f}
            data = {
                'use_model': True,
                'use_attributes': False,  # 禁用属性预测以提高速度
                'model_name': 'mobilenet_v2',
                'cache_bypass': True  # 禁用缓存以获得真实处理时间
            }
            print(f"Sending request with model: {data['model_name']}")
            response = requests.post(API_URL, files=files, data=data, timeout=60)
            request_time = time.time() - start_time
            print(f"Request completed in {request_time:.2f} seconds with status code: {response.status_code}")
            # 打印完整的响应内容
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    print(f"Response: {response_data.get('role', 'unknown')} with similarity {response_data.get('similarity', 0):.2f}")
                    print(f"Processing time: {response_data.get('processing_time', 0):.2f} seconds")
                    print(f"Attributes count: {len(response_data.get('attributes', []))}")
                    print(f"AI predicted role: {response_data.get('ai_predicted_role', 'none')}")
                except Exception as e:
                    print(f"Error parsing response: {e}")
                    print(f"Response content: {response.text[:500]}")
            else:
                print(f"Error response: {response.text[:500]}")
        return response.status_code
    except Exception as e:
        print(f"Error sending request: {e}")
        return 500

# 运行性能测试
def run_performance_test():
    # 获取后端进程ID
    backend_pid = get_backend_pid()
    if not backend_pid:
        print("Backend process not found. Make sure the backend is running.")
        return
    
    print(f"Found backend process with PID: {backend_pid}")
    
    # 测试场景1: 单个请求
    print("\n=== Test 1: Single Request ===")
    monitor_thread = ThreadPoolExecutor(max_workers=1)
    monitor_future = monitor_thread.submit(monitor_process, backend_pid, 10)
    
    # 发送单个请求
    start_time = time.time()
    status_code = send_request()
    request_time = time.time() - start_time
    
    # 获取监控结果
    monitor_results = monitor_future.result()
    
    # 分析结果
    if monitor_results:
        max_cpu = max(r['cpu'] for r in monitor_results)
        max_memory = max(r['memory_mb'] for r in monitor_results)
        avg_cpu = sum(r['cpu'] for r in monitor_results) / len(monitor_results)
        avg_memory = sum(r['memory_mb'] for r in monitor_results) / len(monitor_results)
        
        print(f"Request status code: {status_code}")
        print(f"Request time: {request_time:.2f} seconds")
        print(f"Max CPU usage: {max_cpu:.2f}%")
        print(f"Max memory usage: {max_memory:.2f} MB")
        print(f"Average CPU usage: {avg_cpu:.2f}%")
        print(f"Average memory usage: {avg_memory:.2f} MB")
    
    # 测试场景2: 并发请求
    print("\n=== Test 2: Concurrent Requests ===")
    concurrent_requests = 2
    
    monitor_future = monitor_thread.submit(monitor_process, backend_pid, 20)
    
    # 发送并发请求
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(send_request) for _ in range(concurrent_requests)]
        status_codes = [f.result() for f in futures]
    request_time = time.time() - start_time
    
    # 获取监控结果
    monitor_results = monitor_future.result()
    
    # 分析结果
    if monitor_results:
        max_cpu = max(r['cpu'] for r in monitor_results)
        max_memory = max(r['memory_mb'] for r in monitor_results)
        avg_cpu = sum(r['cpu'] for r in monitor_results) / len(monitor_results)
        avg_memory = sum(r['memory_mb'] for r in monitor_results) / len(monitor_results)
        
        print(f"Concurrent requests: {concurrent_requests}")
        print(f"Request status codes: {status_codes}")
        print(f"Total request time: {request_time:.2f} seconds")
        print(f"Max CPU usage: {max_cpu:.2f}%")
        print(f"Max memory usage: {max_memory:.2f} MB")
        print(f"Average CPU usage: {avg_cpu:.2f}%")
        print(f"Average memory usage: {avg_memory:.2f} MB")
    
    monitor_thread.shutdown()
    
    print("\n=== Test Completed ===")

if __name__ == "__main__":
    run_performance_test()
