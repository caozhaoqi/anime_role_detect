#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试搜索功能框架
由于macOS上的PyTorch限制，此测试仅验证框架结构
"""

import os
import sys

# 设置环境变量
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("="*60)
print("搜索功能框架测试")
print("="*60)

# 1. 测试搜索服务模块导入
print("\n1. 测试搜索服务模块导入...")
try:
    from src.services.search_service.image_search_service import ImageSearchService
    print("   ✓ ImageSearchService 导入成功")
except Exception as e:
    print(f"   ✗ ImageSearchService 导入失败: {e}")

# 2. 测试视频识别服务模块导入
print("\n2. 测试视频识别服务模块导入...")
try:
    from src.services.video_service.video_recognition_service import VideoRecognitionService
    print("   ✓ VideoRecognitionService 导入成功")
except Exception as e:
    print(f"   ✗ VideoRecognitionService 导入失败: {e}")

# 3. 测试搜索路由导入
print("\n3. 测试搜索路由模块导入...")
try:
    from src.api.routes.search_routes import router, init_search_service
    print("   ✓ 搜索路由模块导入成功")
    print("   ✓ 可用端点:")
    print("     - POST /api/search/image (以图搜图)")
    print("     - POST /api/search/build-index (构建索引)")
    print("     - GET /api/search/stats (统计信息)")
    print("     - POST /api/search/video/recognize (视频识别)")
except Exception as e:
    print(f"   ✗ 搜索路由模块导入失败: {e}")

# 4. 测试数据集目录
print("\n4. 测试数据集目录...")
dataset_dir = os.path.join(project_root, "data", "merged_english_dataset")
if os.path.exists(dataset_dir):
    role_count = len([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    total_images = 0
    for role_dir in os.listdir(dataset_dir):
        role_path = os.path.join(dataset_dir, role_dir)
        if os.path.isdir(role_path):
            total_images += len([f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.png'))])
    print(f"   ✓ 数据集目录存在")
    print(f"   - 角色数量: {role_count}")
    print(f"   - 图像总数: {total_images}")
else:
    print("   ✗ 数据集目录不存在")

# 5. 测试API服务启动
print("\n5. 测试API服务启动...")
try:
    # 启动API服务
    import subprocess
    import time
    import requests
    
    cmd = [sys.executable, "src/api/run_api.py"]
    process = subprocess.Popen(cmd, cwd=project_root, env=os.environ.copy())
    
    # 等待服务启动
    time.sleep(10)
    
    # 检查服务状态
    try:
        response = requests.get("http://localhost:8001/api/health")
        if response.status_code == 200:
            print("   ✓ API服务启动成功")
            
            # 检查搜索端点
            response = requests.get("http://localhost:8001/api/openapi.json")
            if response.status_code == 200:
                import json
                data = json.loads(response.text)
                search_endpoints = [p for p in data['paths'].keys() if '/api/search' in p]
                if search_endpoints:
                    print("   ✓ 搜索端点已注册:")
                    for ep in search_endpoints:
                        print(f"     - {ep}")
                else:
                    print("   ✗ 搜索端点未注册")
        else:
            print(f"   ✗ API服务响应异常: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ✗ 无法连接到API服务: {e}")
    
    # 停止服务
    process.terminate()
    process.wait()
    
except Exception as e:
    print(f"   ✗ 启动API服务失败: {e}")

print("\n" + "="*60)
print("测试完成")
print("="*60)
print("\n注意：由于macOS系统的Mutex锁限制，PyTorch相关功能")
print("需要在Linux环境下运行才能获得完整功能。")
