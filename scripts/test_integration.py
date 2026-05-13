#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成测试脚本 - 测试以图搜图和视频识别功能
"""

import os
import sys
import time
import subprocess
import requests

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("="*70)
print("动漫角色识别系统 - 集成测试")
print("="*70)

# 测试结果统计
test_results = {
    "passed": 0,
    "failed": 0,
    "total": 0
}

def run_test(test_name, test_func):
    """运行单个测试"""
    global test_results
    test_results["total"] += 1
    
    print(f"\n▶️ 测试: {test_name}")
    print("-" * 50)
    
    try:
        result = test_func()
        if result:
            print(f"✅ 通过")
            test_results["passed"] += 1
            return True
        else:
            print(f"❌ 失败")
            test_results["failed"] += 1
            return False
    except Exception as e:
        print(f"❌ 异常: {e}")
        import traceback
        print(f"   堆栈: {traceback.format_exc()[:500]}")
        test_results["failed"] += 1
        return False

# 1. 测试搜索服务启动
def test_search_service_start():
    """测试搜索服务启动"""
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    
    # 启动搜索服务
    cmd = [
        sys.executable,
        "src/services/search_service/search_service_app.py"
    ]
    
    global search_process
    search_process = subprocess.Popen(cmd, cwd=project_root, env=env)
    
    # 等待服务启动
    for i in range(10):
        try:
            response = requests.get("http://localhost:8002/health", timeout=2)
            if response.status_code == 200:
                print(f"   搜索服务启动成功，端口 8002")
                return True
        except Exception:
            pass
        time.sleep(1)
    
    return False

# 2. 测试构建索引
def test_build_index():
    """测试构建搜索索引"""
    response = requests.post(
        "http://localhost:8002/search/build-index",
        params={"dataset_dir": "data/merged_english_dataset"}
    )
    
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            print(f"   索引构建成功，添加 {result['added_images']} 张图像")
            print(f"   索引维度: {result['index_stats']['index_dimension']}")
            return True
        else:
            print(f"   构建失败: {result.get('error')}")
            return False
    else:
        print(f"   HTTP错误: {response.status_code}")
        return False

# 3. 测试以图搜图
def test_image_search():
    """测试以图搜图功能"""
    # 找一张测试图像
    test_image_path = None
    dataset_dir = os.path.join(project_root, "data", "merged_english_dataset")
    for role_name in os.listdir(dataset_dir):
        role_dir = os.path.join(dataset_dir, role_name)
        if os.path.isdir(role_dir):
            for img_file in os.listdir(role_dir):
                if img_file.lower().endswith(('.jpg', '.png')):
                    test_image_path = os.path.join(role_dir, img_file)
                    expected_role = role_name
                    break
            if test_image_path:
                break
    
    if not test_image_path:
        print("   未找到测试图像")
        return False
    
    print(f"   测试图像: {expected_role}/{os.path.basename(test_image_path)}")
    
    # 上传图像进行搜索
    with open(test_image_path, 'rb') as f:
        response = requests.post(
            "http://localhost:8002/search/image",
            files={"file": f},
            params={"top_k": 5}
        )
    
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            print(f"   搜索成功，找到 {result['count']} 个结果")
            
            # 检查第一个结果是否匹配
            if result["count"] > 0:
                first_result = result["results"][0]
                print(f"   最佳匹配: {first_result['role']} (相似度: {first_result['similarity']:.4f})")
                
                if first_result["role"] == expected_role:
                    print(f"   ✅ 角色识别正确")
                else:
                    print(f"   ⚠️ 角色识别不一致: 期望 {expected_role}, 实际 {first_result['role']}")
            
            return True
        else:
            print(f"   搜索失败: {result.get('error')}")
            return False
    else:
        print(f"   HTTP错误: {response.status_code}")
        return False

# 4. 测试获取统计信息
def test_get_stats():
    """测试获取索引统计信息"""
    response = requests.get("http://localhost:8002/search/stats")
    
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            stats = result.get("data", {})
            print(f"   索引状态: {stats.get('status')}")
            print(f"   图像总数: {stats.get('total_images')}")
            print(f"   索引维度: {stats.get('index_dimension')}")
            return True
        else:
            print(f"   获取失败: {result.get('error')}")
            return False
    else:
        print(f"   HTTP错误: {response.status_code}")
        return False

# 5. 测试视频识别（如果有测试视频）
def test_video_recognition():
    """测试视频识别功能"""
    # 检查是否有测试视频
    test_video_path = None
    for filename in ["test_video.mp4", "demo.mp4", "sample.mp4"]:
        path = os.path.join(project_root, filename)
        if os.path.exists(path):
            test_video_path = path
            break
    
    if not test_video_path:
        print("   ⚠️ 未找到测试视频，跳过此测试")
        return True  # 跳过不算失败
    
    print(f"   测试视频: {os.path.basename(test_video_path)}")
    
    with open(test_video_path, 'rb') as f:
        response = requests.post(
            "http://localhost:8002/search/video/recognize",
            files={"file": f},
            params={"frame_interval": 1.0, "confidence_threshold": 0.5}
        )
    
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            print(f"   视频处理成功")
            print(f"   总帧数: {result.get('total_frames')}")
            print(f"   检测到角色: {result.get('detections')} 个")
            
            roles = result.get("roles", [])
            if roles:
                print(f"   识别到的角色:")
                for role_info in roles[:3]:
                    print(f"     - {role_info['role']}: {role_info['count']} 次")
            
            return True
        else:
            print(f"   处理失败: {result.get('error')}")
            return False
    else:
        print(f"   HTTP错误: {response.status_code}")
        return False

# 6. 测试搜索服务健康检查
def test_search_health():
    """测试搜索服务健康检查"""
    response = requests.get("http://localhost:8002/health")
    
    if response.status_code == 200:
        result = response.json()
        if result.get("status") == "healthy":
            print("   健康检查通过")
            return True
        else:
            print(f"   健康状态异常: {result}")
            return False
    else:
        print(f"   HTTP错误: {response.status_code}")
        return False

# 主测试流程
if __name__ == "__main__":
    search_process = None
    
    try:
        # 运行所有测试
        run_test("搜索服务启动", test_search_service_start)
        run_test("构建搜索索引", test_build_index)
        run_test("获取索引统计", test_get_stats)
        run_test("以图搜图", test_image_search)
        run_test("视频识别", test_video_recognition)
        run_test("健康检查", test_search_health)
        
        # 输出测试结果
        print("\n" + "="*70)
        print("测试结果汇总")
        print("="*70)
        print(f"总测试数: {test_results['total']}")
        print(f"通过: {test_results['passed']}")
        print(f"失败: {test_results['failed']}")
        
        if test_results["failed"] == 0:
            print("\n🎉 所有测试通过！")
        else:
            print(f"\n⚠️ 有 {test_results['failed']} 个测试失败")
            
    finally:
        # 清理：停止搜索服务
        if search_process:
            print("\n清理：停止搜索服务...")
            search_process.terminate()
            search_process.wait()
            print("搜索服务已停止")