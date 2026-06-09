#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试MPS加速效果
使用 multi_face_detected_anime 数据集中的图片测试模型推理速度
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image
import requests
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_mps_acceleration():
    """测试MPS加速效果"""
    
    print("=" * 70)
    print("🚀 MPS加速测试")
    print("=" * 70)
    
    # 测试数据目录
    test_dir = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/multi_face_detected_anime/multi_face")
    
    if not test_dir.exists():
        print(f"❌ 测试目录不存在: {test_dir}")
        return
    
    # 获取测试图片
    test_images = list(test_dir.glob("*.jpg"))[:5]  # 取前5张测试
    
    if not test_images:
        print("❌ 没有找到测试图片")
        return
    
    print(f"\n📸 测试图片数量: {len(test_images)}")
    print(f"📁 测试目录: {test_dir}")
    print()
    
    # Model Service URL
    model_service_url = "http://localhost:8001"
    
    # 测试参数
    model_name = "efficientnet_b3_loli_optimized_v2_20260529_133654"
    
    total_time = 0
    success_count = 0
    
    for i, img_path in enumerate(test_images, 1):
        print(f"[{i}/{len(test_images)}] 测试: {img_path.name}")
        
        try:
            # 发送分类请求
            start_time = time.time()
            
            with open(img_path, 'rb') as f:
                files = {'file': (img_path.name, f, 'image/jpeg')}
                data = {
                    'model_name': model_name,
                    'use_attributes': False,
                    'cache_bypass': False
                }
                
                response = requests.post(
                    f"{model_service_url}/api/classify",
                    files=files,
                    data=data,
                    timeout=120  # 增加超时时间
                )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            if response.status_code == 200:
                result = response.json()
                role = result.get('role', 'unknown')
                similarity = result.get('similarity', 0.0)
                
                print(f"  ✅ 角色: {role}")
                print(f"  📊 相似度: {similarity:.4f}")
                print(f"  ⏱️  耗时: {elapsed:.2f}秒")
                
                success_count += 1
            else:
                print(f"  ❌ 请求失败: {response.status_code}")
                print(f"  响应: {response.text[:200]}")
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
        
        print()
    
    # 统计结果
    print("=" * 70)
    print("📊 测试结果汇总")
    print("=" * 70)
    print(f"✅ 成功: {success_count}/{len(test_images)}")
    print(f"⏱️  总耗时: {total_time:.2f}秒")
    print(f"⚡ 平均耗时: {total_time/success_count:.2f}秒/张" if success_count > 0 else "N/A")
    print()
    
    # 性能评估
    avg_time = total_time / success_count if success_count > 0 else float('inf')
    
    if avg_time < 2:
        print("🎉 性能优秀! (< 2秒/张)")
        print("   → MPS加速可能已生效")
    elif avg_time < 5:
        print("✨ 性能良好 (2-5秒/张)")
        print("   → 可能是CPU推理或MPS未充分利用")
    elif avg_time < 10:
        print("⚠️  性能一般 (5-10秒/张)")
        print("   → 建议检查MPS配置")
    else:
        print("❌ 性能较差 (> 10秒/张)")
        print("   → 可能在使用CPU推理")
        print("   → 建议:")
        print("     1. 检查PyTorch MPS支持: python -c 'import torch; print(torch.backends.mps.is_available())'")
        print("     2. 检查model-service日志确认使用的设备")
        print("     3. 考虑优化模型或使用更小的模型")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    test_mps_acceleration()
