#!/usr/bin/env python3
"""
测试内存泄漏问题
"""

import os
import sys
import time
import psutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.backend.services.classification_service import classify_image


def get_memory_usage():
    """获取当前进程的内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转换为MB


def test_memory_leak():
    """测试内存泄漏"""
    print("=== 测试内存泄漏 ===")
    
    # 测试图片目录
    test_dir = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/downloaded_images"
    
    # 获取测试图片
    test_images = []
    
    # 遍历日奈和阿罗娜的图片
    for role in ["ri4nai4", "a1luo2na4"]:
        role_dir = os.path.join(test_dir, role)
        if os.path.exists(role_dir):
            for img_file in os.listdir(role_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    test_images.append(os.path.join(role_dir, img_file))
    
    print(f"找到 {len(test_images)} 张测试图片")
    print()
    
    # 初始内存使用
    initial_memory = get_memory_usage()
    print(f"初始内存使用: {initial_memory:.2f} MB")
    
    # 测试配置
    test_configs = [
        ("default", False, False, False, False, "默认模型 (CLIP + FAISS)"),
        ("default", False, True, False, False, "专用模型 (EfficientNet)"),
    ]
    
    # 测试轮数
    test_rounds = 3
    
    for round_num in range(test_rounds):
        print(f"\n=== 第 {round_num + 1} 轮测试 ===")
        
        for model_name, use_coreml, use_model, use_deepdanbooru, use_attributes, description in test_configs:
            print(f"\n测试: {description}")
            
            # 测试前内存
            before_memory = get_memory_usage()
            print(f"测试前内存: {before_memory:.2f} MB")
            
            # 测试前5张图片
            for i, img_path in enumerate(test_images[:5]):
                print(f"  测试第 {i+1} 张图片...")
                
                try:
                    role, similarity, boxes, mode, attributes, text_detections = classify_image(
                        img_path,
                        use_coreml=use_coreml,
                        use_model=use_model,
                        use_deepdanbooru=use_deepdanbooru,
                        use_attributes=use_attributes,
                        model_name=model_name
                    )
                    print(f"    结果: {role}, 相似度: {similarity:.4f}")
                except Exception as e:
                    print(f"    错误: {e}")
                
                # 短暂休眠，避免CPU过载
                time.sleep(0.5)
            
            # 测试后内存
            after_memory = get_memory_usage()
            print(f"测试后内存: {after_memory:.2f} MB")
            print(f"内存变化: {after_memory - before_memory:.2f} MB")
    
    # 最终内存使用
    final_memory = get_memory_usage()
    print(f"\n=== 测试完成 ===")
    print(f"初始内存: {initial_memory:.2f} MB")
    print(f"最终内存: {final_memory:.2f} MB")
    print(f"总内存变化: {final_memory - initial_memory:.2f} MB")
    
    if final_memory - initial_memory < 100:  # 内存增长小于100MB
        print("✅ 内存泄漏测试通过！内存使用稳定")
    else:
        print("❌ 内存泄漏测试失败！内存使用增长过大")


if __name__ == "__main__":
    test_memory_leak()
