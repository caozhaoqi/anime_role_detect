#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示以图搜图功能
"""

import os
import sys
import time

# 设置环境变量
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=" * 60)
print("以图搜图功能演示")
print("=" * 60)

# 初始化搜索服务
print("\n1. 初始化图像搜索服务...")
start_time = time.time()

from src.services.search_service.image_search_service import ImageSearchService

search_service = ImageSearchService()
print(f"   ✓ 服务初始化完成 ({time.time() - start_time:.2f}秒)")

# 构建索引
print("\n2. 构建搜索索引...")
dataset_dir = os.path.join(project_root, "data", "merged_english_dataset")
start_time = time.time()
count = search_service.build_index_from_dataset(dataset_dir)
print(f"   ✓ 索引构建完成，共添加 {count} 张图像 ({time.time() - start_time:.2f}秒)")

# 获取统计信息
stats = search_service.get_stats()
print(f"   - 总图像数: {stats.get('total_images', 0)}")
print(f"   - 索引维度: {stats.get('index_dimension', 0)}")

# 选择测试图片
print("\n3. 选择测试图片...")
test_image_path = None
for role_name in os.listdir(dataset_dir):
    role_dir = os.path.join(dataset_dir, role_name)
    if os.path.isdir(role_dir):
        for img_file in os.listdir(role_dir):
            if img_file.lower().endswith((".jpg", ".png")):
                test_image_path = os.path.join(role_dir, img_file)
                print(f"   ✓ 选择测试图片: {test_image_path}")
                break
        if test_image_path:
            break

# 搜索相似图像
if test_image_path:
    print("\n4. 搜索相似图像...")
    start_time = time.time()
    results = search_service.search_by_path(test_image_path, top_k=5)
    print(f"   ✓ 搜索完成 ({time.time() - start_time:.2f}秒)")

    print("\n   搜索结果:")
    for i, (path, similarity) in enumerate(results, 1):
        role = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        print(f"   {i}. {role}/{filename} (相似度: {similarity:.4f})")

print("\n" + "=" * 60)
print("演示完成")
print("=" * 60)
