#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整演示：以图搜图和视频实时抽帧识别功能
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
print("动漫角色识别系统 - 功能演示")
print("=" * 60)

# 1. 测试以图搜图功能
print("\n" + "=" * 60)
print("功能一：以图搜图")
print("=" * 60)

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
stats = search_service.get_index_stats()
print(f"   - 总图像数: {stats.get('total_images', 0)}")
print(f"   - 索引维度: {stats.get('index_dimension', 0)}")

# 选择测试图片
print("\n3. 选择测试图片...")
test_image_path = None
selected_role = None
for role_name in os.listdir(dataset_dir):
    role_dir = os.path.join(dataset_dir, role_name)
    if os.path.isdir(role_dir):
        for img_file in os.listdir(role_dir):
            if img_file.lower().endswith((".jpg", ".png")):
                test_image_path = os.path.join(role_dir, img_file)
                selected_role = role_name
                print(f"   ✓ 选择测试图片: {role_name}/{img_file}")
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
    print("   " + "-" * 50)
    for i, (path, similarity) in enumerate(results, 1):
        role = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        match_mark = "✓" if role == selected_role else " "
        print(f"   {i}. [{match_mark}] {role}/{filename} (相似度: {similarity:.4f})")

# 2. 测试视频识别功能
print("\n\n" + "=" * 60)
print("功能二：视频实时抽帧识别")
print("=" * 60)

print("\n1. 检查测试视频...")
test_video_path = None
video_paths = [
    os.path.join(project_root, "test_video.mp4"),
    os.path.join(project_root, "demo.mp4"),
]

for path in video_paths:
    if os.path.exists(path):
        test_video_path = path
        print(f"   ✓ 找到测试视频: {path}")
        break

if test_video_path:
    print("\n2. 初始化视频识别服务...")
    from src.services.video_service.video_recognition_service import VideoRecognitionService

    video_service = VideoRecognitionService(frame_interval=1.0, confidence_threshold=0.5)
    print("   ✓ 服务初始化完成")

    print("\n3. 处理视频...")
    start_time = time.time()
    results = video_service.process_video_file(test_video_path)
    print(f"   ✓ 处理完成 ({time.time() - start_time:.2f}秒)")

    # 统计识别结果
    if results:
        role_counts = {}
        for result in results:
            role = result["role"]
            role_counts[role] = role_counts.get(role, 0) + 1

        print("\n   识别结果统计:")
        print("   " + "-" * 50)
        for role, count in sorted(role_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {role}: {count} 次")

        print("\n   部分识别时间点:")
        print("   " + "-" * 50)
        for result in results[:5]:
            print(
                f"   时间 {result['timestamp']:.2f}s: {result['role']} (置信度: {result['similarity']:.4f})"
            )
    else:
        print("   未识别到任何角色")
else:
    print("   ⚠️ 未找到测试视频，跳过视频识别测试")
    print("   提示：请将测试视频命名为 test_video.mp4 或 demo.mp4 放在项目根目录")

print("\n\n" + "=" * 60)
print("演示完成")
print("=" * 60)
print("\n功能说明：")
print("1. 以图搜图：使用CLIP模型提取图像特征，通过Faiss向量索引实现快速搜索")
print("2. 视频识别：使用OpenCV抽帧，结合分类服务实现实时角色识别")
print("3. 弹幕模式：可通过回调函数实现实时弹幕显示角色信息")
