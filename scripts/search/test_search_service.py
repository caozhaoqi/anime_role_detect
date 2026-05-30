#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试图像搜索与视频识别服务
"""

import os
import sys
import requests

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_image_search():
    """测试以图搜图功能"""
    print("=" * 50)
    print("测试以图搜图功能")
    print("=" * 50)

    # 选择一张测试图片
    test_image_path = None
    dataset_dir = os.path.join(project_root, "data", "merged_english_dataset")

    # 找到任意一张图片
    for role_dir in os.listdir(dataset_dir):
        role_path = os.path.join(dataset_dir, role_dir)
        if os.path.isdir(role_path):
            for img_file in os.listdir(role_path):
                if img_file.lower().endswith((".jpg", ".png")):
                    test_image_path = os.path.join(role_path, img_file)
                    break
            if test_image_path:
                break

    if not test_image_path:
        print("❌ 未找到测试图片")
        return

    print(f"测试图片: {test_image_path}")

    # 构建索引（如果需要）
    print("\n1. 构建搜索索引...")
    try:
        response = requests.post("http://localhost:8001/api/search/build-index")
        print(f"   状态: {response.status_code}")
        result = response.json()
        print(f"   添加图像数: {result.get('added_images', 0)}")
    except Exception as e:
        print(f"   构建索引失败（可能已存在）: {e}")

    # 获取统计信息
    print("\n2. 获取索引统计...")
    try:
        response = requests.get("http://localhost:8001/api/search/stats")
        result = response.json()
        print(f"   总图像数: {result.get('total_images', 0)}")
    except Exception as e:
        print(f"   获取统计失败: {e}")

    # 搜索相似图像
    print("\n3. 搜索相似图像...")
    try:
        with open(test_image_path, "rb") as f:
            response = requests.post(
                "http://localhost:8001/api/search/image", files={"file": f}, data={"top_k": 5}
            )

        if response.status_code == 200:
            result = response.json()
            print(f"   找到相似图像: {result['count']} 张")
            print("\n   搜索结果:")
            for i, item in enumerate(result["results"], 1):
                print(
                    f"   {i}. {os.path.basename(item['path'])} (相似度: {item['similarity']:.4f})"
                )
        else:
            print(f"   搜索失败: {response.status_code}")

    except Exception as e:
        print(f"   搜索失败: {e}")


def test_video_recognition():
    """测试视频识别功能"""
    print("\n" + "=" * 50)
    print("测试视频识别功能")
    print("=" * 50)

    # 检查是否有测试视频
    test_video_path = None

    # 检查常见视频路径
    video_paths = [
        os.path.join(project_root, "test_video.mp4"),
        os.path.join(project_root, "demo.mp4"),
    ]

    for path in video_paths:
        if os.path.exists(path):
            test_video_path = path
            break

    if test_video_path:
        print(f"测试视频: {test_video_path}")

        try:
            with open(test_video_path, "rb") as f:
                response = requests.post(
                    "http://localhost:8001/api/video/recognize",
                    files={"file": f},
                    data={"frame_interval": 1.0, "confidence_threshold": 0.5},
                )

            if response.status_code == 200:
                result = response.json()
                print(f"\n   处理帧数: {result['total_frames']}")
                print(f"   检测到角色: {result['detections']} 次")
                print("\n   识别到的角色:")
                for role in result["roles"]:
                    print(f"   - {role['role']}: {role['count']} 次")
            else:
                print(f"   视频识别失败: {response.status_code}")

        except Exception as e:
            print(f"   视频识别失败: {e}")
    else:
        print("❌ 未找到测试视频，跳过视频识别测试")


def test_danmaku_mode():
    """测试弹幕模式"""
    print("\n" + "=" * 50)
    print("测试弹幕模式")
    print("=" * 50)

    print("注意：弹幕模式需要摄像头权限，此测试仅演示API调用")

    try:
        # 启动弹幕模式
        response = requests.post(
            "http://localhost:8001/api/video/danmaku/start",
            data={"video_source": 0, "frame_interval": 1.0},
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")

            # 获取最新弹幕
            response = requests.get("http://localhost:8001/api/danmaku/latest?count=5")
            if response.status_code == 200:
                result = response.json()
                print(f"当前弹幕数量: {result['count']}")

        else:
            print(f"❌ 启动弹幕模式失败: {response.status_code}")

    except Exception as e:
        print(f"⚠️ 弹幕模式测试失败（可能需要摄像头）: {e}")


if __name__ == "__main__":
    print("图像搜索与视频识别服务测试")
    print("-" * 50)

    # 检查服务是否运行
    try:
        response = requests.get("http://localhost:8001/api/health")
        if response.status_code == 200:
            print("✅ 服务已启动")
        else:
            print("❌ 服务未正常运行")
            print("请先运行: python3 src/services/search_service/app.py")
            sys.exit(1)
    except Exception as e:
        print("❌ 无法连接到服务")
        print("请先运行: python3 src/services/search_service/app.py")
        sys.exit(1)

    # 运行测试
    test_image_search()
    test_video_recognition()
    test_danmaku_mode()

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)
