#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试重构效果

验证重构后的模块是否能正常导入和使用
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """
    测试所有重构后的模块是否能正常导入
    """
    print("开始测试模块导入...")
    
    # 测试 monitoring_system 模块
    try:
        from src.utils.monitoring_system import SystemMonitor
        print("✓ monitoring_system 模块导入成功")
    except Exception as e:
        print(f"✗ monitoring_system 模块导入失败: {e}")
    
    # 测试 preprocessing 模块
    try:
        from src.core.preprocessing.preprocessing import ImagePreprocessor
        print("✓ preprocessing 模块导入成功")
    except Exception as e:
        print(f"✗ preprocessing 模块导入失败: {e}")
    
    # 测试 mediapipe_keypoint_detector 模块
    try:
        from src.core.keypoint.mediapipe_keypoint_detector import MediaPipeKeypointDetector
        print("✓ mediapipe_keypoint_detector 模块导入成功")
    except Exception as e:
        print(f"✗ mediapipe_keypoint_detector 模块导入失败: {e}")
    
    # 测试 jm_client_impl 模块
    try:
        from spider_image_system.src.jmcomic.jm_client_impl import JmHtmlClient
        print("✓ jm_client_impl 模块导入成功")
    except Exception as e:
        print(f"✗ jm_client_impl 模块导入失败: {e}")
    
    # 测试 jm_plugin 模块
    try:
        from spider_image_system.src.jmcomic.jm_plugin import JmOptionPlugin
        print("✓ jm_plugin 模块导入成功")
    except Exception as e:
        print(f"✗ jm_plugin 模块导入失败: {e}")
    
    # 测试 jm_toolkit 模块
    try:
        from spider_image_system.src.jmcomic.jm_toolkit import JmcomicText
        print("✓ jm_toolkit 模块导入成功")
    except Exception as e:
        print(f"✗ jm_toolkit 模块导入失败: {e}")
    
    # 测试 image_processor 模块
    try:
        from src.backend.services.image_processor import preprocess_image
        print("✓ image_processor 模块导入成功")
    except Exception as e:
        print(f"✗ image_processor 模块导入失败: {e}")
    
    print("模块导入测试完成")


def test_basic_functions():
    """
    测试基本功能
    """
    print("\n开始测试基本功能...")
    
    # 测试 SystemMonitor
    try:
        from src.utils.monitoring_system import SystemMonitor
        monitor = SystemMonitor()
        print("✓ SystemMonitor 初始化成功")
    except Exception as e:
        print(f"✗ SystemMonitor 测试失败: {e}")
    
    # 测试 ImagePreprocessor
    try:
        from src.core.preprocessing.preprocessing import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        print("✓ ImagePreprocessor 初始化成功")
    except Exception as e:
        print(f"✗ ImagePreprocessor 测试失败: {e}")
    
    # 测试 MediaPipeKeypointDetector
    try:
        from src.core.keypoint.mediapipe_keypoint_detector import MediaPipeKeypointDetector
        detector = MediaPipeKeypointDetector()
        print("✓ MediaPipeKeypointDetector 初始化成功")
        detector.close()
    except Exception as e:
        print(f"✗ MediaPipeKeypointDetector 测试失败: {e}")
    
    print("基本功能测试完成")


if __name__ == "__main__":
    test_imports()
    test_basic_functions()
    print("\n重构测试完成！")
