#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试重构效果

验证重构后的核心模块是否能正常导入和使用
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_monitoring_system():
    """
    测试监控系统模块
    """
    print("测试监控系统模块...")
    try:
        from src.utils.monitoring_system import SystemMonitor
        monitor = SystemMonitor()
        print("✓ SystemMonitor 初始化成功")
        return True
    except Exception as e:
        print(f"✗ SystemMonitor 测试失败: {e}")
        return False


def test_preprocessing():
    """
    测试预处理模块
    """
    print("测试预处理模块...")
    try:
        from src.core.preprocessing.preprocessing import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        print("✓ ImagePreprocessor 初始化成功")
        return True
    except Exception as e:
        print(f"✗ ImagePreprocessor 测试失败: {e}")
        return False


def test_keypoint_detector():
    """
    测试关键点检测器模块
    """
    print("测试关键点检测器模块...")
    try:
        from src.core.keypoint.mediapipe_keypoint_detector import MediaPipeKeypointDetector
        detector = MediaPipeKeypointDetector()
        print("✓ MediaPipeKeypointDetector 初始化成功")
        detector.close()
        return True
    except Exception as e:
        print(f"✗ MediaPipeKeypointDetector 测试失败: {e}")
        return False


def test_jm_modules():
    """
    测试JM模块
    """
    print("测试JM模块...")
    try:
        from spider_image_system.src.jmcomic.jm_client_impl import JmHtmlClient
        print("✓ JmHtmlClient 导入成功")
        
        from spider_image_system.src.jmcomic.jm_plugin import JmOptionPlugin
        print("✓ JmOptionPlugin 导入成功")
        
        from spider_image_system.src.jmcomic.jm_toolkit import JmcomicText
        print("✓ JmcomicText 导入成功")
        
        return True
    except Exception as e:
        print(f"✗ JM模块测试失败: {e}")
        return False


def test_image_processor():
    """
    测试图像处理模块
    """
    print("测试图像处理模块...")
    try:
        from src.backend.services.image_processor import preprocess_image
        print("✓ preprocess_image 导入成功")
        return True
    except Exception as e:
        print(f"✗ image_processor 测试失败: {e}")
        return False


if __name__ == "__main__":
    print("开始测试重构效果...\n")
    
    tests = [
        test_monitoring_system,
        test_preprocessing,
        test_keypoint_detector,
        test_jm_modules,
        test_image_processor
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"测试完成: {passed}/{total} 测试通过")
    
    if passed == total:
        print("✓ 所有重构模块测试通过！")
    else:
        print("✗ 部分测试失败，请检查重构结果")
