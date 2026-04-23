#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多角色检测功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.detection.multi_role_detection import MultiRoleDetector


def test_multi_role_detection():
    """
    测试多角色检测功能
    """
    try:
        # 初始化检测器
        detector = MultiRoleDetector(model_name="resnet18_loli8")
        print("初始化检测器成功")
        
        # 加载模型
        detector._load_trained_model()
        print(f"模型加载状态: model={detector.model is not None}, class_to_idx={detector.class_to_idx is not None}")
        
        # 检测角色
        results = detector.detect_roles("temp/temp_1776678803_微信图片_20260204115846_481_347.jpg")
        print(f"检测结果: {results}")
        print(f"检测到 {len(results)} 个角色")
        
        return results
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    print("开始测试多角色检测功能...")
    results = test_multi_role_detection()
    print("测试完成")
