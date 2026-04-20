#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试扩展后的CharacterClassifier
"""

import sys
import os

# 添加脚本所在目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CharacterClassifier import CharacterClassifier

def test_extended_classifier():
    """
    测试扩展后的分类器
    """
    # 初始化分类器
    classifier = CharacterClassifier()
    
    # 测试一些之前未找到的角色
    test_characters = [
        "Yoimiya",          # 宵宫（英文）
        "Klee",             # 可莉（英文）
        "Nahida",           # 纳西妲（英文）
        "Raiden Shogun",    # 雷电将军（英文）
        "Yae Miko",         # 八重神子（英文）
        "Kokomi",           # 珊瑚宫心海（英文）
        "Furina",           # 芙宁娜（英文）
        "Navia",            # 娜维娅（英文）
        "Clara",            # 克拉拉（英文）
        "Seele",            # 希儿（英文）
        "Bronya",           # 布洛妮娅（英文）
        "Kafka",            # 卡芙卡（英文）
        "Himeko",           # 姬子（英文）
        "Silver Wolf",      # 银狼（英文）
        "Sparkle",          # 花火（英文）
        "Black Swan",       # 黑天鹅（英文）
        "Acheron",          # 黄泉（英文）
        "Firefly",          # 流萤（英文）
        "Robin",            # 知更鸟（英文）
    ]
    
    print("=" * 80)
    print("测试扩展后的CharacterClassifier")
    print("=" * 80)
    
    # 统计结果
    results = {
        "萝莉": 0,
        "可能具有部分萝莉特征": 0,
        "不属于萝莉": 0,
        "未找到角色": 0
    }
    
    for name in test_characters:
        result, tags, category = classifier.classify(name)
        print(result)
        print("-" * 40)
        
        # 统计结果
        results[category] += 1
    
    # 输出统计结果
    print("=" * 80)
    print("测试结果统计:")
    for key, value in results.items():
        print(f"{key}: {value}")
    print("=" * 80)

if __name__ == "__main__":
    test_extended_classifier()
