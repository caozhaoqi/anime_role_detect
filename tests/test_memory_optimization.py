#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试内存优化效果，验证LRU缓存机制
"""

import os
import sys
import time
import psutil
import gc

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from core.feature_extraction.feature_extraction import FeatureExtraction
from core.classification.classification import Classification
from PIL import Image

def get_memory_usage():
    """获取当前进程的内存使用情况"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss': mem_info.rss / (1024 * 1024),  # MB
        'vms': mem_info.vms / (1024 * 1024),  # MB
        'percent': process.memory_percent()
    }

def test_memory_optimization():
    """测试内存优化效果"""
    print("=" * 60)
    print("内存优化效果测试")
    print("=" * 60)
    
    # 记录初始内存使用
    initial_memory = get_memory_usage()
    print(f"初始内存使用: RSS={initial_memory['rss']:.2f}MB, VMS={initial_memory['vms']:.2f}MB, 百分比={initial_memory['percent']:.2f}%")
    
    # 初始化特征提取器
    print("\n初始化特征提取器...")
    extractor = FeatureExtraction()
    memory_after_extractor = get_memory_usage()
    print(f"特征提取器初始化后: RSS={memory_after_extractor['rss']:.2f}MB, VMS={memory_after_extractor['vms']:.2f}MB, 百分比={memory_after_extractor['percent']:.2f}%")
    print(f"内存增长: {memory_after_extractor['rss'] - initial_memory['rss']:.2f}MB")
    
    # 初始化分类器
    print("\n初始化分类器...")
    index_path = 'role_index.faiss'
    classifier = Classification(index_path)
    memory_after_classifier = get_memory_usage()
    print(f"分类器初始化后: RSS={memory_after_classifier['rss']:.2f}MB, VMS={memory_after_classifier['vms']:.2f}MB, 百分比={memory_after_classifier['percent']:.2f}%")
    print(f"内存增长: {memory_after_classifier['rss'] - memory_after_extractor['rss']:.2f}MB")
    
    # 测试多次分类，观察内存使用情况
    print("\n开始测试多次分类...")
    test_image_path = 'data/train/日奈/日奈_000.jpg'
    
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
        return
    
    img = Image.open(test_image_path).convert('RGB')
    
    # 测试10次分类
    for i in range(10):
        print(f"\n第 {i+1} 次分类...")
        
        # 提取特征
        feature = extractor.extract_features(img)
        
        # 分类
        predicted_role, similarity = classifier.classify(feature, top_k=1)
        
        # 记录内存使用
        memory = get_memory_usage()
        print(f"分类结果: {predicted_role}, 相似度: {similarity:.4f}")
        print(f"内存使用: RSS={memory['rss']:.2f}MB, VMS={memory['vms']:.2f}MB, 百分比={memory['percent']:.2f}%")
        
        # 检查内存是否稳定
        if i > 0:
            memory_diff = memory['rss'] - memory_after_classifier['rss']
            print(f"相对初始分类器内存: {memory_diff:.2f}MB")
            
            if abs(memory_diff) < 10:  # 内存增长小于10MB，认为稳定
                print("✓ 内存使用稳定")
            else:
                print("✗ 内存使用不稳定，可能存在内存泄漏")
        
        time.sleep(0.5)  # 等待0.5秒
    
    # 最终内存使用
    final_memory = get_memory_usage()
    print(f"\n最终内存使用: RSS={final_memory['rss']:.2f}MB, VMS={final_memory['vms']:.2f}MB, 百分比={final_memory['percent']:.2f}%")
    print(f"总内存增长: {final_memory['rss'] - initial_memory['rss']:.2f}MB")
    
    # 清理
    print("\n清理资源...")
    del extractor
    del classifier
    gc.collect()
    
    memory_after_cleanup = get_memory_usage()
    print(f"清理后内存使用: RSS={memory_after_cleanup['rss']:.2f}MB, VMS={memory_after_cleanup['vms']:.2f}MB, 百分比={memory_after_cleanup['percent']:.2f}%")
    print(f"内存释放: {final_memory['rss'] - memory_after_cleanup['rss']:.2f}MB")
    
    print("\n" + "=" * 60)
    print("内存优化测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_memory_optimization()
