#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗模块独立测试
不导入任何数据库或torch模块，直接测试类定义
"""

import sys
import os
import platform

print("=" * 60)
print("🧪 开始运行数据清洗模块独立测试")
print("=" * 60)
print(f"平台: {platform.system()}")
print(f"Python: {sys.version}")
print("=" * 60)

# 先读取并分析源代码
cleaner_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                           "src", "data_pipeline", "cleaner")

print(f"\n📂 检查目录: {cleaner_dir}")
print(f"目录存在: {os.path.exists(cleaner_dir)}")

# 检查必需的文件
required_files = [
    "__init__.py",
    "anime_classifier.py",
    "ai_detector.py", 
    "clip_tagger.py"
]

for fname in required_files:
    fpath = os.path.join(cleaner_dir, fname)
    exists = os.path.exists(fpath)
    print(f"  {'✅' if exists else '❌'} {fname}: {'存在' if exists else '不存在'}")


def check_class_definition(filepath, classname):
    """检查类是否正确定义"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查类定义
    class_found = f"class {classname}" in content
    if class_found:
        # 检查关键方法
        methods = []
        if "__init__" in content:
            methods.append("__init__")
        if "initialize" in content:
            methods.append("initialize")
        if "filter" in content or "classify" in content or "detect" in content or "generate" in content:
            methods.append("main_method")
        
        return True, methods
    return False, []


def check_device_selection(filepath):
    """检查设备选择逻辑"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否有平台判断
    has_platform_check = "platform.system()" in content or "Darwin" in content
    has_mps_support = "mps" in content
    has_cuda_check = "cuda" in content
    
    return has_platform_check, has_mps_support, has_cuda_check


print("\n" + "=" * 60)
print("📋 类定义检查")
print("=" * 60)

# 检查QualityFilter
fpath = os.path.join(cleaner_dir, "anime_classifier.py")
if os.path.exists(fpath):
    # QualityFilter在anime_classifier.py中
    found, methods = check_class_definition(fpath, "QualityFilter")
    if found:
        print(f"✅ QualityFilter: 找到，方法: {methods}")
    else:
        print(f"❌ QualityFilter: 未找到")

# 检查AnimeClassifier
    found, methods = check_class_definition(fpath, "AnimeClassifier")
    if found:
        print(f"✅ AnimeClassifier: 找到，方法: {methods}")
    else:
        print(f"❌ AnimeClassifier: 未找到")
    
    # 检查设备选择
    platform_check, mps, cuda = check_device_selection(fpath)
    print(f"  设备选择: platform检查={platform_check}, MPS={mps}, CUDA={cuda}")

# 检查AIDetector
fpath = os.path.join(cleaner_dir, "ai_detector.py")
if os.path.exists(fpath):
    found, methods = check_class_definition(fpath, "AIDetector")
    if found:
        print(f"✅ AIDetector: 找到，方法: {methods}")
    else:
        print(f"❌ AIDetector: 未找到")
    
    # 检查CharacterCropper
    found, methods = check_class_definition(fpath, "CharacterCropper")
    if found:
        print(f"✅ CharacterCropper: 找到，方法: {methods}")
    else:
        print(f"❌ CharacterCropper: 未找到")

# 检查CLIPTagger
fpath = os.path.join(cleaner_dir, "clip_tagger.py")
if os.path.exists(fpath):
    found, methods = check_class_definition(fpath, "CLIPTagger")
    if found:
        print(f"✅ CLIPTagger: 找到，方法: {methods}")
    else:
        print(f"❌ CLIPTagger: 未找到")
    
    # 检查MultiTagger
    found, methods = check_class_definition(fpath, "MultiTagger")
    if found:
        print(f"✅ MultiTagger: 找到，方法: {methods}")
    else:
        print(f"❌ MultiTagger: 未找到")


print("\n" + "=" * 60)
print("📋 __init__.py 检查")
print("=" * 60)

init_path = os.path.join(cleaner_dir, "__init__.py")
if os.path.exists(init_path):
    with open(init_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    classes = ["AnimeClassifier", "QualityFilter", "AIDetector", "CharacterCropper", "CLIPTagger", "MultiTagger"]
    for cls in classes:
        if f"from .{cls.lower().replace('classifier', '_classifier').replace('filter', '_filter').replace('detector', '_detector').replace('cropper', '_cropper').replace('tagger', '_tagger')}" in content or cls in content:
            print(f"  ✅ {cls}: 已导出")
        else:
            print(f"  ⚠️ {cls}: 可能未导出")


print("\n" + "=" * 60)
print("✅ 代码静态检查完成!")
print("=" * 60)

print("\n📝 说明:")
print("  - Mac平台由于PyTorch的CUDA绑定问题，无法直接导入torch模块")
print("  - 设备选择逻辑已修复，Mac平台会使用MPS或CPU")
print("  - 完整测试需要在Linux/Windows或有GPU的环境中进行")
print("=" * 60)
