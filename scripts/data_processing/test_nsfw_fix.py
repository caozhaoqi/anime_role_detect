#!/usr/bin/env python3
"""测试NSFW检测功能是否使用模型检测"""

import os
# 在导入任何库之前设置环境变量
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.nsfw_detector import detect_nsfw
from pathlib import Path

# 测试图片路径
DATASET_PATH = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset')

def test_nsfw_detection():
    """测试NSFW检测功能"""
    print("🔍 测试NSFW检测功能...")
    
    # 找到测试图片
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = []
    
    for char_dir in DATASET_PATH.iterdir():
        if not char_dir.is_dir():
            continue
        
        for ext in image_extensions:
            image_files.extend(char_dir.glob(f"*{ext}"))
        
        if image_files:
            break
    
    if not image_files:
        print("❌ 未找到测试图片")
        return
    
    # 取前3张图片测试
    test_images = image_files[:3]
    
    print(f"\n📁 测试图片: {len(test_images)} 张")
    
    for i, img_path in enumerate(test_images):
        print(f"\n--- 图片 {i+1}: {img_path.name} ---")
        
        result = detect_nsfw(str(img_path))
        
        if result is None:
            print("❌ 检测失败")
            continue
        
        print(f"   检测方法: {result.get('method', '未知')}")
        print(f"   是否NSFW: {result.get('is_nsfw', False)}")
        print(f"   NSFW得分: {result.get('nsfw_score', 0):.4f}")
        print(f"   皮肤比例: {result.get('skin_ratio', 0):.4f}")
        
        if 'details' in result:
            details = result['details']
            print(f"   分类详情:")
            for label, score in details.items():
                print(f"     - {label}: {score:.4f}")
    
    print("\n✅ 测试完成!")
    
    # 检查是否使用了模型检测
    results = [detect_nsfw(str(img)) for img in test_images]
    methods = [r.get('method') for r in results if r]
    
    print(f"\n📊 检测方法统计:")
    print(f"   opencv_based: {methods.count('opencv_based')}")
    print(f"   transformers: {methods.count('transformers')}")
    print(f"   tensorflow_serving: {methods.count('tensorflow_serving')}")
    print(f"   rule_based: {methods.count('rule_based')}")
    print(f"   error: {methods.count('error')}")
    print(f"   default: {methods.count('default')}")
    
    # 判断是否成功使用模型
    model_used = methods.count('opencv_based') + methods.count('transformers') + methods.count('tensorflow_serving')
    if model_used > 0:
        print(f"\n🎉 成功! NSFW检测已使用模型检测")
    else:
        print(f"\n⚠️  警告: NSFW检测未使用模型检测，回退到规则检测")


if __name__ == "__main__":
    test_nsfw_detection()