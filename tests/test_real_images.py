#!/usr/bin/env python3
"""
使用真实图片测试分类功能
"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.backend.services.classification_service import classify_image


def test_real_images():
    """使用真实图片测试分类功能"""
    print("=== 使用真实图片测试分类功能 ===")
    
    # 测试图片目录
    test_dir = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/downloaded_images"
    
    # 获取测试图片
    test_images = []
    
    # 遍历日奈和阿罗娜的图片
    for role in ["ri4nai4", "a1luo2na4"]:
        role_dir = os.path.join(test_dir, role)
        if os.path.exists(role_dir):
            for img_file in os.listdir(role_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    test_images.append((os.path.join(role_dir, img_file), role))
    
    print(f"找到 {len(test_images)} 张测试图片")
    print()
    
    # 测试配置
    test_configs = [
        ("default", False, False, False, False, "默认模型 (CLIP + FAISS)"),
    ]
    
    # 每个模型测试前几张图片
    max_images_per_model = 5
    
    for model_name, use_coreml, use_model, use_deepdanbooru, use_attributes, description in test_configs:
        print(f"\n=== 测试: {description} ===")
        
        correct_count = 0
        total_count = 0
        
        for i, (img_path, expected_role) in enumerate(test_images[:max_images_per_model]):
            print(f"\n测试第 {i+1} 张图片: {os.path.basename(img_path)}")
            print(f"预期角色: {expected_role}")
            
            start_time = time.time()
            
            try:
                role, similarity, boxes, mode, attributes, text_detections = classify_image(
                    img_path,
                    use_coreml=use_coreml,
                    use_model=use_model,
                    use_deepdanbooru=use_deepdanbooru,
                    use_attributes=use_attributes,
                    model_name=model_name
                )
                
                elapsed_time = time.time() - start_time
                print(f"分类结果: 角色={role}, 相似度={similarity:.4f}")
                print(f"使用模式: {mode}")
                print(f"处理时间: {elapsed_time:.2f}秒")
                
                # 验证结果
                # 拼音到中文的映射
                pinyin_to_chinese = {
                    "ri4nai4": "日奈",
                    "a1luo2na4": "阿罗娜"
                }
                expected_chinese = pinyin_to_chinese.get(expected_role, expected_role)
                
                if role == expected_chinese:
                    print("✅ 分类正确!")
                    correct_count += 1
                else:
                    print("❌ 分类错误!")
                    
                total_count += 1
                
            except Exception as e:
                print(f"❌ 分类失败: {e}")
                total_count += 1
        
        # 计算准确率
        if total_count > 0:
            accuracy = (correct_count / total_count) * 100
            print(f"\n=== 测试结果 ===")
            print(f"总测试数: {total_count}")
            print(f"正确数: {correct_count}")
            print(f"准确率: {accuracy:.1f}%")
        
        print("-" * 60)


if __name__ == "__main__":
    test_real_images()
