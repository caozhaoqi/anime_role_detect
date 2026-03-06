#!/usr/bin/env python3
"""
测试系统对采集数据的处理效果
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.preprocessing.preprocessing import Preprocessing


def test_single_character_processing():
    """测试单角色处理"""
    print("测试单角色处理...")
    
    # 初始化预处理模块
    preprocessor = Preprocessing()
    
    # 测试采集到的单角色图片
    test_dir = "tests/test_images/single_character/rem"
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        print(f"处理图片: {img_path}")
        
        try:
            # 处理图片
            normalized_img, boxes = preprocessor.process(img_path)
            print(f"检测到 {len(boxes)} 个角色")
            print(f"预处理后的图像大小: {normalized_img.size}")
            
            # 保存处理后的图像
            output_path = os.path.join("tests/test_results", f"processed_{os.path.basename(img_file)}")
            normalized_img.save(output_path)
            print(f"处理后的图像已保存为: {output_path}")
            print()
        except Exception as e:
            print(f"处理失败: {e}")
            print()


def test_multiple_characters_processing():
    """测试多角色处理"""
    print("测试多角色处理...")
    
    # 初始化预处理模块
    preprocessor = Preprocessing()
    
    # 测试采集到的多角色图片
    test_dir = "tests/test_images/multiple_characters"
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        print(f"处理图片: {img_path}")
        
        try:
            # 处理图片
            processed_characters = preprocessor.process_multiple_characters(img_path)
            print(f"检测到 {len(processed_characters)} 个角色")
            
            # 保存处理后的图像
            for i, character in enumerate(processed_characters):
                output_path = os.path.join("tests/test_results", f"processed_{os.path.splitext(os.path.basename(img_file))[0]}_char{i+1}.jpg")
                character['image'].save(output_path)
                print(f"角色 {i+1} 处理后的图像已保存为: {output_path}")
            print()
        except Exception as e:
            print(f"处理失败: {e}")
            print()


def main():
    """主函数"""
    # 确保输出目录存在
    if not os.path.exists("tests/test_results"):
        os.makedirs("tests/test_results")
    
    # 测试单角色处理
    test_single_character_processing()
    
    # 测试多角色处理
    test_multiple_characters_processing()


if __name__ == "__main__":
    main()
