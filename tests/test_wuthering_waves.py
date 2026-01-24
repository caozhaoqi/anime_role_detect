#!/usr/bin/env python3
"""
测试系统对鸣潮角色图片的识别效果
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.preprocessing.preprocessing import Preprocessing
from src.core.classification.classification import Classification


def test_wuthering_waves_processing():
    """测试鸣潮角色处理"""
    print("测试鸣潮角色处理...")
    
    # 初始化各个模块
    try:
        preprocessor = Preprocessing()
        print("预处理模块初始化成功")
    except Exception as e:
        print(f"预处理模块初始化失败: {e}")
        print("跳过处理测试")
        return
    
    try:
        classifier = Classification()
        print("分类模块初始化成功")
    except Exception as e:
        print(f"分类模块初始化失败: {e}")
        classifier = None
    
    # 跳过标签模块，避免tensorflow依赖问题
    tagger = None
    
    # 测试鸣潮角色图片
    test_dir = "tests/test_images/single_character"
    wuthering_waves_dirs = [d for d in os.listdir(test_dir) if "鸣潮" in d]
    
    for char_dir in wuthering_waves_dirs:
        char_path = os.path.join(test_dir, char_dir)
        print(f"\n处理角色: {char_dir}")
        
        image_files = [f for f in os.listdir(char_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in image_files:
            img_path = os.path.join(char_path, img_file)
            print(f"处理图片: {img_file}")
            
            try:
                # 预处理图片
                normalized_img, boxes = preprocessor.process(img_path)
                print(f"  检测到 {len(boxes)} 个角色")
                print(f"  预处理后的图像大小: {normalized_img.size}")
                
                # 保存预处理后的图像
                output_path = os.path.join("tests/test_results", f"processed_{char_dir}_{os.path.basename(img_file)}")
                normalized_img.save(output_path)
                print(f"  预处理后的图像已保存为: {output_path}")
                
                # 尝试分类
                if classifier:
                    try:
                        class_result = classifier.classify(normalized_img)
                        print(f"  分类结果: {class_result}")
                    except Exception as e:
                        print(f"  分类失败: {e}")
                
                # 跳过标签生成，避免tensorflow依赖问题
                # if tagger:
                #     try:
                #         tags = tagger.generate_tags(normalized_img)
                #         print(f"  生成标签: {tags}")
                #     except Exception as e:
                #         print(f"  标签生成失败: {e}")
                        
            except Exception as e:
                print(f"  处理失败: {e}")


def main():
    """主函数"""
    # 确保输出目录存在
    if not os.path.exists("tests/test_results"):
        os.makedirs("tests/test_results")
    
    # 测试鸣潮角色处理
    test_wuthering_waves_processing()


if __name__ == "__main__":
    main()
