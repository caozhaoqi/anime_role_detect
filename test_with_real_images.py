#!/usr/bin/env python3
"""
使用真实图片测试关键点检测功能
"""

import os
import sys
from PIL import Image
from src.core.keypoint.mediapipe_keypoint_detector import MediaPipeKeypointDetector

def test_with_real_images():
    """使用真实图片测试关键点检测功能"""
    print("\n=== 使用真实图片测试关键点检测功能 ===")
    
    # 初始化关键点检测器
    detector = MediaPipeKeypointDetector()
    
    # 图片目录
    image_dir = "data/downloaded_images"
    
    # 检查目录是否存在
    if not os.path.exists(image_dir):
        print(f"目录不存在: {image_dir}")
        print("使用train目录下的图片进行测试")
        image_dir = "data/train"
    
    print(f"测试目录: {image_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(image_dir):
        print(f"目录不存在: {image_dir}")
        return
    
    # 遍历所有子目录
    for character in os.listdir(image_dir):
        character_dir = os.path.join(image_dir, character)
        if os.path.isdir(character_dir):
            print(f"\n测试角色: {character}")
            
            # 遍历所有图片
            for img_name in os.listdir(character_dir):
                img_path = os.path.join(character_dir, img_name)
                
                # 只处理图片文件
                if os.path.isfile(img_path) and img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.svg')):
                    print(f"测试图片: {img_name}")
                    
                    try:
                        # 检测关键点
                        keypoints = detector.detect_keypoints(img_path)
                        
                        # 打印检测结果
                        print(f"  面部检测: {keypoints['face'] is not None}")
                        print(f"  手部检测: {keypoints['hands'] is not None}")
                        print(f"  姿态检测: {keypoints['pose'] is not None}")
                        
                        # 加载图片进行绘制
                        if img_name.lower().endswith('.svg'):
                            image = detector._load_svg_image(img_path)
                        else:
                            image = Image.open(img_path).convert('RGB')
                        
                        # 可视化关键点
                        annotated_image = detector.draw_keypoints(image, keypoints)
                        
                        # 保存标注结果
                        output_dir = os.path.join("test_results", character)
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"annotated_{img_name}")
                        # 确保输出文件格式正确
                        if img_name.lower().endswith('.svg'):
                            output_path = output_path.replace('.svg', '.png')
                        annotated_image.save(output_path)
                        print(f"  标注结果已保存: {output_path}")
                        
                    except Exception as e:
                        print(f"  处理失败: {e}")
    
    # 关闭检测器
    detector.close()
    print("\n测试完成！")

if __name__ == "__main__":
    test_with_real_images()
