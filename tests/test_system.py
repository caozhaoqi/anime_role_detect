#!/usr/bin/env python3
"""
测试整个系统的功能
"""

import os
import sys
import argparse
from PIL import Image
import cv2
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.keypoint.mediapipe_keypoint_detector import MediaPipeKeypointDetector
from src.core.classification.efficientnet_inference import EfficientNetInference

def test_keypoint_detection():
    """测试关键点检测功能"""
    print("\n=== 测试关键点检测功能 ===")
    
    # 初始化关键点检测器
    detector = MediaPipeKeypointDetector()
    
    # 测试图像路径
    test_image_path = "test_image.jpg"
    
    if not os.path.exists(test_image_path):
        # 创建测试图像
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 绘制更详细的人脸（使用肤色）
        cv2.circle(img, (320, 240), 50, (0, 200, 255), -1)  # 人脸
        cv2.circle(img, (300, 220), 10, (0, 0, 0), -1)  # 左眼
        cv2.circle(img, (340, 220), 10, (0, 0, 0), -1)  # 右眼
        cv2.ellipse(img, (320, 260), (20, 10), 0, 0, 360, (0, 0, 0), 2)  # 嘴巴
        
        # 绘制更详细的手部（使用肤色）
        cv2.circle(img, (100, 300), 25, (0, 200, 255), -1)  # 左手
        cv2.circle(img, (540, 300), 25, (0, 200, 255), -1)  # 右手
        
        # 绘制更详细的身体（使用肤色）
        cv2.rectangle(img, (300, 300), (340, 400), (0, 200, 255), -1)  # 身体
        cv2.rectangle(img, (280, 400), (310, 450), (0, 200, 255), -1)  # 左腿
        cv2.rectangle(img, (330, 400), (360, 450), (0, 200, 255), -1)  # 右腿
        
        # 保存测试图像
        cv2.imwrite(test_image_path, img)
        print(f"测试图像已创建: {test_image_path}")
    
    # 加载测试图像
    image = Image.open(test_image_path).convert('RGB')
    
    # 检测关键点
    print("检测关键点...")
    keypoints = detector.detect_keypoints(image)
    
    # 打印检测结果
    print(f"面部检测: {keypoints['face'] is not None}")
    print(f"手部检测: {keypoints['hands'] is not None}")
    print(f"姿态检测: {keypoints['pose'] is not None}")
    
    # 可视化关键点
    print("可视化关键点...")
    annotated_image = detector.draw_keypoints(image, keypoints)
    annotated_image.save("test_keypoint_annotated.jpg")
    print(f"标注结果已保存: test_keypoint_annotated.jpg")
    
    # 关闭检测器
    detector.close()
    print("关键点检测测试完成！")

def test_model_inference():
    """测试模型推理功能"""
    print("\n=== 测试模型推理功能 ===")
    
    # 初始化模型推理器
    infer = EfficientNetInference()
    
    # 测试图像路径
    test_image_path = "test_image.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"测试图像 {test_image_path} 不存在，请先运行关键点检测测试")
        return
    
    # 测试预测（带关键点检测）
    print("测试预测（带关键点检测）...")
    best_role, best_score, results, keypoints = infer.predict(test_image_path, return_keypoints=True)
    
    # 打印预测结果
    print(f"预测结果: {best_role} (相似度: {best_score:.4f})")
    print(f"关键点检测: {keypoints is not None}")
    if keypoints:
        print(f"面部检测: {keypoints['face'] is not None}")
        print(f"手部检测: {keypoints['hands'] is not None}")
        print(f"姿态检测: {keypoints['pose'] is not None}")
    
    # 关闭推理器
    infer.close()
    print("模型推理测试完成！")

def test_keypoint_annotator():
    """测试关键点标注工具"""
    print("\n=== 测试关键点标注工具 ===")
    
    # 测试图像目录
    test_data_dir = "test_data"
    os.makedirs(test_data_dir, exist_ok=True)
    
    # 创建测试图像
    test_image_path = os.path.join(test_data_dir, "test_image.jpg")
    if not os.path.exists(test_image_path):
        # 创建测试图像
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 绘制更详细的人脸（使用肤色）
        cv2.circle(img, (320, 240), 50, (0, 200, 255), -1)  # 人脸
        cv2.circle(img, (300, 220), 10, (0, 0, 0), -1)  # 左眼
        cv2.circle(img, (340, 220), 10, (0, 0, 0), -1)  # 右眼
        cv2.ellipse(img, (320, 260), (20, 10), 0, 0, 360, (0, 0, 0), 2)  # 嘴巴
        
        # 绘制更详细的手部（使用肤色）
        cv2.circle(img, (100, 300), 25, (0, 200, 255), -1)  # 左手
        cv2.circle(img, (540, 300), 25, (0, 200, 255), -1)  # 右手
        
        # 绘制更详细的身体（使用肤色）
        cv2.rectangle(img, (300, 300), (340, 400), (0, 200, 255), -1)  # 身体
        cv2.rectangle(img, (280, 400), (310, 450), (0, 200, 255), -1)  # 左腿
        cv2.rectangle(img, (330, 400), (360, 450), (0, 200, 255), -1)  # 右腿
        
        # 保存测试图像
        cv2.imwrite(test_image_path, img)
        print(f"测试图像已创建: {test_image_path}")
    
    # 运行关键点标注工具
    print("运行关键点标注工具...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "arona/annotation/keypoint_annotator.py", 
         "--data-dir", test_data_dir, 
         "--output-dir", "test_annotations"],
        capture_output=True,
        text=True
    )
    
    print("标注工具输出:")
    print(result.stdout)
    if result.stderr:
        print("错误:")
        print(result.stderr)
    
    print("关键点标注测试完成！")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试整个系统的功能')
    parser.add_argument('--test', type=str, default='all', 
                       choices=['all', 'keypoint', 'inference', 'annotator'],
                       help='测试类型')
    
    args = parser.parse_args()
    
    print("开始测试整个系统的功能...")
    
    if args.test == 'all' or args.test == 'keypoint':
        test_keypoint_detection()
    
    if args.test == 'all' or args.test == 'inference':
        test_model_inference()
    
    if args.test == 'all' or args.test == 'annotator':
        test_keypoint_annotator()
    
    print("\n所有测试完成！")


if __name__ == "__main__":
    main()
