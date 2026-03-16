#!/usr/bin/env python3
"""
测试关键点检测功能
"""

import os
from PIL import Image
from src.core.keypoint.mediapipe_keypoint_detector import MediaPipeKeypointDetector

# 测试图像路径
TEST_IMAGE = "test_image.jpg"

# 如果测试图像不存在，使用示例图像
if not os.path.exists(TEST_IMAGE):
    # 创建一个简单的测试图像
    from PIL import ImageDraw, ImageFont
    img = Image.new('RGB', (400, 400), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((150, 180), "Test Image", fill='black')
    # 绘制一个简单的人脸
    draw.ellipse([(150, 100), (250, 200)], fill='yellow')
    # 绘制眼睛
    draw.ellipse([(170, 130), (190, 150)], fill='black')
    draw.ellipse([(210, 130), (230, 150)], fill='black')
    # 绘制嘴巴
    draw.line([(180, 170), (220, 170)], fill='black', width=2)
    # 绘制手
    draw.ellipse([(100, 250), (150, 300)], fill='yellow')
    draw.ellipse([(250, 250), (300, 300)], fill='yellow')
    img.save(TEST_IMAGE)
    print(f"创建了测试图像: {TEST_IMAGE}")

# 初始化检测器
detector = MediaPipeKeypointDetector()

print("开始测试关键点检测...")

try:
    # 加载测试图像
    image = Image.open(TEST_IMAGE)
    print(f"加载图像成功: {TEST_IMAGE}")
    
    # 检测关键点
    keypoints = detector.detect_keypoints(image)
    print("检测结果:")
    print(f"面部检测: {'成功' if keypoints['face'] else '失败'}")
    if keypoints['face']:
        print(f"  检测到 {len(keypoints['face'])} 个面部")
        for i, face in enumerate(keypoints['face']):
            print(f"  面部 {i+1} 边界框: {face['bbox']}")
            print(f"  面部 {i+1} 关键点数量: {len(face['keypoints'])}")
    
    print(f"手部检测: {'成功' if keypoints['hands'] else '失败'}")
    if keypoints['hands']:
        print(f"  检测到 {len(keypoints['hands'])} 只手")
        for i, hand in enumerate(keypoints['hands']):
            print(f"  手 {i+1} 边界框: {hand['bbox']}")
    
    print(f"姿态检测: {'成功' if keypoints['pose'] else '失败'}")
    if keypoints['pose']:
        print("  检测到身体姿态")
        print(f"  身体边界框: {keypoints['pose']['bbox']}")
        print(f"  身体关键点数量: {len(keypoints['pose']['keypoints'])}")
    
    # 绘制关键点
    annotated_image = detector.draw_keypoints(image, keypoints)
    output_path = "annotated_test_image.jpg"
    annotated_image.save(output_path)
    print(f"标注图像已保存为: {output_path}")
    print("测试完成！")
    
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()
finally:
    detector.close()
