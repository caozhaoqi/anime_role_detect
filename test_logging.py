#!/usr/bin/env python3
"""
测试训练脚本的日志功能
"""

import os
import sys
import shutil
import tempfile
from PIL import Image, ImageDraw

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

def create_test_data(output_dir):
    """创建测试数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建两个测试角色
    characters = ['测试角色A', '测试角色B']
    
    for char in characters:
        char_dir = os.path.join(output_dir, char)
        os.makedirs(char_dir, exist_ok=True)
        
        # 为每个角色创建5张测试图像
        for i in range(5):
            img = Image.new('RGB', (300, 300), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((100, 150), f"{char} {i+1}", fill='black')
            # 绘制一个简单的人脸
            draw.ellipse([(100, 80), (200, 180)], fill='yellow')
            # 绘制眼睛
            draw.ellipse([(120, 110), (140, 130)], fill='black')
            draw.ellipse([(160, 110), (180, 130)], fill='black')
            img.save(os.path.join(char_dir, f'test_{i+1}.jpg'))
    
    print(f"测试数据已创建: {output_dir}")

def create_test_annotations(output_dir):
    """创建测试关键点标注"""
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建简单的标注文件
    annotations_a = {}
    annotations_b = {}
    
    for i in range(5):
        annotations_a[f'test_{i+1}.jpg'] = {
            'image_path': f'测试角色A/test_{i+1}.jpg',
            'keypoints': {
                'face': [{
                    'bbox': {'x1': 100, 'y1': 80, 'x2': 200, 'y2': 180},
                    'keypoints': [
                        {'x': 130, 'y': 120, 'type': 'eye'},
                        {'x': 170, 'y': 120, 'type': 'eye'},
                        {'x': 150, 'y': 130, 'type': 'face_center'}
                    ]
                }],
                'hands': None,
                'pose': None
            }
        }
        
        annotations_b[f'test_{i+1}.jpg'] = {
            'image_path': f'测试角色B/test_{i+1}.jpg',
            'keypoints': {
                'face': [{
                    'bbox': {'x1': 100, 'y1': 80, 'x2': 200, 'y2': 180},
                    'keypoints': [
                        {'x': 130, 'y': 120, 'type': 'eye'},
                        {'x': 170, 'y': 120, 'type': 'eye'},
                        {'x': 150, 'y': 130, 'type': 'face_center'}
                    ]
                }],
                'hands': None,
                'pose': None
            }
        }
    
    with open(os.path.join(output_dir, '测试角色A_keypoints.json'), 'w', encoding='utf-8') as f:
        json.dump(annotations_a, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, '测试角色B_keypoints.json'), 'w', encoding='utf-8') as f:
        json.dump(annotations_b, f, ensure_ascii=False, indent=2)
    
    print(f"测试标注已创建: {output_dir}")

def main():
    """主函数"""
    # 创建临时目录
    test_dir = tempfile.mkdtemp(prefix='anime_test_')
    data_dir = os.path.join(test_dir, 'train')
    annot_dir = os.path.join(test_dir, 'keypoint_annotations')
    output_dir = os.path.join(test_dir, 'models')
    
    try:
        print("="*80)
        print("创建测试数据...")
        print("="*80)
        create_test_data(data_dir)
        create_test_annotations(annot_dir)
        
        print("\n" + "="*80)
        print("测试数据结构:")
        print("="*80)
        print(f"数据目录: {data_dir}")
        print(f"标注目录: {annot_dir}")
        print(f"输出目录: {output_dir}")
        
        print("\n" + "="*80)
        print("测试数据创建完成！")
        print("="*80)
        print(f"\n现在可以运行以下命令测试训练脚本:")
        print(f"python3 arona/training/train_with_keypoints.py --data-dir {data_dir} --keypoint-annotations-dir {annot_dir} --output-dir {output_dir} --epochs 2 --batch-size 2")
        print(f"\n临时目录: {test_dir}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        # 清理临时目录
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
