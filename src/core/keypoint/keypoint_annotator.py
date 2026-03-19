#!/usr/bin/env python3
"""
关键点标注工具
用于为训练数据生成关键点标注
"""

import os
import sys
import json
import argparse
from tqdm import tqdm
from PIL import Image

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.keypoint.mediapipe_keypoint_detector import MediaPipeKeypointDetector

class KeypointAnnotator:
    """关键点标注器"""
    
    def __init__(self, data_dir, output_dir, visualize=False):
        """初始化标注器
        
        Args:
            data_dir: 训练数据目录
            output_dir: 标注输出目录
            visualize: 是否生成可视化标注结果
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.visualize = visualize
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        if visualize:
            os.makedirs(os.path.join(output_dir, 'visualization'), exist_ok=True)
        
        # 初始化关键点检测器
        self.detector = MediaPipeKeypointDetector()
    
    def process_images(self):
        """处理所有图像并生成标注"""
        # 遍历数据目录
        for root, dirs, files in os.walk(self.data_dir):
            # 跳过隐藏目录
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            # 处理当前目录下的图像
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            
            if images:
                # 获取相对路径作为标注文件的键
                rel_path = os.path.relpath(root, self.data_dir)
                if rel_path == '.':
                    rel_path = ''
                
                # 生成标注文件路径
                annotation_file = os.path.join(self.output_dir, f"{rel_path.replace('/', '_')}_keypoints.json")
                
                # 处理图像
                annotations = self._process_directory(root, images, rel_path)
                
                # 保存标注
                if annotations:
                    self._save_annotations(annotation_file, annotations)
    
    def _process_directory(self, directory, images, rel_path):
        """处理单个目录下的图像
        
        Args:
            directory: 目录路径
            images: 图像文件列表
            rel_path: 相对路径
            
        Returns:
            dict: 标注数据
        """
        annotations = {}
        
        for image_file in tqdm(images, desc=f"处理 {rel_path or '根目录'}"):
            image_path = os.path.join(directory, image_file)
            
            try:
                # 加载图像
                image = Image.open(image_path).convert('RGB')
                
                # 检测关键点
                keypoints = self.detector.detect_keypoints(image)
                
                # 生成标注数据
                # 确保所有数据都是JSON可序列化的类型
                def convert_to_serializable(obj):
                    if isinstance(obj, (list, tuple)):
                        return [convert_to_serializable(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (int, float, str, bool, type(None))):
                        return obj
                    else:
                        return str(obj)
                
                serializable_keypoints = convert_to_serializable(keypoints)
                
                annotation = {
                    'image_path': os.path.join(rel_path, image_file),
                    'keypoints': serializable_keypoints
                }
                
                # 可视化
                if self.visualize:
                    self._visualize_keypoints(image, keypoints, rel_path, image_file)
                
                # 添加到标注字典
                annotations[image_file] = annotation
                
            except Exception as e:
                print(f"处理图像 {image_file} 失败: {e}")
        
        return annotations
    
    def _visualize_keypoints(self, image, keypoints, rel_path, image_file):
        """可视化关键点
        
        Args:
            image: PIL图像对象
            keypoints: 关键点数据
            rel_path: 相对路径
            image_file: 图像文件名
        """
        try:
            # 绘制关键点
            annotated_image = self.detector.draw_keypoints(image, keypoints)
            
            # 创建输出目录
            viz_dir = os.path.join(self.output_dir, 'visualization', rel_path)
            os.makedirs(viz_dir, exist_ok=True)
            
            # 保存可视化结果
            output_path = os.path.join(viz_dir, image_file)
            annotated_image.save(output_path)
        except Exception as e:
            print(f"可视化失败: {e}")
    
    def _save_annotations(self, annotation_file, annotations):
        """保存标注文件
        
        Args:
            annotation_file: 标注文件路径
            annotations: 标注数据
        """
        try:
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)
            print(f"标注文件已保存: {annotation_file}")
        except Exception as e:
            print(f"保存标注文件失败: {e}")
    
    def close(self):
        """关闭检测器"""
        self.detector.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='关键点标注工具')
    parser.add_argument('--data-dir', type=str, default='data/downloaded_images', help='训练数据目录')
    parser.add_argument('--output-dir', type=str, default='data/keypoint_annotations', help='标注输出目录')
    parser.add_argument('--visualize', action='store_true', default=True, help='生成可视化标注结果')
    
    args = parser.parse_args()
    
    # 创建标注器
    annotator = KeypointAnnotator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        visualize=args.visualize
    )
    
    try:
        # 处理图像
        annotator.process_images()
        print("标注完成！")
    finally:
        # 关闭检测器
        annotator.close()


if __name__ == "__main__":
    main()
