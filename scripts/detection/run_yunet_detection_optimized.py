#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YuNet 人脸检测优化脚本

优化要点：
1. 使用 cv2.FaceDetectorYN (YuNet) 替代 Haar 级联
2. 降低置信度阈值提高召回率
3. 自动切割多角色图片
4. 输入尺寸对齐优化
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"📦 OpenCV 版本: {cv2.__version__}")


class YuNetFaceDetector:
    """
    YuNet 人脸检测器
    """
    
    def __init__(self, score_threshold=0.5, nms_threshold=0.3, top_k=5000):
        """
        初始化 YuNet 检测器
        
        Args:
            score_threshold: 置信度阈值，建议动漫场景使用 0.5-0.6
            nms_threshold: NMS阈值
            top_k: 最大检测数量
        """
        self.detector = None
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._initialize_detector()
    
    def _initialize_detector(self):
        """
        初始化 YuNet 检测器（兼容 OpenCV 4.13+）
        """
        try:
            # 尝试获取 OpenCV 的 YuNet 模型路径
            model_path = self._find_model_path()
            
            if not model_path:
                print("❌ 未找到 YuNet 模型文件，将回退到 Haar 级联")
                return
            
            print(f"🔧 加载模型: {model_path}")
            
            # OpenCV 4.13+ 使用新的 API
            if cv2.__version__ >= "4.13.0":
                # 新 API 需要手动读取模型
                self.detector = cv2.FaceDetectorYN.create(
                    model=model_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=self.top_k
                )
            else:
                # 旧 API
                self.detector = cv2.FaceDetectorYN.create(
                    model_path=model_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=self.top_k
                )
            
            if self.detector is not None:
                print("✅ YuNet 检测器初始化成功")
                print(f"   - 置信度阈值: {self.score_threshold}")
                print(f"   - NMS阈值: {self.nms_threshold}")
            else:
                print("❌ YuNet 检测器创建失败，将回退到 Haar 级联")
                
        except Exception as e:
            print(f"❌ YuNet 初始化失败: {e}")
            self.detector = None
    
    def _find_model_path(self):
        """
        查找 YuNet 模型文件路径
        """
        model_name = 'face_detection_yunet_2023mar.onnx'
        
        # 尝试多个可能的路径
        search_paths = [
            cv2.data.haarcascades + model_name,
            '/usr/local/share/opencv4/face_detector/' + model_name,
            '/usr/share/opencv4/face_detector/' + model_name,
            str(project_root / 'models' / model_name),
            str(project_root / 'weights' / model_name),
            str(project_root / 'data' / 'models' / model_name),
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        # 如果找不到预训练模型，尝试下载
        print(f"⚠️ 未找到 YuNet 模型，尝试下载...")
        try:
            import urllib.request
            model_url = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{model_name}"
            download_path = str(project_root / 'models' / model_name)
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            
            print(f"📥 下载模型: {model_url}")
            urllib.request.urlretrieve(model_url, download_path)
            
            if os.path.exists(download_path):
                print(f"✅ 模型下载成功: {download_path}")
                return download_path
        except Exception as e:
            print(f"❌ 模型下载失败: {e}")
        
        return None
    
    def _resize_for_detection(self, image_np):
        """
        将图像等比例缩放至宽或高为 320（32 的倍数）
        
        Args:
            image_np: 原始图像
            
        Returns:
            缩放后的图像, 缩放比例
        """
        h, w = image_np.shape[:2]
        
        # 确保尺寸是 32 的倍数
        target_size = 320
        scale = min(target_size / w, target_size / h)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            new_w = (new_w // 32) * 32
            new_h = (new_h // 32) * 32
            image_resized = cv2.resize(image_np, (new_w, new_h))
        else:
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            image_resized = cv2.resize(image_np, (new_w, new_h))
        
        return image_resized, scale
    
    def detect_faces(self, image_path):
        """
        检测图像中的人脸
        
        Args:
            image_path: 图像路径
            
        Returns:
            人脸检测结果列表
        """
        if self.detector is None:
            return self._fallback_haar_detection(image_path)
        
        try:
            # 读取图像（BGR格式）
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 无法读取图像: {image_path}")
                return []
            
            # 缩放图像以提高检测精度
            image_resized, scale = self._resize_for_detection(image)
            
            # 设置输入尺寸
            h, w = image_resized.shape[:2]
            self.detector.setInputSize((w, h))
            
            # 执行检测
            _, faces = self.detector.detect(image_resized)
            
            if faces is None:
                return []
            
            # 转换回原始尺寸
            results = []
            for face in faces:
                x, y, w_face, h_face, confidence = face[:5]
                landmarks = face[5:] if len(face) > 5 else []
                
                # 转换到原始坐标
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(w_face / scale)
                h_orig = int(h_face / scale)
                
                results.append({
                    'x': x_orig,
                    'y': y_orig,
                    'w': w_orig,
                    'h': h_orig,
                    'confidence': float(confidence),
                    'landmarks': landmarks.tolist() if hasattr(landmarks, 'tolist') else []
                })
            
            print(f"🔍 YuNet 检测到 {len(results)} 个人脸")
            return results
            
        except Exception as e:
            print(f"❌ YuNet 检测失败: {e}")
            return self._fallback_haar_detection(image_path)
    
    def _fallback_haar_detection(self, image_path):
        """
        回退到 Haar 级联检测
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05, 
                minNeighbors=3, 
                minSize=(20, 20), 
                maxSize=(200, 200)
            )
            
            results = []
            for (x, y, w, h) in faces:
                results.append({
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'confidence': 0.8,
                    'landmarks': []
                })
            
            print(f"⚠️ Haar 回退检测到 {len(results)} 个人脸")
            return results
            
        except Exception as e:
            print(f"❌ Haar 检测也失败: {e}")
            return []


def auto_crop_faces(image_path, output_dir, padding=0.3):
    """
    自动切割多角色图片
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        padding: 边界框扩充比例
        
    Returns:
        切割后的文件列表
    """
    detector = YuNetFaceDetector(score_threshold=0.5)
    faces = detector.detect_faces(image_path)
    
    if len(faces) == 0:
        return []
    
    # 读取图像
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取原始文件名
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    cropped_files = []
    
    for i, face in enumerate(faces):
        x, y, w_face, h_face = face['x'], face['y'], face['w'], face['h']
        
        # 扩充边界框，保留头发特征
        pad_w = int(w_face * padding)
        pad_h = int(h_face * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + w_face + pad_w)
        y2 = min(h, y + h_face + pad_h // 2)
        
        # 切割图像
        cropped = image[y1:y2, x1:x2]
        
        # 保存切割后的图像
        output_filename = f"{name_without_ext}_crop_{i+1}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, cropped)
        cropped_files.append(output_path)
        
        print(f"✂️ 切割人脸 {i+1}: {output_filename} (置信度: {face['confidence']:.2f})")
    
    return cropped_files


def test_single_image(image_path):
    """
    测试单张图像
    """
    print(f"\n{'='*60}")
    print(f"测试图像: {image_path}")
    print('='*60)
    
    # 使用不同阈值测试
    for threshold in [0.3, 0.5, 0.7]:
        print(f"\n🎯 阈值: {threshold}")
        detector = YuNetFaceDetector(score_threshold=threshold)
        faces = detector.detect_faces(image_path)
        
        if faces:
            for i, face in enumerate(faces):
                print(f"  人脸 {i+1}: ({face['x']}, {face['y']}, {face['w']}, {face['h']}) 置信度: {face['confidence']:.3f}")
        else:
            print("  未检测到人脸")
    
    # 测试自动切割
    print("\n✂️ 自动切割测试")
    output_dir = './crop_test'
    cropped = auto_crop_faces(image_path, output_dir, padding=0.3)
    print(f"  切割出 {len(cropped)} 张图片")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YuNet 人脸检测优化工具")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        test_single_image(args.input)
    else:
        print(f"❌ 文件不存在: {args.input}")