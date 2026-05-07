#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用人脸检测清理非单人图片"""
import os
import sys
from pathlib import Path
import cv2

IMG_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images')
TRASH_DIR = IMG_DIR / 'trash_multi_face'

# 人脸检测器路径
CASCADE_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/src/utils/xml_data/haarcascade_frontalface_default.xml'

def count_faces(image_path):
    """检测图片中的人脸数量"""
    try:
        # 加载人脸分类器
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            return -1  # 无法读取
        
        # 转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        return len(faces)
    except Exception as e:
        print(f"   ⚠️ 人脸检测失败 {image_path.name}: {e}")
        return -1

def cleanup_multi_face_images(max_faces=1):
    """清理包含多个人脸的图片"""
    print("=" * 60)
    print("🗑️ 使用人脸检测清理非单人图片")
    print("=" * 60)
    print(f"最大允许人脸数: {max_faces}")
    
    TRASH_DIR.mkdir(exist_ok=True)
    
    deleted_count = 0
    processed_count = 0
    multi_face_count = 0
    
    for folder in IMG_DIR.iterdir():
        if not folder.is_dir() or folder.name in ['其他', 'trash', 'trash_nsfw', 'trash_multi_face']:
            continue
            
        folder_name = folder.name
        print(f"\n📁 处理目录: {folder_name}")
        
        for img_path in folder.glob('*'):
            if not img_path.is_file():
                continue
                
            processed_count += 1
            
            face_count = count_faces(img_path)
            
            if face_count == -1:
                continue
                
            if face_count > max_faces:
                multi_face_count += 1
                
                # 移动到垃圾目录
                dest = TRASH_DIR / f"{folder_name}_{img_path.name}"
                os.rename(img_path, dest)
                deleted_count += 1
                
                print(f"   ❌ 删除 [多人脸]: {img_path.name}")
                print(f"      检测到人脸数: {face_count}")

    print("\n" + "=" * 60)
    print("✅ 非单人图片清理完成!")
    print(f"处理图片数: {processed_count}")
    print(f"检测到多人脸: {multi_face_count}")
    print(f"删除图片数: {deleted_count}")
    print(f"删除的文件已移至: {TRASH_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    # 可以通过命令行参数设置最大人脸数
    max_faces = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cleanup_multi_face_images(max_faces)