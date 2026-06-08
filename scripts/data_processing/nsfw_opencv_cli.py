#!/usr/bin/env python3
"""使用OpenCV进行简单的NSFW检测（备用方案）"""

import os
import sys
import json
import cv2
import numpy as np

def detect_nsfw_opencv(image_path):
    """使用OpenCV进行NSFW检测"""
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "无法读取图片"}
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 皮肤颜色范围（HSV）
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 创建皮肤掩码
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 计算皮肤像素比例
        total_pixels = image.shape[0] * image.shape[1]
        skin_pixels = cv2.countNonZero(skin_mask)
        skin_ratio = skin_pixels / total_pixels
        
        # 检测图像亮度（可能表示暴露程度）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        
        # 检测边缘密度（可能表示衣物纹理）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.countNonZero(edges) / total_pixels
        
        # 综合判断NSFW概率
        nsfw_score = skin_ratio * 0.7 + (1 - edge_density) * 0.3
        
        # 判断是否为NSFW
        is_nsfw = nsfw_score > 0.35
        
        # 确定类别
        if nsfw_score > 0.6:
            predicted_label = "porn"
        elif nsfw_score > 0.4:
            predicted_label = "sexy"
        elif nsfw_score > 0.2:
            predicted_label = "hentai"
        else:
            predicted_label = "neutral"
        
        # 构建详细结果
        details = {
            "drawings": 0.1 if nsfw_score < 0.2 else 0.05,
            "hentai": max(0, min(1, (nsfw_score - 0.2) * 2.5)) if nsfw_score < 0.4 else 0.1,
            "neutral": max(0, min(1, 1 - nsfw_score * 1.5)),
            "porn": max(0, min(1, (nsfw_score - 0.5) * 2)),
            "sexy": max(0, min(1, nsfw_score * 1.2 - 0.1))
        }
        
        result = {
            "is_nsfw": bool(is_nsfw),
            "skin_ratio": float(skin_ratio),
            "nsfw_score": float(nsfw_score),
            "details": details,
            "method": "opencv_based",
            "predicted_label": predicted_label,
            "confidence": float(min(1.0, nsfw_score * 1.5))
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "请提供图片路径"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = detect_nsfw_opencv(image_path)
    print(json.dumps(result))