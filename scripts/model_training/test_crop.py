#!/usr/bin/env python3
"""测试裁剪效果"""
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# 测试裁剪效果
test_imgs = list(Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset/Furina").glob("*.jpg"))[:3]
detector = YOLO('yolov8n.pt')

for img_path in test_imgs:
    img = Image.open(img_path)
    print(f"\n原图尺寸: {img.size}")
    
    results = detector.predict(img, verbose=False, conf=0.3)
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        print(f"检测框: {box}")
        
        # 裁剪
        x1, y1, x2, y2 = box
        padding = 0.3
        px, py = int((x2-x1)*padding), int((y2-y1)*padding)
        cropped = img.crop((max(0,x1-px), max(0,y1-py), min(img.width,x2+px), min(img.height,y2+py)))
        print(f"裁剪后尺寸: {cropped.size}")
        
        cropped.save(f"/tmp/test_crop_{img_path.name}")
        print(f"已保存: /tmp/test_crop_{img_path.name}")
    else:
        print("未检测到目标")