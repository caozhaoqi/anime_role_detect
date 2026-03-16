#!/usr/bin/env python3
"""
MediaPipe关键点检测器
用于检测面部、手部和身体姿态的关键点
"""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os

# 尝试导入svglib和reportlab，如果不可用则跳过SVG支持
try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    from io import BytesIO
    SVG_SUPPORT = True
except ImportError:
    print("警告: svglib和reportlab库不可用，SVG格式图片将被跳过")
    SVG_SUPPORT = False

class MediaPipeKeypointDetector:
    """使用MediaPipe进行关键点检测"""
    
    def __init__(self, min_detection_confidence=0.5):
        """初始化检测器
        
        Args:
            min_detection_confidence: 检测置信度阈值
        """
        # 配置参数
        self.min_detection_confidence = min_detection_confidence
        
        # 由于MediaPipe版本问题，我们使用OpenCV的Haar级联分类器作为替代
        # 加载面部和眼睛检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # 加载更多的分类器以提高检测率
        self.profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # 优化肤色检测参数，提高手部检测的准确性
        # 调整为更适合动漫角色的肤色范围
        # 为不同类型的动漫角色设置多个肤色范围
        self.skin_ranges = [
            # 浅色皮肤
            (np.array([0, 10, 60], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8)),
            # 中等皮肤
            (np.array([0, 15, 40], dtype=np.uint8), np.array([30, 255, 200], dtype=np.uint8)),
            # 深色皮肤
            (np.array([0, 20, 20], dtype=np.uint8), np.array([30, 255, 150], dtype=np.uint8))
        ]
    
    def detect_keypoints(self, image):
        """检测图像中的关键点
        
        Args:
            image: PIL图像对象或图像路径
            
        Returns:
            dict: 包含各种关键点的字典
        """
        # 加载图像
        if isinstance(image, str):
            # 处理SVG格式图片
            if image.lower().endswith('.svg'):
                image = self._load_svg_image(image)
            else:
                try:
                    image = Image.open(image).convert('RGB')
                except Exception as e:
                    print(f"加载图像失败: {e}")
                    return {
                        'face': None,
                        'hands': None,
                        'pose': None
                    }
        
        # 转换为OpenCV格式
        img_array = np.array(image)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 检测关键点
        face_keypoints = self._detect_face_keypoints(img_rgb)
        hand_keypoints = self._detect_hand_keypoints(img_rgb)
        pose_keypoints = self._detect_pose_keypoints(img_rgb, face_keypoints, hand_keypoints)
        
        results = {
            'face': face_keypoints,
            'hands': hand_keypoints,
            'pose': pose_keypoints
        }
        
        return results
    
    def _load_svg_image(self, svg_path):
        """加载SVG格式图片
        
        Args:
            svg_path: SVG文件路径
            
        Returns:
            PIL.Image: 转换后的图像对象
        """
        if not SVG_SUPPORT:
            print("SVG支持不可用，跳过SVG文件")
            return Image.new('RGB', (256, 256), color='white')
        
        try:
            # 使用svglib将SVG转换为ReportLab图形
            drawing = svg2rlg(svg_path)
            # 渲染为PNG
            png_data = renderPM.drawToString(drawing, fmt='PNG')
            # 从字节数据创建PIL图像
            return Image.open(BytesIO(png_data)).convert('RGB')
        except Exception as e:
            print(f"加载SVG图片失败: {e}")
            # 返回空白图像作为 fallback
            return Image.new('RGB', (256, 256), color='white')
    
    def _detect_face_keypoints(self, image):
        """检测面部关键点"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 调整Haar级联分类器的参数，提高面部检测的成功率
        # 减小scaleFactor，增加minNeighbors，提高检测精度
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.03,  # 更精细的缩放
            minNeighbors=2,     # 更少的邻居数，提高检测率
            minSize=(20, 20),   # 最小面部大小
            maxSize=(400, 400)  # 最大面部大小
        )
        
        # 检测侧面脸
        profile_faces = self.profile_face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.03, 
            minNeighbors=2,
            minSize=(20, 20),
            maxSize=(400, 400)
        )
        
        # 合并检测结果，避免重复
        all_faces = []
        face_centers = []
        
        # 处理正面脸
        for (x, y, w, h) in faces:
            center = (x + w//2, y + h//2)
            # 检查是否与已添加的面部重叠
            overlap = False
            for (cx, cy) in face_centers:
                distance = ((center[0] - cx) ** 2 + (center[1] - cy) ** 2) ** 0.5
                if distance < min(w, h) * 0.5:
                    overlap = True
                    break
            if not overlap:
                all_faces.append((x, y, w, h))
                face_centers.append(center)
        
        # 处理侧面脸
        for (x, y, w, h) in profile_faces:
            center = (x + w//2, y + h//2)
            # 检查是否与已添加的面部重叠
            overlap = False
            for (cx, cy) in face_centers:
                distance = ((center[0] - cx) ** 2 + (center[1] - cy) ** 2) ** 0.5
                if distance < min(w, h) * 0.5:
                    overlap = True
                    break
            if not overlap:
                all_faces.append((x, y, w, h))
                face_centers.append(center)
        
        if len(all_faces) == 0:
            return None
        
        face_keypoints = []
        
        for (x, y, w, h) in all_faces:
            # 计算面部边界框
            bbox = {
                'x1': x,
                'y1': y,
                'x2': x + w,
                'y2': y + h
            }
            
            # 检测眼睛作为关键点
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.1, 
                minNeighbors=2,
                minSize=(5, 5)
            )
            
            keypoints = []
            # 添加眼睛关键点
            for (ex, ey, ew, eh) in eyes:
                keypoints.append({'x': x + ex + ew//2, 'y': y + ey + eh//2})
            
            # 添加面部中心点
            keypoints.append({'x': x + w//2, 'y': y + h//2})
            
            face_keypoints.append({
                'keypoints': keypoints,
                'bbox': bbox
            })
        
        return face_keypoints
    
    def _detect_hand_keypoints(self, image):
        """检测手部关键点（使用肤色检测）"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 使用多个肤色范围进行检测
        masks = []
        for lower, upper in self.skin_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)
        
        # 合并所有掩码
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # 形态学操作，优化手部检测
        # 先进行膨胀操作，填充小的孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel)
        # 再进行腐蚀操作，去除噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        # 关闭操作，填充内部孔洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # 打开操作，去除外部噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hand_keypoints = []
        
        # 按轮廓面积排序，优先处理面积较大的轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # 只处理前10个最大的轮廓
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # 进一步减小最小面积阈值，提高检测率
                # 计算边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 过滤掉过大或过小的轮廓，同时考虑宽高比
                aspect_ratio = w / float(h)
                if 15 < w < 400 and 15 < h < 400 and 0.3 < aspect_ratio < 3.0:
                    # 计算轮廓的周长和面积比，用于判断是否为手部
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        # 手部的圆形度通常在0.2到0.8之间
                        if 0.2 < circularity < 0.8:
                            bbox = {
                                'x1': x,
                                'y1': y,
                                'x2': x + w,
                                'y2': y + h
                            }
                            
                            # 计算轮廓的质心作为关键点
                            M = cv2.moments(contour)
                            if M['m00'] > 0:
                                cx = int(M['m10'] / M['m00'])
                                cy = int(M['m01'] / M['m00'])
                                
                                # 计算手部的关键点，包括指尖和手掌中心
                                keypoints = [{'x': cx, 'y': cy, 'type': 'palm'}]
                                
                                # 尝试检测指尖
                                hull = cv2.convexHull(contour, returnPoints=False)
                                if len(hull) > 3:
                                    defects = cv2.convexityDefects(contour, hull)
                                    if defects is not None:
                                        for i in range(defects.shape[0]):
                                            s, e, f, d = defects[i, 0]
                                            start = tuple(contour[s][0])
                                            end = tuple(contour[e][0])
                                            far = tuple(contour[f][0])
                                            # 计算距离，只保留较远的点作为指尖
                                            try:
                                                # 确保far是正确的点坐标格式
                                                if isinstance(far, tuple) and len(far) == 2:
                                                    # 计算指尖和手掌中心的距离
                                                    dist = ((end[0] - cx) ** 2 + (end[1] - cy) ** 2) ** 0.5
                                                    if d > 1000 and dist > 20:
                                                        keypoints.append({'x': end[0], 'y': end[1], 'type': 'finger'})
                                            except Exception as e:
                                                pass
                                
                                # 尝试区分左右手
                                # 假设图像左侧为左手，右侧为右手
                                image_center = image.shape[1] // 2
                                handedness = 'Left' if cx < image_center else 'Right'
                                
                                hand_keypoints.append({
                                    'keypoints': keypoints,
                                    'bbox': bbox,
                                    'handedness': handedness,
                                    'area': area,
                                    'circularity': circularity
                                })
        
        # 过滤重叠的手部检测结果
        filtered_hand_keypoints = []
        for i, hand1 in enumerate(hand_keypoints):
            overlap = False
            for j, hand2 in enumerate(hand_keypoints):
                if i != j:
                    # 计算两个边界框的重叠区域
                    x1 = max(hand1['bbox']['x1'], hand2['bbox']['x1'])
                    y1 = max(hand1['bbox']['y1'], hand2['bbox']['y1'])
                    x2 = min(hand1['bbox']['x2'], hand2['bbox']['x2'])
                    y2 = min(hand1['bbox']['y2'], hand2['bbox']['y2'])
                    if x1 < x2 and y1 < y2:
                        overlap_area = (x2 - x1) * (y2 - y1)
                        hand1_area = (hand1['bbox']['x2'] - hand1['bbox']['x1']) * (hand1['bbox']['y2'] - hand1['bbox']['y1'])
                        hand2_area = (hand2['bbox']['x2'] - hand2['bbox']['x1']) * (hand2['bbox']['y2'] - hand2['bbox']['y1'])
                        if overlap_area / min(hand1_area, hand2_area) > 0.5:
                            overlap = True
                            break
            if not overlap:
                filtered_hand_keypoints.append(hand1)
        
        return filtered_hand_keypoints if filtered_hand_keypoints else None
    
    def _detect_pose_keypoints(self, image, face_keypoints, hand_keypoints):
        """检测身体姿态关键点（结合面部和手部的检测结果）"""
        # 改进姿态检测算法，结合面部和手部的检测结果
        
        # 收集所有检测到的点
        points = []
        
        # 添加面部点
        face_center = None
        if face_keypoints:
            for face in face_keypoints:
                bbox = face['bbox']
                points.append((bbox['x1'], bbox['y1']))
                points.append((bbox['x2'], bbox['y2']))
                # 添加面部中心点
                for point in face['keypoints']:
                    points.append((point['x'], point['y']))
                # 计算面部中心点
                face_center = (bbox['x1'] + bbox['x2']) // 2, (bbox['y1'] + bbox['y2']) // 2
        
        # 添加手部点
        hand_points = []
        if hand_keypoints:
            for hand in hand_keypoints:
                bbox = hand['bbox']
                points.append((bbox['x1'], bbox['y1']))
                points.append((bbox['x2'], bbox['y2']))
                # 添加手部中心点
                for point in hand['keypoints']:
                    points.append((point['x'], point['y']))
                    hand_points.append((point['x'], point['y']))
        
        # 如果没有足够的点，尝试使用肤色检测
        if len(points) < 2:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # 使用多个肤色范围
            masks = []
            for lower, upper in self.skin_ranges:
                mask = cv2.inRange(hsv, lower, upper)
                masks.append(mask)
            
            # 合并所有掩码
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for mask in masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    points.append((x, y))
                    points.append((x + w, y + h))
        
        if len(points) < 2:
            return None
        
        # 计算身体边界框
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        bbox = {
            'x1': min(x_coords),
            'y1': min(y_coords),
            'x2': max(x_coords),
            'y2': max(y_coords)
        }
        
        # 扩展边界框，使其更符合身体比例
        width = bbox['x2'] - bbox['x1']
        height = bbox['y2'] - bbox['y1']
        
        # 调整边界框，使其更符合人体比例
        if height < width * 1.5:
            # 增加高度
            new_height = int(width * 2.0)  # 进一步增加高度，更符合人体比例
            center_y = (bbox['y1'] + bbox['y2']) // 2
            bbox['y1'] = max(0, center_y - new_height // 2)
            bbox['y2'] = bbox['y1'] + new_height
        
        # 重新计算宽高
        width = bbox['x2'] - bbox['x1']
        height = bbox['y2'] - bbox['y1']
        
        # 更详细的身体关键点
        keypoints = []
        
        # 头部关键点（使用面部中心点）
        if face_center:
            head_x, head_y = face_center
        else:
            head_x, head_y = (bbox['x1'] + bbox['x2']) // 2, bbox['y1']
        keypoints.append({'x': head_x, 'y': head_y, 'visibility': 1.0, 'name': 'Head'})  # 头部
        
        # 颈部关键点
        neck_x, neck_y = head_x, head_y + height // 6
        keypoints.append({'x': neck_x, 'y': neck_y, 'visibility': 1.0, 'name': 'Neck'})  # 颈部
        
        # 肩部关键点
        left_shoulder_x, left_shoulder_y = bbox['x1'] + width // 4, neck_y
        right_shoulder_x, right_shoulder_y = bbox['x2'] - width // 4, neck_y
        keypoints.append({'x': left_shoulder_x, 'y': left_shoulder_y, 'visibility': 1.0, 'name': 'Left Shoulder'})  # 左肩
        keypoints.append({'x': right_shoulder_x, 'y': right_shoulder_y, 'visibility': 1.0, 'name': 'Right Shoulder'})  # 右肩
        
        # 肘部关键点（使用手部位置）
        left_elbow_x, left_elbow_y = left_shoulder_x, neck_y + height // 3
        right_elbow_x, right_elbow_y = right_shoulder_x, neck_y + height // 3
        
        # 如果有手部点，调整肘部位置
        if hand_points:
            for (hx, hy) in hand_points:
                if hx < head_x:  # 左手
                    left_elbow_x, left_elbow_y = (left_shoulder_x + hx) // 2, (left_shoulder_y + hy) // 2
                else:  # 右手
                    right_elbow_x, right_elbow_y = (right_shoulder_x + hx) // 2, (right_shoulder_y + hy) // 2
        
        keypoints.append({'x': left_elbow_x, 'y': left_elbow_y, 'visibility': 1.0, 'name': 'Left Elbow'})  # 左肘
        keypoints.append({'x': right_elbow_x, 'y': right_elbow_y, 'visibility': 1.0, 'name': 'Right Elbow'})  # 右肘
        
        # 腰部关键点
        waist_x, waist_y = head_x, neck_y + height // 2
        keypoints.append({'x': waist_x, 'y': waist_y, 'visibility': 1.0, 'name': 'Waist'})  # 腰部
        
        # 臀部关键点
        hip_x, hip_y = head_x, bbox['y2'] - height // 4
        keypoints.append({'x': hip_x, 'y': hip_y, 'visibility': 1.0, 'name': 'Hip'})  # 臀部
        
        # 膝盖关键点
        left_knee_x, left_knee_y = bbox['x1'] + width // 4, bbox['y2'] - height // 6
        right_knee_x, right_knee_y = bbox['x2'] - width // 4, bbox['y2'] - height // 6
        keypoints.append({'x': left_knee_x, 'y': left_knee_y, 'visibility': 1.0, 'name': 'Left Knee'})  # 左膝
        keypoints.append({'x': right_knee_x, 'y': right_knee_y, 'visibility': 1.0, 'name': 'Right Knee'})  # 右膝
        
        # 脚踝关键点
        left_ankle_x, left_ankle_y = bbox['x1'] + width // 4, bbox['y2']
        right_ankle_x, right_ankle_y = bbox['x2'] - width // 4, bbox['y2']
        keypoints.append({'x': left_ankle_x, 'y': left_ankle_y, 'visibility': 1.0, 'name': 'Left Ankle'})  # 左脚踝
        keypoints.append({'x': right_ankle_x, 'y': right_ankle_y, 'visibility': 1.0, 'name': 'Right Ankle'})  # 右脚踝
        
        return {
            'keypoints': keypoints,
            'bbox': bbox
        }
    
    def draw_keypoints(self, image, keypoints):
        """在图像上绘制关键点和边界框
        
        Args:
            image: PIL图像对象
            keypoints: 检测到的关键点
            
        Returns:
            PIL.Image: 绘制了关键点的图像
        """
        # 转换为OpenCV格式
        img_array = np.array(image)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 绘制面部关键点
        if keypoints.get('face'):
            for face in keypoints['face']:
                # 绘制边界框
                bbox = face['bbox']
                cv2.rectangle(img_rgb, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 2)
                cv2.putText(img_rgb, 'Face', (bbox['x1'], bbox['y1'] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # 绘制关键点
                for point in face['keypoints']:
                    cv2.circle(img_rgb, (point['x'], point['y']), 3, (0, 255, 0), -1)
        
        # 绘制手部关键点
        if keypoints.get('hands'):
            for hand in keypoints['hands']:
                # 绘制边界框
                bbox = hand['bbox']
                # 根据左右手使用不同的颜色
                color = (0, 0, 255) if hand.get('handedness') == 'Left' else (255, 0, 0)
                cv2.rectangle(img_rgb, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color, 2)
                hand_label = f"Hand ({hand.get('handedness', 'Unknown')})"
                cv2.putText(img_rgb, hand_label, (bbox['x1'], bbox['y1'] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # 绘制关键点
                for point in hand['keypoints']:
                    # 根据关键点类型使用不同的颜色和大小
                    if point.get('type') == 'palm':
                        cv2.circle(img_rgb, (point['x'], point['y']), 5, color, -1)
                        cv2.putText(img_rgb, 'Palm', (point['x'] + 10, point['y']), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    elif point.get('type') == 'finger':
                        cv2.circle(img_rgb, (point['x'], point['y']), 3, (0, 255, 0), -1)
                        cv2.putText(img_rgb, 'Finger', (point['x'] + 10, point['y']), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制身体姿态关键点
        if keypoints.get('pose'):
            pose = keypoints['pose']
            # 绘制边界框
            if pose['bbox']:
                bbox = pose['bbox']
                cv2.rectangle(img_rgb, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255), 2)
                cv2.putText(img_rgb, 'Pose', (bbox['x1'], bbox['y1'] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # 绘制关键点
            for i, point in enumerate(pose['keypoints']):
                if point.get('visibility', 1.0) > 0.5:
                    # 绘制关键点
                    cv2.circle(img_rgb, (point['x'], point['y']), 4, (0, 0, 255), -1)
                    # 显示关键点名称
                    if 'name' in point:
                        cv2.putText(img_rgb, point['name'], (point['x'] + 10, point['y']), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 绘制身体连接线
            if pose['keypoints']:
                # 身体连接点（更新为包含脚踝）
                connections = [
                    (0, 1),  # 头部 -> 颈部
                    (1, 2),  # 颈部 -> 左肩
                    (1, 3),  # 颈部 -> 右肩
                    (1, 6),  # 颈部 -> 腰部
                    (2, 4),  # 左肩 -> 左肘
                    (3, 5),  # 右肩 -> 右肘
                    (6, 7),  # 腰部 -> 臀部
                    (7, 8),  # 臀部 -> 左膝
                    (7, 9),  # 臀部 -> 右膝
                    (8, 10),  # 左膝 -> 左脚踝
                    (9, 11),  # 右膝 -> 右脚踝
                ]
                
                for conn in connections:
                    if conn[0] < len(pose['keypoints']) and conn[1] < len(pose['keypoints']):
                        p1 = pose['keypoints'][conn[0]]
                        p2 = pose['keypoints'][conn[1]]
                        if p1.get('visibility', 1.0) > 0.5 and p2.get('visibility', 1.0) > 0.5:
                            cv2.line(img_rgb, (p1['x'], p1['y']), (p2['x'], p2['y']), (0, 0, 255), 2)
        
        # 转换回PIL格式
        img_pil = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        return img_pil
    
    def close(self):
        """关闭检测器"""
        # OpenCV的级联分类器不需要显式关闭
        pass

if __name__ == "__main__":
    # 测试代码
    detector = MediaPipeKeypointDetector()
    
    # 测试图像路径
    test_image = "path/to/test/image.jpg"
    
    try:
        # 检测关键点
        keypoints = detector.detect_keypoints(test_image)
        print("检测结果:")
        print(f"面部检测: {'成功' if keypoints['face'] else '失败'}")
        print(f"手部检测: {'成功' if keypoints['hands'] else '失败'}")
        print(f"姿态检测: {'成功' if keypoints['pose'] else '失败'}")
        
        # 绘制关键点
        from PIL import Image
        image = Image.open(test_image)
        annotated_image = detector.draw_keypoints(image, keypoints)
        annotated_image.save("annotated_image.jpg")
        print("标注图像已保存为 annotated_image.jpg")
        
    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        detector.close()
