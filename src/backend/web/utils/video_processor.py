import cv2
import tempfile
import os
import sys
from loguru import logger

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.web.config.config import MAX_PROCESSED_FRAMES


def process_video(video_path, frame_skip=5):
    """处理视频文件，提取帧并进行分类
    
    Args:
        video_path: 视频路径
        frame_skip: 帧跳过间隔，用于减少处理帧数
    
    Returns:
        (video_results, overall_role, overall_similarity): 视频帧检测结果、整体角色、整体相似度
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_results = []
    frame_count = 0
    processed_frames = 0
    role_counts = {}
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 跳过一些帧以提高处理速度
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # 将帧保存为临时图像文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_frame:
                temp_frame_path = temp_frame.name
            
            # 保存帧
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                # 对帧进行分类
                from core.classification.general_classification import get_classifier
                
                classifier = get_classifier()
                role, similarity, boxes = classifier.classify_image(temp_frame_path)
                
                # 计算时间戳
                timestamp = frame_count / fps
                
                # 添加结果
                video_results.append({
                    'frame': frame_count,
                    'role': role,
                    'similarity': similarity,
                    'timestamp': timestamp,
                    'boxes': boxes
                })
                
                # 统计角色出现次数
                if role not in role_counts:
                    role_counts[role] = 0
                role_counts[role] += similarity
                
                processed_frames += 1
                
                # 限制处理的帧数，避免处理时间过长
                if processed_frames >= MAX_PROCESSED_FRAMES:
                    break
                    
            except Exception as e:
                logger.debug(f"处理帧 {frame_count} 时出错: {e}")
            finally:
                # 清理临时文件
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
            
            frame_count += 1
    
    finally:
        cap.release()
    
    # 确定整体角色
    overall_role = "未知"
    overall_similarity = 0.0
    
    if role_counts:
        # 选择出现次数最多的角色
        overall_role = max(role_counts, key=role_counts.get)
        overall_similarity = role_counts[overall_role] / processed_frames if processed_frames > 0 else 0
    
    return video_results, overall_role, overall_similarity
