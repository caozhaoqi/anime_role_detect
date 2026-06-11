#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe 关键点检测器

负责使用 MediaPipe 进行关键点检测
"""

import cv2
import numpy as np
import mediapipe as mp
from src.core.logging.global_logger import get_logger

logger = get_logger("mediapipe_keypoint_detector")


class MediaPipeKeypointDetector:
    """
    MediaPipe 关键点检测器类
    """

    def __init__(
        self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ):
        """
        初始化 MediaPipe 关键点检测器

        Args:
            model_complexity: 模型复杂度
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        # 初始化 MediaPipe 姿态估计
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # 初始化 MediaPipe 绘图工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        logger.info("MediaPipe 关键点检测器初始化完成")

    def detect(self, image):
        """
        检测关键点

        Args:
            image: 图像路径或 numpy 数组

        Returns:
            list: 关键点列表
        """
        try:
            # 加载/转换图像
            if isinstance(image, str):
                image = cv2.imread(image)
                if image is None:
                    logger.error(f"无法加载图像: {image}")
                    return []
            elif hasattr(image, 'convert'):  # PIL Image
                image = np.array(image.convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 转换为 RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 检测关键点
            results = self.pose.process(image_rgb)

            # 提取关键点
            keypoints = []
            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    keypoint = {
                        "id": idx,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility,
                    }
                    keypoints.append(keypoint)

            logger.debug(f"检测到 {len(keypoints)} 个关键点")
            return keypoints
        except Exception as e:
            logger.error(f"检测关键点失败: {e}")
            return []

    def draw(self, image, keypoints):
        """
        绘制关键点

        Args:
            image: 图像路径或 numpy 数组
            keypoints: 关键点列表

        Returns:
            numpy.ndarray: 绘制后的图像
        """
        try:
            # 加载图像
            if isinstance(image, str):
                image = cv2.imread(image)
                if image is None:
                    logger.error(f"无法加载图像: {image}")
                    return None

            # 创建空白图像
            output_image = image.copy()

            # 绘制关键点
            if keypoints:
                # 构建 MediaPipe 格式的关键点
                from mediapipe.framework.formats import landmark_pb2

                pose_landmarks = landmark_pb2.NormalizedLandmarkList()

                for keypoint in keypoints:
                    landmark = pose_landmarks.landmark.add()
                    landmark.x = keypoint["x"]
                    landmark.y = keypoint["y"]
                    landmark.z = keypoint["z"]
                    landmark.visibility = keypoint["visibility"]

                # 绘制骨架
                self.mp_drawing.draw_landmarks(
                    output_image,
                    pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            return output_image
        except Exception as e:
            logger.error(f"绘制关键点失败: {e}")
            return None

    def get_keypoint(self, keypoints, keypoint_id):
        """
        获取指定ID的关键点

        Args:
            keypoints: 关键点列表
            keypoint_id: 关键点ID

        Returns:
            dict: 关键点信息
        """
        for keypoint in keypoints:
            if keypoint["id"] == keypoint_id:
                return keypoint
        return None

    def get_keypoint_position(self, keypoints, keypoint_id):
        """
        获取指定ID的关键点位置

        Args:
            keypoints: 关键点列表
            keypoint_id: 关键点ID

        Returns:
            tuple: (x, y, z) 位置
        """
        keypoint = self.get_keypoint(keypoints, keypoint_id)
        if keypoint:
            return (keypoint["x"], keypoint["y"], keypoint["z"])
        return (0, 0, 0)

    def calculate_angle(self, keypoints, point1_id, point2_id, point3_id):
        """
        计算三个关键点之间的角度

        Args:
            keypoints: 关键点列表
            point1_id: 第一个关键点ID
            point2_id: 第二个关键点ID（顶点）
            point3_id: 第三个关键点ID

        Returns:
            float: 角度（度）
        """
        try:
            # 获取三个点的位置
            p1 = self.get_keypoint_position(keypoints, point1_id)
            p2 = self.get_keypoint_position(keypoints, point2_id)
            p3 = self.get_keypoint_position(keypoints, point3_id)

            # 计算向量
            v1 = (p1[0] - p2[0], p1[1] - p2[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])

            # 计算点积
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]

            # 计算向量长度
            v1_length = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
            v2_length = (v2[0] ** 2 + v2[1] ** 2) ** 0.5

            # 计算角度
            if v1_length > 0 and v2_length > 0:
                import math

                angle = math.acos(dot_product / (v1_length * v2_length))
                return math.degrees(angle)
            return 0.0
        except Exception as e:
            logger.error(f"计算角度失败: {e}")
            return 0.0

    def close(self):
        """
        关闭检测器
        """
        try:
            self.pose.close()
            logger.info("MediaPipe 关键点检测器已关闭")
        except Exception as e:
            logger.error(f"关闭检测器失败: {e}")
