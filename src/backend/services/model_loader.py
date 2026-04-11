#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型加载服务
负责加载和管理各种模型
"""

import threading
from src.core.logging.global_logger import get_logger

logger = get_logger("model_loader")

# 延迟导入模块，避免在服务启动时执行模块初始化代码
Preprocessing = None
WDViTV3Tagger = None
MediaPipeKeypointDetector = None
AIRolePredictor = None
CacheManager = None
get_model = None

# 初始化预处理、关键点检测、标签生成和角色预测实例
preprocessor = None
keypoint_detector = None
tagger = None
role_predictor = None

# 添加线程锁，用于保护模块导入和初始化过程，避免锁竞争问题
load_models_lock = threading.Lock()


def load_models():
    """
    延迟加载模型
    """
    global preprocessor, keypoint_detector, tagger, role_predictor, Preprocessing, WDViTV3Tagger, MediaPipeKeypointDetector, AIRolePredictor, CacheManager, get_model
    
    # 使用线程锁保护模块导入和初始化过程，避免锁竞争问题
    with load_models_lock:
        # 延迟导入模块
        if Preprocessing is None:
            try:
                from src.core.preprocessing.preprocessing import Preprocessing
                logger.info("预处理模块导入成功")
            except Exception as e:
                logger.error(f"预处理模块导入失败: {e}")
        # 暂时跳过标签生成模块的导入，避免锁竞争问题
        # if WDViTV3Tagger is None:
        #     try:
        #         from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
        #         logger.info("标签生成模块导入成功")
        #     except Exception as e:
        #         logger.error(f"标签生成模块导入失败: {e}")
        # 暂时跳过关键点检测模块的导入，避免锁竞争问题
        # if MediaPipeKeypointDetector is None:
        #     try:
        #         from src.core.keypoint.mediapipe_keypoint_detector import MediaPipeKeypointDetector
        #         logger.info("关键点检测模块导入成功")
        #     except Exception as e:
        #         logger.error(f"关键点检测模块导入失败: {e}")
        if AIRolePredictor is None:
            try:
                from src.scripts.ai_role_prediction import AIRolePredictor
                logger.info("角色预测模块导入成功")
            except Exception as e:
                logger.error(f"角色预测模块导入失败: {e}")
        if CacheManager is None:
            try:
                from src.utils.cache_manager import CacheManager
                logger.info("缓存管理器导入成功")
            except Exception as e:
                logger.error(f"缓存管理器导入失败: {e}")
        if get_model is None:
            try:
                from src.core.classification.models import get_model
                logger.info("模型获取模块导入成功")
            except Exception as e:
                logger.error(f"模型获取模块导入失败: {e}")
        
        if preprocessor is None and Preprocessing is not None:
            try:
                preprocessor = Preprocessing()
                logger.info("预处理模块初始化成功")
            except Exception as e:
                logger.error(f"预处理模块初始化失败: {e}")
        if keypoint_detector is None and MediaPipeKeypointDetector is not None:
            try:
                keypoint_detector = MediaPipeKeypointDetector()
                logger.info("关键点检测模块初始化成功")
            except Exception as e:
                logger.error(f"关键点检测模块初始化失败: {e}")
        if tagger is None and WDViTV3Tagger is not None:
            try:
                tagger = WDViTV3Tagger()
                tagger.load_model()
                logger.info("标签生成模块初始化成功")
            except Exception as e:
                logger.error(f"标签生成模块初始化失败: {e}")
        if role_predictor is None and AIRolePredictor is not None:
            try:
                role_predictor = AIRolePredictor()
                logger.info("角色预测模块初始化成功")
            except Exception as e:
                logger.error(f"角色预测模块初始化失败: {e}")


def get_preprocessor():
    """
    获取预处理实例
    
    Returns:
        预处理实例
    """
    global preprocessor
    if preprocessor is None:
        load_models()
    return preprocessor


def get_keypoint_detector():
    """
    获取关键点检测实例
    
    Returns:
        关键点检测实例
    """
    global keypoint_detector
    if keypoint_detector is None:
        load_models()
    return keypoint_detector


def get_tagger():
    """
    获取标签生成实例
    
    Returns:
        标签生成实例
    """
    global tagger
    if tagger is None:
        load_models()
    return tagger


def get_role_predictor():
    """
    获取角色预测实例
    
    Returns:
        角色预测实例
    """
    global role_predictor
    if role_predictor is None:
        load_models()
    return role_predictor
