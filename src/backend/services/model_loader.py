#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型加载服务
负责加载和管理各种模型
"""

import threading
import gc
import torch
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


def cleanup_memory():
    """
    清理内存，减少内存占用
    """
    try:
        # 强制垃圾回收
        gc.collect()
        
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 清理MPS缓存（macOS）
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        logger.debug("内存清理完成")
    except Exception as e:
        logger.error(f"内存清理失败: {e}")


def load_preprocessor():
    """
    加载预处理模块
    """
    global preprocessor, Preprocessing
    with load_models_lock:
        if Preprocessing is None:
            try:
                from src.core.preprocessing.preprocessing import Preprocessing
                logger.info("预处理模块导入成功")
            except Exception as e:
                logger.error(f"预处理模块导入失败: {e}")
        
        if preprocessor is None and Preprocessing is not None:
            try:
                preprocessor = Preprocessing()
                logger.info("预处理模块初始化成功")
            except Exception as e:
                logger.error(f"预处理模块初始化失败: {e}")


def load_keypoint_detector():
    """
    加载关键点检测模块
    """
    global keypoint_detector, MediaPipeKeypointDetector
    with load_models_lock:
        if MediaPipeKeypointDetector is None:
            try:
                from src.core.keypoint.mediapipe_keypoint_detector import MediaPipeKeypointDetector
                logger.info("关键点检测模块导入成功")
            except Exception as e:
                logger.error(f"关键点检测模块导入失败: {e}")
        
        if keypoint_detector is None and MediaPipeKeypointDetector is not None:
            try:
                keypoint_detector = MediaPipeKeypointDetector()
                logger.info("关键点检测模块初始化成功")
            except Exception as e:
                logger.error(f"关键点检测模块初始化失败: {e}")


def load_tagger():
    """
    加载标签生成模块
    """
    global tagger, WDViTV3Tagger
    with load_models_lock:
        if WDViTV3Tagger is None:
            try:
                from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
                logger.info("标签生成模块导入成功")
            except Exception as e:
                logger.error(f"标签生成模块导入失败: {e}")
        
        if tagger is None and WDViTV3Tagger is not None:
            try:
                tagger = WDViTV3Tagger()
                tagger.load_model()
                logger.info("标签生成模块初始化成功")
            except Exception as e:
                logger.error(f"标签生成模块初始化失败: {e}")


def load_role_predictor():
    """
    加载角色预测模块
    """
    global role_predictor, AIRolePredictor
    with load_models_lock:
        if AIRolePredictor is None:
            try:
                from src.scripts.ai_role_prediction import AIRolePredictor
                logger.info("角色预测模块导入成功")
            except Exception as e:
                logger.error(f"角色预测模块导入失败: {e}")
        
        if role_predictor is None and AIRolePredictor is not None:
            try:
                role_predictor = AIRolePredictor()
                logger.info("角色预测模块初始化成功")
            except Exception as e:
                logger.error(f"角色预测模块初始化失败: {e}")


def load_cache_manager():
    """
    加载缓存管理器
    """
    global CacheManager
    with load_models_lock:
        if CacheManager is None:
            try:
                from src.utils.cache_manager import CacheManager
                logger.info("缓存管理器导入成功")
            except Exception as e:
                logger.error(f"缓存管理器导入失败: {e}")


def load_model_module():
    """
    加载模型获取模块
    """
    global get_model
    with load_models_lock:
        if get_model is None:
            try:
                from src.core.classification.models import get_model
                logger.info("模型获取模块导入成功")
            except Exception as e:
                logger.error(f"模型获取模块导入失败: {e}")


def load_models():
    """
    延迟加载模型
    """
    # 这里保持向后兼容，实际使用单独的加载函数
    load_preprocessor()
    load_role_predictor()
    load_cache_manager()
    load_model_module()


def get_preprocessor():
    """
    获取预处理实例
    
    Returns:
        预处理实例
    """
    global preprocessor
    if preprocessor is None:
        load_preprocessor()
    return preprocessor


def get_keypoint_detector():
    """
    获取关键点检测实例
    
    Returns:
        关键点检测实例
    """
    global keypoint_detector
    if keypoint_detector is None:
        load_keypoint_detector()
    return keypoint_detector


def get_tagger():
    """
    获取标签生成实例
    
    Returns:
        标签生成实例
    """
    global tagger
    if tagger is None:
        load_tagger()
    return tagger


def get_role_predictor():
    """
    获取角色预测实例
    
    Returns:
        角色预测实例
    """
    global role_predictor
    if role_predictor is None:
        load_role_predictor()
    return role_predictor
