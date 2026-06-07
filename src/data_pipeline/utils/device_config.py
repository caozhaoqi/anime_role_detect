#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备配置工具 - 在导入PyTorch之前设置环境变量，避免CUDA mutex错误
"""

import os
import platform


def configure_device():
    """
    配置设备环境变量，必须在导入任何PyTorch模块之前调用
    
    主要解决Mac上PyTorch的CUDA mutex错误问题
    """
    # Mac平台禁用CUDA，避免mutex错误
    if platform.system() == "Darwin":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["FORCE_CPU"] = "1"
        return "cpu"
    
    # 检查是否已禁用CUDA
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        return "cpu"
    
    # 其他平台尝试使用CUDA（延迟到实际使用时再检测）
    return None


def get_device(device: str = None) -> str:
    """
    获取设备类型
    
    Args:
        device: 指定设备，如果为None则自动选择
    
    Returns:
        设备字符串: 'cpu', 'cuda', 或 'mps'
    """
    if device is not None:
        return device
    
    # 先配置环境变量
    configured = configure_device()
    if configured:
        return configured
    
    # 其他平台延迟检测CUDA（不在此调用torch.cuda.is_available()）
    return "cuda"  # 让实际使用的代码去检测


# 在模块加载时就配置环境变量
configure_device()