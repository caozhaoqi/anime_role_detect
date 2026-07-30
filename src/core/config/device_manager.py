#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备管理器 - CUDA → MPS → CPU 自动检测链

全局单例，所有模块通过 DeviceManager.get_device() 获取推理设备。
检测顺序：CUDA → MPS → CPU，顺序不可更改。
"""

import os
from typing import Optional

from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("device_manager")


class DeviceManager:
    """CUDA → MPS → CPU 自动检测链（单例）"""

    _instance: Optional["DeviceManager"] = None
    _device: Optional[str] = None

    def __new__(cls) -> "DeviceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_device(cls) -> str:
        """返回推理设备字符串: 'cuda' / 'mps' / 'cpu'

        检测顺序: CUDA → MPS → CPU
        支持 FORCE_DEVICE 环境变量强制指定设备（值为 'cuda' / 'mps' / 'cpu'）。
        """
        if cls._device is not None:
            return cls._device

        # 优先检查环境变量强制指定
        force_device = os.getenv("FORCE_DEVICE", "").strip().lower()
        if force_device in ("cuda", "mps", "cpu"):
            cls._device = force_device
            logger.info(f"设备由 FORCE_DEVICE 环境变量强制指定: {force_device}")
            return cls._device

        try:
            import torch
        except ImportError:
            logger.warning("PyTorch 未安装，回退到 CPU")
            cls._device = "cpu"
            return cls._device

        # 1. CUDA 优先
        if torch.cuda.is_available():
            cls._device = "cuda"
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"
            logger.info(f"检测到 CUDA 可用，使用 NVIDIA GPU: {gpu_name}")
            return cls._device

        # 2. MPS 次之
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cls._device = "mps"
            logger.info("检测到 MPS 可用，使用 Apple Silicon GPU 加速")
            return cls._device

        # 3. 回退 CPU
        cls._device = "cpu"
        logger.info("未检测到 GPU，使用 CPU 推理")
        return cls._device

    @classmethod
    def to_device(cls, model) -> object:
        """将模型迁移到检测到的设备

        Args:
            model: PyTorch 模型实例

        Returns:
            迁移到目标设备后的模型
        """
        import torch

        device = cls.get_device()
        return model.to(device)

    @classmethod
    def is_gpu_available(cls) -> bool:
        """是否 GPU 可用（CUDA 或 MPS）

        Returns:
            True 如果 CUDA 或 MPS 可用，否则 False
        """
        try:
            import torch
        except ImportError:
            return False

        if torch.cuda.is_available():
            return True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
        return False

    @classmethod
    def reset(cls) -> None:
        """重置缓存的设备信息（主要用于测试）"""
        cls._device = None
