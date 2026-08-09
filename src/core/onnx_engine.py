#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 推理引擎核心模块
为 API 服务提供高性能推理支持
"""

import os
import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache


class ONNXEngine:
    """ONNX 推理引擎"""

    _instances = {}  # 单例缓存

    def __new__(cls, model_path: str, use_gpu: bool = False):
        """单例模式，避免重复加载模型"""
        key = (model_path, use_gpu)
        if key not in cls._instances:
            cls._instances[key] = super(ONNXEngine, cls).__new__(cls)
        return cls._instances[key]

    def __init__(self, model_path: str, use_gpu: bool = False):
        """初始化推理引擎"""
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.input_size = None

        # 只在第一次创建时初始化
        if self.session is None:
            self._initialize()

    def _initialize(self):
        """初始化 ONNX 会话"""
        # 设置推理提供者
        if self.use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.model_path, providers=providers)

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        # 计算输入尺寸
        if len(self.input_shape) == 4:
            self.input_size = self.input_shape[2]
        else:
            self.input_size = 224

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """预处理图像"""
        # 统一到 RGB：RGBA/CMYK 会产生 4 通道（与 mean/std 的 (3,1,1) 广播失败），
        # 调色板图 P 经 np.array 得到的是**调色板索引**而非亮度，若按灰度分支处理
        # 会静默产出错误像素值。必须在转 numpy 之前归一化通道。
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 调整大小
        image = image.resize((self.input_size, self.input_size))

        # 转换为 numpy array
        image = np.array(image).astype(np.float32)

        # 如果是灰度图，转换为 RGB（convert("RGB") 后通常不会再触发，保留作兜底）
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # 转换通道顺序: HWC -> CHW
        image = image.transpose(2, 0, 1)

        # 归一化 (ImageNet 均值和标准差)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image = (image / 255.0 - mean) / std

        # 添加 batch 维度
        image = np.expand_dims(image, axis=0)

        return image

    def predict(self, image: Image.Image) -> np.ndarray:
        """进行推理"""
        input_data = self.preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return outputs[0]

    def predict_batch(self, images: List[Image.Image]) -> np.ndarray:
        """批量推理"""
        batch_data = np.concatenate([self.preprocess(img) for img in images], axis=0)
        outputs = self.session.run([self.output_name], {self.input_name: batch_data})
        return outputs[0]


class ModelManager:
    """模型管理器"""

    def __init__(self, models_dir: str = "models/onnx"):
        self.models_dir = Path(models_dir)
        self.engines: Dict[str, ONNXEngine] = {}

    def load_model(self, model_name: str, use_gpu: bool = False) -> ONNXEngine:
        """加载模型"""
        if model_name in self.engines:
            return self.engines[model_name]

        # 查找模型文件
        model_path = self.models_dir / f"{model_name}.onnx"
        if not model_path.exists():
            # 尝试其他格式
            for ext in [".onnx"]:
                candidate = self.models_dir / f"{model_name}{ext}"
                if candidate.exists():
                    model_path = candidate
                    break

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_name}")

        engine = ONNXEngine(str(model_path), use_gpu=use_gpu)
        self.engines[model_name] = engine

        return engine

    def unload_model(self, model_name: str):
        """卸载模型"""
        if model_name in self.engines:
            del self.engines[model_name]
            # 清理单例缓存
            keys_to_remove = []
            for key in ONNXEngine._instances.keys():
                if key[0].endswith(f"{model_name}.onnx"):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del ONNXEngine._instances[key]

    def list_models(self) -> List[str]:
        """列出可用模型"""
        models = []
        if self.models_dir.exists():
            for file in self.models_dir.iterdir():
                if file.suffix == ".onnx":
                    models.append(file.stem)
        return models


# 全局模型管理器实例
model_manager = ModelManager()


def get_engine(model_name: str, use_gpu: bool = False) -> ONNXEngine:
    """获取推理引擎"""
    return model_manager.load_model(model_name, use_gpu)


def release_engine(model_name: str):
    """释放推理引擎"""
    model_manager.unload_model(model_name)


def list_available_models() -> List[str]:
    """列出可用模型"""
    return model_manager.list_models()


# FastAPI 依赖注入
def get_model_manager() -> ModelManager:
    """获取模型管理器（FastAPI 依赖）"""
    return model_manager
