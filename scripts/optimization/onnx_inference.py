#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 模型推理器
使用 ONNX Runtime 进行高性能推理
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import time

class ONNXModel:
    """ONNX 模型推理类"""
    
    def __init__(self, model_path, use_gpu=False):
        """
        初始化 ONNX 模型
        
        Args:
            model_path: ONNX 模型文件路径
            use_gpu: 是否使用 GPU 推理
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        
        # 设置推理提供者
        if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            self.session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            print("使用 GPU 推理")
        else:
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            print("使用 CPU 推理")
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # 计算输入尺寸
        self.input_size = self.input_shape[2]  # 假设格式为 [batch, channels, height, width]
        
        print(f"模型加载完成: {model_path}")
        print(f"输入形状: {self.input_shape}")
        print(f"输入尺寸: {self.input_size}x{self.input_size}")
    
    def preprocess(self, image):
        """
        预处理图像
        
        Args:
            image: PIL Image 对象
        
        Returns:
            numpy array, 形状为 [1, 3, height, width]
        """
        # 调整大小
        image = image.resize((self.input_size, self.input_size))
        
        # 转换为 numpy array
        image = np.array(image).astype(np.float32)
        
        # 如果是灰度图，转换为 RGB
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
    
    def predict(self, image):
        """
        进行推理
        
        Args:
            image: PIL Image 对象
        
        Returns:
            numpy array: 预测结果
        """
        # 预处理
        input_data = self.preprocess(image)
        
        # 推理
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        
        return outputs[0]
    
    def predict_batch(self, images):
        """
        批量推理
        
        Args:
            images: PIL Image 对象列表
        
        Returns:
            numpy array: 预测结果
        """
        # 预处理所有图像
        batch_data = np.concatenate([self.preprocess(img) for img in images], axis=0)
        
        # 推理
        outputs = self.session.run([self.output_name], {self.input_name: batch_data})
        
        return outputs[0]
    
    def benchmark(self, iterations=100):
        """
        性能基准测试
        
        Args:
            iterations: 测试迭代次数
        
        Returns:
            float: 平均推理时间 (ms)
            float: FPS
        """
        # 创建随机输入
        dummy_input = np.random.randn(1, 3, self.input_size, self.input_size).astype(np.float32)
        
        # 预热
        for _ in range(10):
            self.session.run([self.output_name], {self.input_name: dummy_input})
        
        # 正式测试
        start_time = time.time()
        for _ in range(iterations):
            self.session.run([self.output_name], {self.input_name: dummy_input})
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) / iterations * 1000
        fps = iterations / (end_time - start_time)
        
        print(f"\n性能基准测试结果:")
        print(f"迭代次数: {iterations}")
        print(f"平均推理时间: {avg_time_ms:.2f} ms")
        print(f"FPS: {fps:.2f}")
        
        return avg_time_ms, fps

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ONNX 模型推理测试')
    parser.add_argument('--model', '-m', type=str, required=True, help='ONNX 模型文件路径')
    parser.add_argument('--image', '-i', type=str, help='测试图像路径')
    parser.add_argument('--benchmark', '-b', action='store_true', help='运行性能基准测试')
    parser.add_argument('--gpu', '-g', action='store_true', help='使用 GPU 推理')
    
    args = parser.parse_args()
    
    # 加载模型
    model = ONNXModel(args.model, use_gpu=args.gpu)
    
    # 运行基准测试
    if args.benchmark:
        model.benchmark()
    
    # 测试图像推理
    if args.image:
        image = Image.open(args.image).convert('RGB')
        result = model.predict(image)
        print(f"\n推理结果形状: {result.shape}")
        print(f"预测类别: {np.argmax(result)}")
        print(f"置信度: {np.max(result):.4f}")

if __name__ == "__main__":
    main()
