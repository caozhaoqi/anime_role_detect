#!/usr/bin/env python3
"""
综合性能测试脚本
测试所有推理模式的性能：PyTorch、Core ML、WebAssembly
"""
import os
import sys
import time
import logging
import argparse
import platform
from PIL import Image
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('inference_performance_test')

def load_image(image_path, input_size=(224, 224)):
    """加载并预处理图像
    
    Args:
        image_path: 图像路径
        input_size: 输入尺寸 (width, height)
    
    Returns:
        预处理后的图像数组
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize(input_size)
    
    # 转换为numpy数组
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    
    # 标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # 调整维度 (H, W, C) -> (C, H, W)
    image_array = np.transpose(image_array, (2, 0, 1))
    
    # 添加批次维度
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def test_pytorch_inference(model_path, test_images, input_size=(224, 224)):
    """测试PyTorch推理性能
    
    Args:
        model_path: PyTorch模型路径
        test_images: 测试图像列表
        input_size: 输入尺寸
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError as e:
        logger.error(f"缺少必要的库: {e}")
        return None
    
    logger.info("加载PyTorch模型...")
    
    # 加载模型
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # 提取类别信息
    class_to_idx = checkpoint.get('class_to_idx', None)
    num_classes = len(class_to_idx) if class_to_idx else 131
    
    # 重建模型
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # 加载权重
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 修复键名不匹配问题
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            name = k[9:]  # 移除 'backbone.'
        else:
            name = k
        new_state_dict[name] = v
    
    # 加载权重
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    # 测试推理速度
    logger.info("开始测试PyTorch推理速度...")
    inference_times = []
    
    for img_path in test_images:
        try:
            # 加载和预处理图像
            image_array = load_image(img_path, input_size)
            input_tensor = torch.tensor(image_array, device=device)
            
            # 推理
            with torch.no_grad():
                start_time = time.time()
                torch_output = model(input_tensor.float())
                end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            inference_times.append(inference_time)
            
            logger.info(f"PyTorch推理时间: {inference_time:.2f}ms - {os.path.basename(img_path)}")
        except Exception as e:
            logger.error(f"PyTorch推理失败: {e}")
    
    # 计算平均推理时间
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        logger.info(f"\n=== PyTorch模型性能报告 ===")
        logger.info(f"平均推理时间: {avg_time:.2f}ms")
        logger.info(f"最快推理时间: {min(inference_times):.2f}ms")
        logger.info(f"最慢推理时间: {max(inference_times):.2f}ms")
        
        return avg_time
    
    return None

def test_coreml_inference(mlmodel_path, test_images, input_size=(224, 224)):
    """测试Core ML推理性能
    
    Args:
        mlmodel_path: Core ML模型路径
        test_images: 测试图像列表
        input_size: 输入尺寸
    """
    if platform.system() != 'Darwin':
        logger.warning("Core ML测试仅支持macOS平台")
        return None
    
    try:
        import coremltools
    except ImportError as e:
        logger.error(f"缺少必要的库: {e}")
        return None
    
    logger.info("加载Core ML模型...")
    
    # 加载Core ML模型
    try:
        coreml_model = coremltools.models.MLModel(mlmodel_path)
    except Exception as e:
        logger.error(f"加载Core ML模型失败: {e}")
        return None
    
    # 测试推理速度
    logger.info("开始测试Core ML推理速度...")
    inference_times = []
    
    for img_path in test_images:
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            image = image.resize(input_size)
            
            # 准备输入
            input_data = {'input': image}
            
            # 推理
            start_time = time.time()
            coreml_output = coreml_model.predict(input_data)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            inference_times.append(inference_time)
            
            logger.info(f"Core ML推理时间: {inference_time:.2f}ms - {os.path.basename(img_path)}")
        except Exception as e:
            logger.error(f"Core ML推理失败: {e}")
    
    # 计算平均推理时间
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        logger.info(f"\n=== Core ML模型性能报告 ===")
        logger.info(f"平均推理时间: {avg_time:.2f}ms")
        logger.info(f"最快推理时间: {min(inference_times):.2f}ms")
        logger.info(f"最慢推理时间: {max(inference_times):.2f}ms")
        
        return avg_time
    
    return None

def test_onnx_inference(onnx_path, test_images, input_size=(224, 224)):
    """测试ONNX推理性能
    
    Args:
        onnx_path: ONNX模型路径
        test_images: 测试图像列表
        input_size: 输入尺寸
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        logger.error(f"缺少必要的库: {e}")
        return None
    
    logger.info("加载ONNX模型...")
    
    # 加载ONNX模型
    try:
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
    except Exception as e:
        logger.error(f"加载ONNX模型失败: {e}")
        return None
    
    # 测试推理速度
    logger.info("开始测试ONNX推理速度...")
    inference_times = []
    
    for img_path in test_images:
        try:
            # 加载和预处理图像
            image_array = load_image(img_path, input_size)
            
            # 转换为float32类型
            image_array = image_array.astype(np.float32)
            
            # 推理
            start_time = time.time()
            onnx_output = session.run([output_name], {input_name: image_array})
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            inference_times.append(inference_time)
            
            logger.info(f"ONNX推理时间: {inference_time:.2f}ms - {os.path.basename(img_path)}")
        except Exception as e:
            logger.error(f"ONNX推理失败: {e}")
    
    # 计算平均推理时间
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        logger.info(f"\n=== ONNX模型性能报告 ===")
        logger.info(f"平均推理时间: {avg_time:.2f}ms")
        logger.info(f"最快推理时间: {min(inference_times):.2f}ms")
        logger.info(f"最慢推理时间: {max(inference_times):.2f}ms")
        
        return avg_time
    
    return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='综合推理性能测试工具')
    
    # 输入参数
    parser.add_argument('--model_path', type=str, required=True, help='PyTorch模型路径')
    parser.add_argument('--test_dir', type=str, default='data/split_dataset/val', help='测试图像目录')
    parser.add_argument('--num_images', type=int, default=10, help='测试图像数量')
    parser.add_argument('--input_size', type=int, nargs=2, default=[224, 224], help='输入尺寸 (height width)')
    
    args = parser.parse_args()
    
    # 收集测试图像
    test_images = []
    for root, dirs, files in os.walk(args.test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                test_images.append(os.path.join(root, file))
        if len(test_images) >= args.num_images:
            break
    
    test_images = test_images[:args.num_images]
    
    if not test_images:
        logger.error("未找到测试图像")
        return
    
    logger.info(f"找到 {len(test_images)} 张测试图像")
    
    # 测试所有推理模式
    results = {}
    
    # 1. PyTorch推理
    pytorch_time = test_pytorch_inference(args.model_path, test_images, tuple(args.input_size))
    if pytorch_time:
        results['PyTorch'] = pytorch_time
    
    # 2. Core ML推理（仅macOS）
    if platform.system() == 'Darwin':
        mlmodel_path = args.model_path.replace('.pth', '.mlpackage')
        if os.path.exists(mlmodel_path):
            coreml_time = test_coreml_inference(mlmodel_path, test_images, tuple(args.input_size))
            if coreml_time:
                results['Core ML'] = coreml_time
    
    # 3. ONNX推理
    onnx_path = args.model_path.replace('.pth', '.onnx')
    if os.path.exists(onnx_path):
        onnx_time = test_onnx_inference(onnx_path, test_images, tuple(args.input_size))
        if onnx_time:
            results['ONNX'] = onnx_time
    
    # 性能对比
    logger.info("\n" + "="*50)
    logger.info("=== 综合性能对比 ===")
    logger.info("="*50)
    
    if results:
        # 按推理时间排序
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        
        for mode, time in sorted_results:
            logger.info(f"{mode:10s}: {time:8.2f}ms")
        
        # 计算加速比
        if len(results) > 1:
            baseline = sorted_results[-1][1]  # 最慢的
            logger.info("\n=== 加速比对比 (相对于最慢模式) ===")
            for mode, time in sorted_results:
                speedup = baseline / time
                logger.info(f"{mode:10s}: {speedup:6.2f}x")
    else:
        logger.warning("没有可用的推理模式测试结果")


if __name__ == "__main__":
    main()
