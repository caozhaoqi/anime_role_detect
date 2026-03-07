#!/usr/bin/env python3
"""
Core ML模型性能测试脚本
测试Core ML模型在Apple设备上的推理速度和准确性
"""
import os
import sys
import time
import logging
import argparse
from PIL import Image
import numpy as np
import platform

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('coreml_performance_test')

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

def test_coreml_model(mlmodel_path, test_images, input_size=(224, 224)):
    """测试Core ML模型性能
    
    Args:
        mlmodel_path: Core ML模型路径
        test_images: 测试图像列表
        input_size: 输入尺寸
    """
    if platform.system() != 'Darwin':
        logger.error("Core ML测试仅支持macOS平台")
        return None
    
    try:
        import coremltools
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError as e:
        logger.error(f"缺少必要的库: {e}")
        return None
    
    # 加载Core ML模型
    logger.info(f"加载Core ML模型: {mlmodel_path}")
    try:
        coreml_model = coremltools.models.MLModel(mlmodel_path)
    except Exception as e:
        logger.error(f"加载Core ML模型失败: {e}")
        return None
    
    # 加载对应的PyTorch模型作为对比
    torch_model_path = mlmodel_path.replace('.mlpackage', '.pth')
    if os.path.exists(torch_model_path):
        logger.info(f"加载PyTorch模型作为对比: {torch_model_path}")
        
        # 重建模型
        device = torch.device('cpu')
        checkpoint = torch.load(torch_model_path, map_location=device)
        
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
    else:
        model = None
        logger.warning(f"未找到对应的PyTorch模型: {torch_model_path}")
    
    # 测试推理速度
    logger.info("开始测试推理速度...")
    
    # Core ML推理时间
    coreml_times = []
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
            coreml_times.append(inference_time)
            
            logger.info(f"Core ML推理时间: {inference_time:.2f}ms - {os.path.basename(img_path)}")
        except Exception as e:
            logger.error(f"Core ML推理失败: {e}")
    
    # PyTorch推理时间
    torch_times = []
    if model:
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
                torch_times.append(inference_time)
                
                logger.info(f"PyTorch推理时间: {inference_time:.2f}ms - {os.path.basename(img_path)}")
            except Exception as e:
                logger.error(f"PyTorch推理失败: {e}")
    
    # 计算平均推理时间
    if coreml_times:
        avg_coreml_time = sum(coreml_times) / len(coreml_times)
        logger.info(f"\n=== Core ML模型性能报告 ===")
        logger.info(f"平均推理时间: {avg_coreml_time:.2f}ms")
        logger.info(f"最快推理时间: {min(coreml_times):.2f}ms")
        logger.info(f"最慢推理时间: {max(coreml_times):.2f}ms")
    
    if torch_times:
        avg_torch_time = sum(torch_times) / len(torch_times)
        logger.info(f"\n=== PyTorch模型性能报告 ===")
        logger.info(f"平均推理时间: {avg_torch_time:.2f}ms")
        logger.info(f"最快推理时间: {min(torch_times):.2f}ms")
        logger.info(f"最慢推理时间: {max(torch_times):.2f}ms")
        
        # 计算加速比
        speedup = avg_torch_time / avg_coreml_time
        logger.info(f"\n=== 性能对比 ===")
        logger.info(f"Core ML比PyTorch快: {speedup:.2f}倍")
    
    return {
        'coreml_times': coreml_times,
        'torch_times': torch_times,
        'avg_coreml_time': avg_coreml_time if coreml_times else None,
        'avg_torch_time': avg_torch_time if torch_times else None,
        'speedup': speedup if torch_times and coreml_times else None
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Core ML模型性能测试工具')
    
    # 输入参数
    parser.add_argument('--model_path', type=str, required=True, help='Core ML模型路径')
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
    
    # 测试模型
    test_coreml_model(args.model_path, test_images, tuple(args.input_size))


if __name__ == "__main__":
    main()
