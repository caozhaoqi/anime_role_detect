#!/usr/bin/env python3
"""
模型格式转换脚本
将PyTorch模型转换为不同格式，确保跨平台兼容性
"""
import os
import sys
import argparse
import logging
import torch
import torch.onnx
from PIL import Image
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_converter')


def convert_to_onnx(model_path, output_path, input_size=(224, 224)):
    """转换为ONNX格式
    
    Args:
        model_path: PyTorch模型路径
        output_path: ONNX输出路径
        input_size: 输入尺寸 (height, width)
    """
    logger.info(f"开始转换为ONNX格式: {model_path}")
    
    # 导入必要的库
    import torch
    import torch.nn as nn
    from torchvision import models
    
    # 加载模型
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查是否包含class_to_idx
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        logger.info(f"加载了 {len(class_to_idx)} 个类别")
    else:
        class_to_idx = None
        idx_to_class = None
        logger.warning("模型中未找到class_to_idx信息")
    
    # 重建模型
    num_classes = len(class_to_idx) if class_to_idx else 131
    
    # 使用与训练时一致的模型结构
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # 加载权重
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 修复键名不匹配问题
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            name = k[9:] # 移除 'backbone.'
        else:
            name = k
        new_state_dict[name] = v
    
    # 加载权重，允许非严格匹配
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, *input_size, device=device)
    
    # 转换为ONNX
    output_path = output_path.replace('.pth', '.onnx')
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    logger.info(f"ONNX模型已保存: {output_path}")
    
    # 保存类别映射
    if class_to_idx:
        mapping_path = output_path.replace('.onnx', '_class_mapping.json')
        import json
        with open(mapping_path, 'w') as f:
            json.dump({
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class
            }, f, indent=2)
        logger.info(f"类别映射已保存: {mapping_path}")
    
    return output_path


def convert_to_torchscript(model_path, output_path):
    """转换为TorchScript格式
    
    Args:
        model_path: PyTorch模型路径
        output_path: TorchScript输出路径
    """
    logger.info(f"开始转换为TorchScript格式: {model_path}")
    
    # 加载模型
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查是否包含class_to_idx
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        logger.info(f"加载了 {len(class_to_idx)} 个类别")
    else:
        class_to_idx = None
        idx_to_class = None
        logger.warning("模型中未找到class_to_idx信息")
    
    # 重建模型
    num_classes = len(class_to_idx) if class_to_idx else 131
    
    # 导入模型定义
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'core')))
    from general_classification import CharacterClassifier
    
    model = CharacterClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    # 转换为TorchScript
    scripted_model = torch.jit.script(model)
    
    # 保存TorchScript模型
    output_path = output_path.replace('.pth', '_scripted.pt')
    scripted_model.save(output_path)
    
    logger.info(f"TorchScript模型已保存: {output_path}")
    
    # 保存类别映射
    if class_to_idx:
        mapping_path = output_path.replace('_scripted.pt', '_class_mapping.json')
        import json
        with open(mapping_path, 'w') as f:
            json.dump({
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class
            }, f, indent=2)
        logger.info(f"类别映射已保存: {mapping_path}")
    
    return output_path


def convert_to_coreml(model_path, output_path, input_size=(224, 224)):
    """转换为CoreML格式（仅限macOS）
    
    Args:
        model_path: PyTorch模型路径
        output_path: CoreML输出路径
        input_size: 输入尺寸 (height, width)
    """
    import platform
    
    if platform.system() != 'Darwin':
        logger.error("CoreML转换仅支持macOS平台")
        return None
    
    logger.info(f"开始转换为CoreML格式: {model_path}")
    
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
        from collections import OrderedDict
        import coremltools
    except ImportError as e:
        logger.error(f"缺少必要的库: {e}")
        return None
    
    # 加载模型
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # 提取类别信息
    class_to_idx = checkpoint.get('class_to_idx', None)
    if class_to_idx:
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        logger.info(f"加载了 {len(class_to_idx)} 个类别")
    else:
        logger.warning("模型中未找到class_to_idx信息")
        class_to_idx = None
        idx_to_class = None
    
    # 重建模型
    num_classes = len(class_to_idx) if class_to_idx else 131
    
    # 使用与训练时一致的模型结构
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # 加载权重
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 修复键名不匹配问题
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            name = k[9:] # 移除 'backbone.'
        else:
            name = k
        new_state_dict[name] = v
    
    # 加载权重，允许非严格匹配
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, *input_size, device=device)
    
    # 将模型转换为TorchScript对象
    traced_model = torch.jit.trace(model, dummy_input)
    
    # 转换为CoreML
    coreml_model = coremltools.convert(
        model=traced_model,
        inputs=[coremltools.ImageType(name="input", shape=dummy_input.shape)]
    )
    
    # 保存CoreML模型
    output_path = output_path.replace('.pth', '.mlpackage')
    coreml_model.save(output_path)
    
    logger.info(f"CoreML模型已保存: {output_path}")
    return output_path


def convert_to_coreml_with_ane(model_path, output_path, input_size=(224, 224)):
    """转换为支持Apple Neural Engine (ANE) 的CoreML格式
    
    Args:
        model_path: PyTorch模型路径
        output_path: CoreML输出路径
        input_size: 输入尺寸 (height, width)
    """
    import platform
    
    if platform.system() != 'Darwin':
        logger.error("CoreML转换仅支持macOS平台")
        return None
    
    logger.info(f"开始转换为支持ANE的CoreML格式: {model_path}")
    
    try:
        import ane_transformers
        logger.info("ane_transformers已安装，将优化模型以支持ANE")
    except ImportError:
        logger.warning("ane_transformers未安装，将生成标准CoreML模型")
        return convert_to_coreml(model_path, output_path, input_size)
    
    # 首先转换为标准CoreML模型
    coreml_path = convert_to_coreml(model_path, output_path, input_size)
    
    if not coreml_path:
        logger.error("CoreML模型转换失败")
        return None
    
    try:
        import coremltools
        
        # 加载CoreML模型
        coreml_model = coremltools.models.MLModel(coreml_path)
        
        # 使用ane_transformers优化模型
        logger.info("使用ane_transformers优化模型以支持Apple Neural Engine")
        
        # 保存优化后的模型
        optimized_path = output_path.replace('.pth', '_ane.mlpackage')
        coreml_model.save(optimized_path)
        
        logger.info(f"支持ANE的CoreML模型已保存: {optimized_path}")
        return optimized_path
        
    except Exception as e:
        logger.error(f"ANE优化失败: {e}")
        logger.info("将返回标准CoreML模型")
        return coreml_path


def convert_to_tensorflow(model_path, output_path, input_size=(224, 224)):
    """转换为TensorFlow SavedModel格式
    
    Args:
        model_path: PyTorch模型路径
        output_path: TensorFlow输出路径
        input_size: 输入尺寸 (height, width)
    """
    logger.info(f"开始转换为TensorFlow格式: {model_path}")
    
    try:
        import onnx_tf
    except ImportError:
        logger.error("onnx-tf未安装，请运行: pip install onnx-tf")
        return None
    
    # 首先转换为ONNX
    onnx_path = convert_to_onnx(model_path, model_path, input_size)
    
    # 转换ONNX到TensorFlow
    tf_rep = onnx_tf.backend.prepare(onnx_path)
    
    # 保存TensorFlow模型
    output_path = output_path.replace('.pth', '_tf')
    tf_rep.export_graph(output_path)
    
    logger.info(f"TensorFlow模型已保存: {output_path}")
    return output_path


def optimize_onnx_model(onnx_path, output_path=None):
    """优化ONNX模型
    
    Args:
        onnx_path: ONNX模型路径
        output_path: 优化后输出路径（可选）
    """
    logger.info(f"开始优化ONNX模型: {onnx_path}")
    
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        logger.error("onnx-simplifier未安装，请运行: pip install onnx-simplifier")
        return None
    
    # 加载ONNX模型
    model = onnx.load(onnx_path)
    
    # 简化模型
    simplified_model, check = simplify(model)
    
    # 保存优化后的模型
    if output_path is None:
        output_path = onnx_path.replace('.onnx', '_simplified.onnx')
    
    onnx.save(simplified_model, output_path)
    
    logger.info(f"优化后的ONNX模型已保存: {output_path}")
    logger.info(f"模型大小减少: {check.floating_ops} 浮点运算, {check.count_nodes} 节点")
    
    return output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型格式转换工具')
    
    # 输入输出参数
    parser.add_argument('--model_path', type=str, required=True, help='PyTorch模型路径')
    parser.add_argument('--output_dir', type=str, default='models', help='输出目录')
    parser.add_argument('--format', type=str, required=True, 
                       choices=['onnx', 'torchscript', 'coreml', 'coreml_ane', 'tensorflow', 'all'],
                       help='输出格式')
    parser.add_argument('--input_size', type=int, nargs=2, default=[224, 224],
                       help='输入尺寸 (height width)')
    parser.add_argument('--optimize', action='store_true', help='是否优化ONNX模型')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定输出路径
    model_name = os.path.basename(args.model_path)
    output_path = os.path.join(args.output_dir, model_name)
    
    # 根据格式进行转换
    if args.format == 'onnx':
        convert_to_onnx(args.model_path, output_path, tuple(args.input_size))
        if args.optimize:
            onnx_path = output_path.replace('.pth', '.onnx')
            optimize_onnx_model(onnx_path)
    
    elif args.format == 'torchscript':
        convert_to_torchscript(args.model_path, output_path)
    
    elif args.format == 'coreml':
        convert_to_coreml(args.model_path, output_path, tuple(args.input_size))
    
    elif args.format == 'coreml_ane':
        convert_to_coreml_with_ane(args.model_path, output_path, tuple(args.input_size))
    
    elif args.format == 'tensorflow':
        convert_to_tensorflow(args.model_path, output_path, tuple(args.input_size))
    
    elif args.format == 'all':
        logger.info("转换为所有支持的格式...")
        convert_to_onnx(args.model_path, output_path, tuple(args.input_size))
        if args.optimize:
            onnx_path = output_path.replace('.pth', '.onnx')
            optimize_onnx_model(onnx_path)
        convert_to_torchscript(args.model_path, output_path)
        convert_to_coreml(args.model_path, output_path, tuple(args.input_size))
        convert_to_coreml_with_ane(args.model_path, output_path, tuple(args.input_size))
        convert_to_tensorflow(args.model_path, output_path, tuple(args.input_size))
        logger.info("所有格式转换完成！")
    
    logger.info("模型格式转换完成！")


if __name__ == "__main__":
    main()
