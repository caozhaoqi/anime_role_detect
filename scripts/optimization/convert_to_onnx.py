#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 模型转换脚本
将 PyTorch 模型转换为 ONNX 格式，提升推断性能
"""

import argparse
import os
import torch
import torchvision.models as models
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_model(model_name, num_classes=1000):
    """加载预训练模型"""
    model_name = model_name.lower()

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def convert_to_onnx(model, input_size, output_path, opset_version=13):
    """将模型转换为 ONNX 格式"""
    # 创建示例输入
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # 设置模型为评估模式
    model.eval()

    # 导出为 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"ONNX 模型已保存到: {output_path}")


def quantize_model(model_path, output_path):
    """对 ONNX 模型进行量化"""
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType

        # 加载 ONNX 模型
        model = onnx.load(model_path)

        # 动态量化
        quantize_dynamic(model_path, output_path, weight_type=QuantType.QUInt8)

        print(f"量化模型已保存到: {output_path}")
        return True
    except ImportError:
        print("警告: onnxruntime 未安装，跳过量化")
        return False


def main():
    parser = argparse.ArgumentParser(description="将 PyTorch 模型转换为 ONNX 格式")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=["efficientnet_b0", "efficientnet_b3", "mobilenet_v2", "resnet50"],
        help="模型名称",
    )
    parser.add_argument("--weights", "-w", type=str, required=True, help="PyTorch 权重文件路径")
    parser.add_argument("--num_classes", "-n", type=int, default=78, help="类别数量")
    parser.add_argument("--input_size", "-s", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 ONNX 文件路径")
    parser.add_argument("--quantize", "-q", action="store_true", help="是否进行量化")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset 版本")

    args = parser.parse_args()

    # 设置输出路径
    if args.output is None:
        output_dir = PROJECT_ROOT / "models" / "onnx"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.model}.onnx"
    else:
        output_path = Path(args.output)

    print(f"正在加载模型: {args.model}")
    print(f"权重文件: {args.weights}")
    print(f"输入尺寸: {args.input_size}x{args.input_size}")
    print(f"类别数量: {args.num_classes}")

    # 加载模型
    model = load_model(args.model, args.num_classes)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    # 转换为 ONNX
    convert_to_onnx(model, args.input_size, output_path, args.opset)

    # 量化模型
    if args.quantize:
        quantized_path = output_path.parent / f"{args.model}_quantized.onnx"
        quantize_model(output_path, quantized_path)

    print("\n转换完成!")


if __name__ == "__main__":
    main()
