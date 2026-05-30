#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型导出脚本 - 导出 CoreML 格式
针对 Apple Silicon 优化，使用 ANE 神经网络引擎

注: coremltools 9.0 的量化 API 有较大变化，
    完整的 INT8/FP16 量化功能需要在稳定版本中实现
"""

import os
import sys
import json
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

MODEL_NAME = "efficientnet_b3_loli_optimized_v2_20260529_133654"
MODEL_DIR = os.path.join(project_root, "models", MODEL_NAME)
OUTPUT_DIR = os.path.join(MODEL_DIR, "coreml")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_torch_model():
    """加载 PyTorch 模型"""
    print("=" * 60)
    print("🔄 加载 PyTorch 模型...")
    print("=" * 60)

    model_path = os.path.join(MODEL_DIR, "model_full.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "model_best.pth")

    with open(os.path.join(MODEL_DIR, "training_results.json"), "r") as f:
        config = json.load(f)

    num_classes = config.get("num_classes", 74)

    # 创建模型
    model = models.efficientnet_b3(num_classes=num_classes)

    # 加载权重
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    print(f"✅ 模型加载成功: {model_path}")
    return model, config


def create_sample_inputs():
    """创建示例输入用于 traced model"""
    print("\n📝 创建示例输入...")

    # 图像预处理
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 创建一个随机图像
    dummy_image = Image.new("RGB", (224, 224))
    input_tensor = transform(dummy_image).unsqueeze(0)

    print(f"   输入形状: {input_tensor.shape}")
    return input_tensor


def export_coreml(model, sample_inputs):
    """导出 CoreML 模型（Float32，自动使用 ANE/GPU/CPU）"""
    print("\n" + "=" * 60)
    print("🚀 导出 CoreML 模型...")
    print("=" * 60)

    try:
        import coremltools as ct

        # 使用 torch.jit.trace 追踪模型
        print("📌 Tracing 模型...")
        traced_model = torch.jit.trace(model, sample_inputs)

        # 转换为 CoreML，使用 ALL 计算单元（ANE + GPU + CPU）
        print("📌 转换为 CoreML 格式...")
        print("   计算单元: ANE + GPU + CPU")
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.ImageType(name="input", shape=sample_inputs.shape)],
            compute_units=ct.ComputeUnit.ALL,  # 使用 ANE + GPU + CPU
        )

        # 保存模型
        output_path = os.path.join(OUTPUT_DIR, "model_float.mlpackage")
        coreml_model.save(output_path)

        # 获取模型大小
        model_size = os.path.getsize(output_path) / (1024 * 1024)

        print(f"\n✅ CoreML 模型导出成功!")
        print(f"   路径: {output_path}")
        print(f"   大小: {model_size:.2f} MB")
        print(f"   计算单元: ANE (神经网络引擎) + GPU + CPU")

        return True, model_size

    except Exception as e:
        print(f"❌ CoreML 导出失败: {e}")
        import traceback

        traceback.print_exc()
        return False, 0


def benchmark_pytorch(num_runs=50):
    """测试 PyTorch 模型性能（作为对比基准）"""
    print("\n" + "=" * 60)
    print(f"⚡ PyTorch 基准测试 ({num_runs} 次推理)...")
    print("=" * 60)

    try:
        import time

        # 加载模型
        model_path = os.path.join(MODEL_DIR, "model_full.pth")
        with open(os.path.join(MODEL_DIR, "training_results.json"), "r") as f:
            config = json.load(f)

        num_classes = config.get("num_classes", 74)
        model = models.efficientnet_b3(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location="mps", weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.eval()

        # 预热
        print("   预热中...")
        dummy_input = torch.randn(1, 3, 224, 224)
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model = model.to("mps")
            dummy_input = dummy_input.to("mps")

        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)

        # 测试推理时间
        print(f"   运行 {num_runs} 次推理...")
        times = []
        with torch.no_grad():
            for i in range(num_runs):
                start = time.time()
                _ = model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif (
                    hasattr(torch.backends.mps, "is_available")
                    and torch.backends.mps.is_available()
                ):
                    torch.mps.synchronize()
                elapsed = (time.time() - start) * 1000  # ms
                times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / avg_time

        device = (
            "MPS"
            if hasattr(torch.backends.mps, "is_available") and torch.backends.mps.is_available()
            else "CPU"
        )

        print(f"\n📊 PyTorch ({device}) 性能结果:")
        print(f"   平均推理时间: {avg_time:.2f} ms")
        print(f"   标准差: {std_time:.2f} ms")
        print(f"   FPS: {fps:.2f}")

        return avg_time, fps

    except Exception as e:
        print(f"❌ PyTorch 基准测试失败: {e}")
        import traceback

        traceback.print_exc()
        return 0, 0


def main():
    print("=" * 60)
    print("🚀 CoreML 模型导出工具")
    print("=" * 60)
    print(f"\n模型: {MODEL_NAME}")
    print(f"输出目录: {OUTPUT_DIR}")

    # 加载模型
    model, config = load_torch_model()

    # 创建示例输入
    sample_inputs = create_sample_inputs()

    # 获取原始模型大小
    original_size = os.path.getsize(os.path.join(MODEL_DIR, "model_full.pth")) / (1024 * 1024)
    print(f"\n📦 原始 PyTorch 模型大小: {original_size:.2f} MB")

    results = {}

    # 导出 CoreML 模型
    coreml_success, coreml_size = export_coreml(model, sample_inputs)
    if coreml_success:
        results["coreml_float"] = {
            "size_mb": coreml_size,
            "compression_ratio": original_size / coreml_size if coreml_size > 0 else 0,
        }

    # PyTorch 基准测试
    pytorch_time, pytorch_fps = benchmark_pytorch(num_runs=50)
    results["pytorch_mps"] = {"avg_time_ms": pytorch_time, "fps": pytorch_fps}

    # 保存结果
    results_path = os.path.join(OUTPUT_DIR, "export_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "original_size_mb": original_size,
                "results": results,
                "note": "CoreML 模型使用 ANE + GPU + CPU 加速，PyTorch 使用 MPS 加速",
            },
            f,
            indent=2,
        )

    # 打印总结
    print("\n" + "=" * 60)
    print("📋 导出结果总结")
    print("=" * 60)
    print(f"\n原始模型: {original_size:.2f} MB")

    if coreml_success:
        r = results["coreml_float"]
        print(f"\nCoreML Float 模型:")
        print(f"   大小: {r['size_mb']:.2f} MB")
        print(f"   加速: ANE (神经网络引擎) + GPU + CPU")

    print(f"\nPyTorch MPS 基准:")
    print(f"   FPS: {results['pytorch_mps']['fps']:.2f}")

    print(f"\n结果已保存: {results_path}")
    print("=" * 60)

    print("\n💡 说明:")
    print("   coremltools 9.0 的量化 API 有较大变化")
    print("   完整的 INT8/FP16 量化功能需要在稳定版本中实现")
    print("   当前 CoreML 模型已启用 ANE 加速，可直接用于 iOS/macOS 应用")


if __name__ == "__main__":
    main()
