#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全转换 WD Vit Tagger v3 到 CoreML 格式
在导入 coremltools 前先封锁 MPS 探测，避免 macOS 锁竞争问题
"""

import os
import sys
import json

# ==================== 【关键步骤：在导入任何可能触发MPS的库前封锁MPS】 ====================
# 设置环境变量禁用 MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 先导入 torch 并封锁 MPS
import torch
if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
    torch.set_num_threads(1)
    print("[SAFE_BOOT] 已封锁 MPS 探测")

# 设置Hugging Face缓存目录
os.environ["HF_HOME"] = os.path.join(
    os.path.dirname(__file__), "huggingface_cache"
)

# 现在可以安全导入其他库
import coremltools as ct
from transformers import AutoProcessor, AutoModelForImageClassification

def main():
    print("=== WD Vit Tagger v3 -> CoreML 安全转换脚本 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CoreML Tools版本: {ct.__version__}")
    
    # 模型名称和输出路径
    model_name = "SmilingWolf/wd-vit-tagger-v3"
    output_dir = "./coreml_models"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载 PyTorch 模型和处理器
        print(f"\n[1/4] 加载模型: {model_name}")
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        
        # 设置模型为评估模式
        model.eval()
        model.to("cpu")  # 强制使用CPU
        print("模型加载完成")
        
        # 创建示例输入（448x448 RGB图像）
        print("\n[2/4] 创建示例输入...")
        dummy_input = torch.randn(1, 3, 448, 448)
        
        # 创建包装类，确保输出是张量而不是字典
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                outputs = self.model(x)
                # 只返回 logits 张量
                return outputs.logits
        
        wrapped_model = ModelWrapper(model)
        
        # 追踪模型
        print("\n[3/4] 追踪模型...")
        traced_model = torch.jit.trace(wrapped_model, (dummy_input,), check_trace=False)
        print("模型追踪完成")
        
        # 转换为 CoreML 格式
        print("\n[4/4] 转换为 CoreML 格式...")
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.ImageType(name="image", shape=(1, 3, 448, 448))],
            outputs=[ct.TensorType(name="logits")],
            compute_units=ct.ComputeUnit.CPU_AND_NE  # 使用 CPU 和 ANE
        )
        
        # 保存模型
        model_path = os.path.join(output_dir, "wd_tagger.mlpackage")
        coreml_model.save(model_path)
        print(f"CoreML 模型保存到: {model_path}")
        
        # 保存标签映射
        labels = model.config.id2label
        labels_path = os.path.join(output_dir, "wd_tagger_labels.json")
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)
        print(f"标签映射保存到: {labels_path}")
        
        print("\n=== ✅ 转换完成！ ===")
        print("\n使用方法：")
        print("1. 确保 coreml_models/ 目录包含：")
        print("   - wd_tagger.mlpackage")
        print("   - wd_tagger_labels.json")
        print("2. 修改 wd_vit_v3_tagger.py 中的 USE_COREML = True")
        print("3. 运行测试脚本验证")
        
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
