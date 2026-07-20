import sys
import os


from src.core.classification.efficientnet_inference import EfficientNetInference


def test_model_loading():
    """测试模型加载是否成功"""
    print("开始测试模型加载...")

    try:
        # 初始化推理器
        infer = EfficientNetInference()

        # 检查类别数量
        num_classes = len(infer.classes)
        print(f"类别数量: {num_classes}")

        # 检查模型是否成功加载
        if infer.model is not None:
            print("✓ 模型加载成功！")
        else:
            print("✗ 模型加载失败！")
            return False

        # 检查类别数量是否与模型一致
        # 获取模型最后一层的输出维度
        import torch

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(infer.device)
            output = infer.model(dummy_input)
            model_output_dim = output.shape[1]
            print(f"模型输出维度: {model_output_dim}")

        if num_classes == model_output_dim:
            print(f"✓ 类别数量匹配: {num_classes} = {model_output_dim}")
            return True
        else:
            print(f"✗ 类别数量不匹配: {num_classes} != {model_output_dim}")
            return False

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 所有测试通过！模型加载修复成功！")
    else:
        print("\n❌ 测试失败！请检查错误信息。")
