import os
import platform
import json
from PIL import Image

# 检查是否在Mac平台
if platform.system() != "Darwin":
    print("此测试仅在Mac平台上运行")
    exit(1)

# 尝试导入coremltools
try:
    import coremltools

    print("✅ coremltools已安装")
except ImportError:
    print("❌ coremltools未安装")
    exit(1)

# 检查Core ML模型是否存在
coreml_model_path = os.path.join("../models", "character_classifier_best_improved.mlpackage")
if not os.path.exists(coreml_model_path):
    print(f"❌ Core ML模型不存在: {coreml_model_path}")
    exit(1)

print(f"✅ Core ML模型存在: {coreml_model_path}")

# 加载Core ML模型
try:
    coreml_model = coremltools.models.MLModel(coreml_model_path)
    print("✅ Core ML模型加载成功")
    # 打印模型信息
    print(f"模型输入: {coreml_model.get_spec().description.input}")
    print(f"模型输出: {coreml_model.get_spec().description.output}")
except Exception as e:
    print(f"❌ Core ML模型加载失败: {e}")
    exit(1)

# 加载类别映射
mapping_path = os.path.join("../models", "character_classifier_best_improved_class_mapping.json")
idx_to_class = None
if os.path.exists(mapping_path):
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
        idx_to_class = mapping["idx_to_class"]
    print(f"✅ 类别映射加载成功，包含 {len(idx_to_class)} 个类别")
else:
    print("⚠️  类别映射文件不存在")

# 寻找测试图像
test_image = None
for root, dirs, files in os.walk(".."):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            test_image = os.path.join(root, file)
            print(f"✅ 找到测试图像: {test_image}")
            break
    if test_image:
        break

if not test_image:
    print("❌ 找不到测试图像")
    exit(1)

# 测试Core ML推理
try:
    print("\n=== 测试Core ML推理 ===")
    # 加载并预处理图像
    image = Image.open(test_image).convert("RGB")
    image = image.resize((224, 224))
    print(f"图像大小: {image.size}")

    # Core ML推理
    output = coreml_model.predict({"input": image})
    print(f"推理输出键: {list(output.keys())}")

    # 获取预测结果
    if "var_874" in output:
        predictions = output["var_874"]
    elif "output" in output:
        predictions = output["output"]
    else:
        # 尝试找到输出键
        output_keys = [k for k in output.keys() if k != "input"]
        if output_keys:
            predictions = output[output_keys[0]]
        else:
            raise ValueError("无法找到Core ML模型输出")

    print(f"预测结果形状: {predictions.shape}")

    # 获取最高概率的类别
    if len(predictions.shape) == 2:
        predictions = predictions[0]

    import numpy as np

    # 应用softmax转换为概率
    exp_predictions = np.exp(predictions - np.max(predictions))  # 数值稳定
    probabilities = exp_predictions / np.sum(exp_predictions)

    predicted_idx = int(np.argmax(probabilities))
    similarity = float(probabilities[predicted_idx])

    print(f"预测类别索引: {predicted_idx}")
    print(f"相似度: {similarity:.4f}")

    # 转换为角色名称
    if idx_to_class and predicted_idx in idx_to_class:
        role = idx_to_class[predicted_idx]
        print(f"预测角色: {role}")
    else:
        role = f"类别_{predicted_idx}"
        print(f"预测角色: {role}")

    print("\n🎉 Core ML模型测试成功！")
    print(f"结果: {role} (相似度: {similarity:.4f})")

except Exception as e:
    print(f"❌ Core ML推理失败: {e}")
    import traceback

    traceback.print_exc()
