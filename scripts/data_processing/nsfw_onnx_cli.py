#!/usr/bin/env python3
"""使用ONNX Runtime进行NSFW检测"""

import os
import sys
import json
import onnxruntime as ort
import numpy as np
from PIL import Image

def preprocess(image_path, input_size=(224, 224)):
    """图像预处理"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(input_size, Image.Resampling.LANCZOS)
    
    # 转换为numpy数组
    img_array = np.array(image).astype(np.float32)
    
    # 归一化 (0-255 -> 0-1)
    img_array = img_array / 255.0
    
    # 添加batch维度
    img_array = np.expand_dims(img_array, axis=0)
    
    # 调整通道顺序 (HWC -> CHW)
    img_array = np.transpose(img_array, (0, 3, 1, 2))
    
    return img_array

def detect_nsfw_onnx(image_path):
    """使用ONNX模型检测NSFW"""
    LABELS = ["drawings", "hentai", "neutral", "porn", "sexy"]
    
    # 设置ONNX Runtime使用CPU
    providers = ["CPUExecutionProvider"]
    
    # 尝试加载ONNX模型
    model_path = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/nsfw_model_img/src/main/resources/mobilenet_v2_140_224/saved_model.onnx"
    
    if not os.path.exists(model_path):
        print(json.dumps({"error": f"ONNX模型不存在: {model_path}"}))
        sys.exit(1)
    
    # 创建ONNX Runtime会话
    session = ort.InferenceSession(model_path, providers=providers)
    
    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 预处理图像
    img_array = preprocess(image_path)
    
    # 执行推理
    outputs = session.run([output_name], {input_name: img_array})
    
    # 获取预测结果
    logits = outputs[0][0]
    
    # 计算softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    # 构建结果
    details = {}
    for i, label in enumerate(LABELS):
        details[label] = float(probabilities[i])
    
    max_score = float(np.max(probabilities))
    max_index = int(np.argmax(probabilities))
    predicted_label = LABELS[max_index]
    
    nsfw_categories = ["porn", "sexy", "hentai"]
    nsfw_score = sum(details.get(cat, 0) for cat in nsfw_categories)
    is_nsfw = nsfw_score > 0.5
    skin_ratio = details.get("sexy", 0) * 0.6 + details.get("porn", 0) * 0.4
    
    result = {
        "is_nsfw": is_nsfw,
        "skin_ratio": float(skin_ratio),
        "nsfw_score": float(nsfw_score),
        "details": details,
        "method": "onnx",
        "predicted_label": predicted_label,
        "confidence": max_score
    }
    
    print(json.dumps(result))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "请提供图片路径"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    detect_nsfw_onnx(image_path)