#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSFW检测服务
负责检测图像是否包含敏感内容
基于 https://gitee.com/caozhaoqi/nsfw_model_img 项目实现
支持本地TensorFlow模型检测，同时支持TensorFlow Serving作为备用
"""

import os
import sys
import requests
import json
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s'
)
logger = logging.getLogger("nsfw_detector")

# TensorFlow Serving 配置
TF_SERVING_URL = "http://localhost:8501/v1/models/nsfw_model:predict"

# 本地模型路径
LOCAL_MODEL_PATH = Path(__file__).parent.parent.parent / "archived" / "nsfw_model_img" / "src" / "main" / "resources" / "mobilenet_v2_140_224"
TFLITE_MODEL_PATH = LOCAL_MODEL_PATH / "saved_model.tflite"

# 标签定义
LABELS = ["drawings", "hentai", "neutral", "porn", "sexy"]

# 模型实例（单例）
_model = None
_processor = None


def load_transformers_model():
    """加载HuggingFace Transformers模型"""
    global _model, _processor
    
    if _model is not None and _processor is not None:
        return _model, _processor
    
    try:
        # 在导入transformers之前设置环境变量（macOS兼容）
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        
        # 使用Falconsai的NSFW检测模型
        model_name = "Falconsai/nsfw_image_detection"
        
        _processor = AutoImageProcessor.from_pretrained(model_name)
        _model = AutoModelForImageClassification.from_pretrained(model_name)
        
        # 设置为评估模式
        _model.eval()
        
        logger.info(f"✅ 成功加载Transformers模型: {model_name}")
        return _model, _processor
    
    except ImportError as e:
        logger.warning(f"transformers库未安装，无法使用模型: {e}")
        return None, None
    except Exception as e:
        logger.error(f"加载Transformers模型失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        return None, None


def preprocess_image(image_source):
    """图像预处理"""
    try:
        from PIL import Image
        import numpy as np
        
        # 加载图像
        img = Image.open(image_source).convert("RGB")
        
        # 调整大小到224x224
        img = img.resize((224, 224))
        
        # 转换为numpy数组
        img_array = np.array(img, dtype=np.float32)
        
        # 归一化到[0, 1]
        img_array = img_array / 255.0
        
        # 添加批次维度
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        logger.error(f"图像预处理失败: {e}")
        return None


def detect_nsfw_with_local_model(image_source):
    """使用OpenCV进行NSFW检测（macOS兼容）"""
    try:
        import subprocess
        import json
        
        # 获取OpenCV检测脚本路径
        script_path = Path(__file__).parent.parent.parent / "scripts" / "data_processing" / "nsfw_opencv_cli.py"
        
        if not script_path.exists():
            logger.warning(f"OpenCV检测脚本不存在: {script_path}")
            return None
        
        # 使用subprocess调用独立脚本
        result = subprocess.run(
            [sys.executable, str(script_path), str(image_source)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            # 解析JSON结果
            try:
                output = json.loads(result.stdout)
                if "error" in output:
                    logger.error(f"OpenCV检测失败: {output['error']}")
                    return None
                logger.info(f"OpenCV检测成功，is_nsfw: {output.get('is_nsfw')}")
                return output
            except json.JSONDecodeError as e:
                logger.error(f"解析JSON失败: {e}")
                logger.error(f"脚本输出: {result.stdout}")
                return None
        else:
            logger.error(f"OpenCV脚本执行失败，返回码: {result.returncode}")
            logger.error(f"错误输出: {result.stderr}")
            return None
    
    except subprocess.TimeoutExpired:
        logger.error("OpenCV脚本执行超时")
        return None
    except Exception as e:
        logger.error(f"调用OpenCV脚本失败: {e}")
        return None


def detect_nsfw_with_tf_serving(image_source):
    """
    使用 TensorFlow Serving 进行 NSFW 检测

    Args:
        image_source: 图像路径或内存缓冲区(BytesIO)

    Returns:
        dict: NSFW检测结果
    """
    try:
        logger.info(f"使用 TensorFlow Serving 进行NSFW检测: {TF_SERVING_URL}")

        # 预处理图像
        from PIL import Image
        import numpy as np

        # 加载图像
        img = Image.open(image_source).convert("RGB")

        # 调整大小
        img = img.resize((224, 224))

        # 转换为数组
        img_array = np.array(img)

        # 归一化
        img_array = img_array / 255.0

        # 添加批次维度
        img_array = np.expand_dims(img_array, axis=0)

        # 构建请求数据
        request_data = {"instances": img_array.tolist()}

        # 发送请求
        response = requests.post(TF_SERVING_URL, json=request_data, timeout=30)

        if response.status_code == 200:
            result = response.json()
            logger.info(f"TensorFlow Serving 检测结果: {result}")

            if "predictions" in result and len(result["predictions"]) > 0:
                scores = result["predictions"][0]

                # 标签顺序
                labels = ["drawings", "hentai", "neutral", "porn", "sexy"]

                # 构建结果字典
                details = {}
                for i, label in enumerate(labels):
                    details[label] = float(scores[i])

                # 确定最高概率的类别
                max_score = float(max(scores))
                max_index = np.argmax(scores)
                predicted_label = labels[max_index]

                # 调整NSFW判断阈值
                nsfw_categories = ["porn", "sexy", "hentai"]

                # 基于类别设置不同的阈值
                thresholds = {"porn": 0.4, "sexy": 0.6, "hentai": 0.5}

                # 判断是否为NSFW
                is_nsfw = False
                if predicted_label in nsfw_categories:
                    threshold = thresholds.get(predicted_label, 0.5)
                    is_nsfw = max_score > threshold

                # 计算综合NSFW得分
                nsfw_score = 0
                for category in nsfw_categories:
                    nsfw_score += details.get(category, 0)
                nsfw_score = min(nsfw_score, 1.0)

                # 改进皮肤比例计算
                skin_ratio = 0.0
                if predicted_label in ["porn", "sexy"]:
                    # 结合置信度和类别权重计算皮肤比例
                    skin_ratio = max_score * 0.8 + (details.get("sexy", 0) * 0.2)
                skin_ratio = min(skin_ratio, 1.0)

                logger.info(
                    f"NSFW检测完成，类别: {predicted_label}, 置信度: {max_score:.4f}, is_nsfw: {is_nsfw}"
                )

                return {
                    "is_nsfw": is_nsfw,
                    "skin_ratio": float(skin_ratio),
                    "nsfw_score": float(nsfw_score),
                    "details": details,
                    "method": "tensorflow_serving"
                }
            else:
                logger.error(f"TensorFlow Serving 返回格式错误: {result}")
        else:
            logger.error(f"TensorFlow Serving 请求失败，状态码: {response.status_code}")
            logger.error(f"响应内容: {response.text}")
    except Exception as e:
        logger.error(f"TensorFlow Serving 检测失败: {e}")
        import traceback

        logger.error(f"异常堆栈: {traceback.format_exc()}")

    # 失败时返回None
    return None


def detect_nsfw(image_source):
    """
    NSFW检测

    Args:
        image_source: 图像路径或内存缓冲区(BytesIO)

    Returns:
        dict: NSFW检测结果
    """
    if isinstance(image_source, str):
        logger.info(f"开始NSFW检测: {image_source}")
    else:
        logger.info("开始NSFW检测: 内存缓冲区")

    try:
        # 检查文件是否存在（如果是文件路径）
        if isinstance(image_source, str) and not os.path.exists(image_source):
            logger.error(f"图像文件不存在: {image_source}")
            return {"is_nsfw": False, "skin_ratio": 0.0, "method": "error"}

        # 优先尝试使用本地TensorFlow模型
        local_result = detect_nsfw_with_local_model(image_source)
        if local_result is not None:
            return local_result
        logger.info("本地模型不可用，尝试TensorFlow Serving")

        # 尝试使用 TensorFlow Serving 进行检测
        tf_result = detect_nsfw_with_tf_serving(image_source)
        if tf_result is not None:
            return tf_result

        # 所有方法都失败时，使用基于规则的检测
        logger.warning("所有检测方法都失败，使用基于规则的NSFW检测")
        result = rule_based_detection(image_source)
        result["method"] = "rule_based"
        return result
    except Exception as e:
        logger.error(f"NSFW检测失败: {e}")
        import traceback

        logger.error(f"异常堆栈: {traceback.format_exc()}")

    # 默认返回值
    logger.info("NSFW检测返回默认值")
    return {"is_nsfw": False, "skin_ratio": 0.0, "method": "default"}


def rule_based_detection(image_source):
    """
    基于规则的NSFW检测（作为备用）

    Args:
        image_source: 图像路径或内存缓冲区

    Returns:
        dict: NSFW检测结果
    """
    try:
        from PIL import Image
        import numpy as np

        # 加载图像
        img = Image.open(image_source).convert("RGB")
        img_array = np.array(img)

        # 定义皮肤颜色范围（HSL）
        def rgb_to_hsl(r, g, b):
            r, g, b = r / 255.0, g / 255.0, b / 255.0
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            h, s, l = 0, 0, (max_val + min_val) / 2

            if max_val != min_val:
                d = max_val - min_val
                s = d / (2 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
                if max_val == r:
                    h = (g - b) / d + (6 if g < b else 0)
                elif max_val == g:
                    h = (b - r) / d + 2
                else:
                    h = (r - g) / d + 4
                h /= 6

            return h, s, l

        # 统计皮肤颜色像素数量
        skin_pixels = 0
        total_pixels = img_array.shape[0] * img_array.shape[1]

        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                r, g, b = img_array[i, j]
                h, s, l = rgb_to_hsl(r, g, b)

                # 皮肤颜色的HSL范围
                if (0.0 <= h <= 0.1 or 0.9 <= h <= 1.0) and 0.1 <= s <= 0.3 and 0.4 <= l <= 0.7:
                    skin_pixels += 1

        # 计算皮肤颜色像素比例
        skin_ratio = skin_pixels / total_pixels

        # 调整NSFW判断阈值
        # 基于皮肤比例设置动态阈值
        if skin_ratio > 0.5:
            is_nsfw = True
        elif skin_ratio > 0.3:
            # 中等皮肤比例，需要进一步判断
            is_nsfw = True
        else:
            is_nsfw = False

        # 构建检测结果
        scores = {
            "neutral": 1.0 - skin_ratio,
            "porn": skin_ratio * 0.5,
            "sexy": skin_ratio * 0.3,
            "hentai": skin_ratio * 0.1,
            "drawings": (1.0 - skin_ratio) * 0.5,
        }

        # 计算综合NSFW得分
        nsfw_categories = ["porn", "sexy", "hentai"]
        nsfw_score = 0
        for category in nsfw_categories:
            nsfw_score += scores.get(category, 0)
        nsfw_score = min(nsfw_score, 1.0)

        # 确定最高概率的类别
        max_score = 0
        predicted_label = "neutral"
        for label, score in scores.items():
            if score > max_score:
                max_score = score
                predicted_label = label

        # 输出详细日志
        logger.info(
            f"基于规则的NSFW检测完成，类别: {predicted_label}, 置信度: {max_score:.4f}, is_nsfw: {is_nsfw}"
        )

        return {
            "is_nsfw": is_nsfw,
            "skin_ratio": float(skin_ratio),
            "nsfw_score": float(nsfw_score),
            "details": scores,
        }
    except Exception as e:
        logger.error(f"基于规则的NSFW检测失败: {e}")
        return {"is_nsfw": False, "skin_ratio": 0.0}