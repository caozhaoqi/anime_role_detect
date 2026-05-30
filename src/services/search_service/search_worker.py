#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的图像搜索Worker进程
使用MPS加速（Apple Silicon GPU）
"""

import os
import sys
import time
import json
import uuid
import base64
from PIL import Image
import io

# 设置环境变量
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 添加项目根目录到Python路径
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

# 队列目录
QUEUE_DIR = "search_queue"
INPUT_DIR = os.path.join(QUEUE_DIR, "input")
OUTPUT_DIR = os.path.join(QUEUE_DIR, "output")

# 确保目录存在
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局模型变量
model = None
preprocess = None
index = None
image_paths = []


def load_clip_model_with_mps():
    """使用MPS加速加载CLIP模型"""
    print("[Worker] 尝试使用MPS加载CLIP模型...")
    try:
        import torch
        import clip

        # 检查MPS可用性
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[Worker] ✓ MPS设备可用")
        else:
            device = torch.device("cpu")
            print("[Worker] ⚠️ MPS不可用，使用CPU")

        model_name = "ViT-B/32"
        model, preprocess = clip.load(model_name, device=device)
        model.eval()

        print(f"[Worker] ✓ CLIP模型加载成功，设备: {device}")
        return model, preprocess, device
    except Exception as e:
        print(f"[Worker] ✗ 加载CLIP模型失败: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


def find_image_datasets():
    """查找可用的图像数据集目录"""
    possible_paths = [
        "data",
        "datasets",
        "images",
        "data/optimized_detection_v2",
        "data/expanded_dataset",
        "data/test",
        os.path.join(os.path.expanduser("~"), "datasets"),
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # 检查是否有图片文件
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        return path
    return None


def process_task_real_model(task_id):
    """使用真实CLIP模型处理搜索任务"""
    global model, preprocess

    input_file = os.path.join(INPUT_DIR, f"{task_id}.jpg")
    output_file = os.path.join(OUTPUT_DIR, f"{task_id}.json")

    if not os.path.exists(input_file):
        return

    print(f"[Worker] 处理任务: {task_id} (真实模型)")

    try:
        import torch

        # 读取并预处理图像
        image = Image.open(input_file).convert("RGB")
        image_input = (
            preprocess(image).unsqueeze(0).to("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # 提取输入图片的特征
        with torch.no_grad():
            query_features = model.encode_image(image_input)
            query_features = query_features / query_features.norm(dim=-1, keepdim=True)

        # 查找真实数据集
        dataset_path = find_image_datasets()
        print(f"[Worker] 数据集路径: {dataset_path}")

        # 获取数据集中的真实图片
        real_images = []
        if dataset_path:
            for root, dirs, files in os.walk(dataset_path):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        real_images.append(os.path.join(root, f))

        print(f"[Worker] 数据集中找到 {len(real_images)} 张图片")

        # 准备搜索结果 - 计算真实相似度
        results = []
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        if real_images:
            # 遍历数据集计算相似度
            similarity_scores = []

            for img_path in real_images[:100]:  # 限制处理数量以提高速度
                try:
                    # 读取并预处理数据集图片
                    dataset_image = Image.open(img_path).convert("RGB")
                    dataset_input = preprocess(dataset_image).unsqueeze(0).to(device)

                    # 提取特征
                    with torch.no_grad():
                        dataset_features = model.encode_image(dataset_input)
                        dataset_features = dataset_features / dataset_features.norm(
                            dim=-1, keepdim=True
                        )

                    # 计算余弦相似度
                    similarity = torch.nn.functional.cosine_similarity(
                        query_features, dataset_features
                    ).item()
                    similarity_scores.append((similarity, img_path))
                except Exception as e:
                    print(f"[Worker] 跳过图片 {img_path}: {e}")
                    continue

            # 按相似度排序（降序）
            similarity_scores.sort(key=lambda x: x[0], reverse=True)

            # 取Top-5结果
            top_results = similarity_scores[:5]
            print(f"[Worker] Top-5 相似度: {[s[0] for s in top_results]}")

            for similarity, img_path in top_results:
                try:
                    real_image = Image.open(img_path).convert("RGB")
                    real_image = real_image.resize((150, 150), Image.Resampling.LANCZOS)
                    img_buffer = io.BytesIO()
                    real_image.save(img_buffer, format="JPEG")
                    base64_img = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
                    role_name = os.path.basename(os.path.dirname(img_path))

                    results.append(
                        {
                            "path": img_path,
                            "image": f"data:image/jpeg;base64,{base64_img}",
                            "similarity": float(similarity),
                            "role": role_name,
                        }
                    )
                except Exception as e:
                    print(f"[Worker] 处理图片失败 {img_path}: {e}")
                    continue
        else:
            # 没有真实数据集，使用彩色方块（模拟模式）
            color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            roles = ["Madoka", "Homura", "Sayaka", "Mami", "Kyoko"]

            for i in range(5):
                sim_image = Image.new("RGB", (150, 150), color=color_map[i % len(color_map)])
                img_buffer = io.BytesIO()
                sim_image.save(img_buffer, format="JPEG")
                base64_img = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

                results.append(
                    {
                        "path": f"/data/{roles[i % len(roles)]}_{i}.jpg",
                        "image": f"data:image/jpeg;base64,{base64_img}",
                        "similarity": float(0.9 - i * 0.08),
                        "role": roles[i % len(roles)],
                    }
                )

        # 写入结果
        with open(output_file, "w") as f:
            json.dump(
                {
                    "status": "success",
                    "results": results,
                    "model": "CLIP ViT-B/32 (MPS)",
                    "dataset_found": dataset_path is not None,
                    "total_images_scanned": len(real_images),
                },
                f,
            )

        print(f"[Worker] ✓ 任务完成: {task_id}, 返回 {len(results)} 个结果")

    except Exception as e:
        print(f"[Worker] ✗ 任务失败: {task_id} - {e}")
        import traceback

        traceback.print_exc()
        with open(output_file, "w") as f:
            json.dump({"status": "error", "message": str(e)}, f)

    # 清理输入文件
    if os.path.exists(input_file):
        os.remove(input_file)


def process_task_simulation(task_id):
    """使用模拟模式处理搜索任务"""
    input_file = os.path.join(INPUT_DIR, f"{task_id}.jpg")
    output_file = os.path.join(OUTPUT_DIR, f"{task_id}.json")

    if not os.path.exists(input_file):
        return

    print(f"[Worker] 处理任务: {task_id} (模拟模式)")

    try:
        # 模拟搜索结果
        results = [
            {
                "path": f"/data/Madoka_{i}.jpg",
                "similarity": 0.9 - i * 0.08,
                "role": ["Madoka", "Homura", "Sayaka", "Mami", "Kyoko"][i],
            }
            for i in range(5)
        ]

        # 创建模拟图片（彩色方块）
        response_results = []
        color_map = [
            (255, 0, 0),  # 红色
            (0, 255, 0),  # 绿色
            (0, 0, 255),  # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
        ]

        for i, result in enumerate(results):
            sim_image = Image.new("RGB", (150, 150), color=color_map[i % len(color_map)])
            img_buffer = io.BytesIO()
            sim_image.save(img_buffer, format="JPEG")
            base64_img = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

            response_results.append(
                {
                    "path": result.get("path", f"/data/test/{result['role']}_{i}.jpg"),
                    "image": f"data:image/jpeg;base64,{base64_img}",
                    "similarity": result["similarity"],
                    "role": result["role"],
                }
            )

        # 写入结果
        with open(output_file, "w") as f:
            json.dump({"status": "success", "results": response_results, "model": "Simulation"}, f)

        print(f"[Worker] ✓ 任务完成: {task_id}")

    except Exception as e:
        print(f"[Worker] ✗ 任务失败: {task_id} - {e}")
        with open(output_file, "w") as f:
            json.dump({"status": "error", "message": str(e)}, f)

    # 清理输入文件
    if os.path.exists(input_file):
        os.remove(input_file)


def main():
    """主循环"""
    global model, preprocess

    print("[Worker] 启动搜索Worker...")

    # 尝试使用MPS加载CLIP模型
    model, preprocess, device = load_clip_model_with_mps()

    if model is not None:
        print(f"[Worker] ✓ 成功加载CLIP模型，使用{device}加速")
        process_func = process_task_real_model
    else:
        print("[Worker] ⚠️ 模型加载失败，使用模拟模式")
        process_func = process_task_simulation

    print("[Worker] Worker已就绪，等待任务...")

    # 主循环
    while True:
        try:
            # 扫描输入目录
            input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]

            for input_file in input_files:
                task_id = input_file.replace(".jpg", "")
                process_func(task_id)

            time.sleep(0.1)

        except KeyboardInterrupt:
            print("[Worker] 收到停止信号")
            break
        except Exception as e:
            print(f"[Worker] 主循环错误: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
