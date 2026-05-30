#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多角色检测脚本
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.detection.multi_role_detection import MultiRoleDetector


def detect_multi_roles(image_path):
    """
    检测图像中的多个角色

    Args:
        image_path: 图像路径
    """
    print(f"🔍 开始检测图像: {image_path}")

    # 创建检测器
    detector = MultiRoleDetector(model_name="efficientnet_b0")

    # 执行检测
    results = detector.detect_roles(image_path)

    # 输出结果
    print(f"\n✅ 检测完成！")
    print(f"检测到 {len(results)} 个角色")
    print("-" * 60)

    for i, result in enumerate(results):
        print(f"\n角色 {i+1}:")
        print(f"  🎭 角色名: {result['role']}")
        print(f"  📊 相似度: {result['similarity']:.4f}")
        print(f"  📦 置信度: {result['confidence']:.4f}")
        print(
            f"  📏 边界框: x1={result['bbox']['x1']}, y1={result['bbox']['y1']}, x2={result['bbox']['x2']}, y2={result['bbox']['y2']}"
        )

        if result["attributes"]:
            print(f"  🏷️  属性标签: {', '.join(result['attributes'][:5])}")

    return results


def batch_detect(image_dir):
    """
    批量检测目录中的图像

    Args:
        image_dir: 图像目录路径
    """
    image_dir = Path(image_dir)

    if not image_dir.exists():
        print(f"❌ 目录不存在: {image_dir}")
        return

    # 获取所有图像文件
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = []

    for ext in image_extensions:
        image_files.extend(image_dir.rglob(f"*{ext}"))

    print(f"📂 找到 {len(image_files)} 张图像")

    # 创建检测器
    detector = MultiRoleDetector(model_name="efficientnet_b0")

    # 统计结果
    total_roles = 0
    multi_role_count = 0

    # 批量检测
    for i, image_path in enumerate(image_files[:10]):  # 只检测前10张作为示例
        print(f"\n{'='*60}")
        print(f"处理图像 {i+1}/{min(10, len(image_files))}: {image_path.name}")
        print("=" * 60)

        try:
            results = detector.detect_roles(str(image_path))
            total_roles += len(results)

            if len(results) > 1:
                multi_role_count += 1
                print(f"⚠️  检测到多个角色 ({len(results)}个)")

            for j, result in enumerate(results):
                print(f"  角色 {j+1}: {result['role']} (相似度: {result['similarity']:.2f})")

        except Exception as e:
            print(f"❌ 检测失败: {e}")

    # 输出统计
    print(f"\n{'='*60}")
    print("📊 检测统计")
    print("=" * 60)
    print(f"检测图像数: {min(10, len(image_files))}")
    print(f"检测到角色总数: {total_roles}")
    print(f"含多角色的图像数: {multi_role_count}")
    print(f"平均每图角色数: {total_roles / min(10, len(image_files)):.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多角色检测工具")
    parser.add_argument("--image", type=str, help="单张图像路径")
    parser.add_argument("--dir", type=str, help="图像目录路径（批量检测）")

    args = parser.parse_args()

    if args.image:
        detect_multi_roles(args.image)
    elif args.dir:
        batch_detect(args.dir)
    else:
        print("❌ 请指定 --image 或 --dir 参数")
        parser.print_help()
