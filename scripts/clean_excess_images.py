#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理超过300张图片的角色目录，保留最多300张
"""

import os


def clean_excess_images(dataset_path, max_count=300):
    """
    清理超过max_count的图片

    :param dataset_path: 数据集目录
    :param max_count: 最大保留数量
    """
    total_deleted = 0
    cleaned_roles = []

    for dir_name in sorted(os.listdir(dataset_path)):
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.isdir(dir_path):
            continue

        # 获取所有jpg图片
        images = [f for f in os.listdir(dir_path) if f.lower().endswith(".jpg")]
        images.sort()  # 按名称排序，保留序号小的

        count = len(images)
        if count > max_count:
            excess = count - max_count
            # 删除多余的图片（保留前max_count张）
            to_delete = images[max_count:]

            print(f"{dir_name}: {count} -> {max_count} (删除 {excess} 张)")

            for img in to_delete:
                img_path = os.path.join(dir_path, img)
                try:
                    os.remove(img_path)
                    total_deleted += 1
                except Exception as e:
                    print(f"  删除失败 {img}: {e}")

            cleaned_roles.append((dir_name, count, max_count, excess))

    return cleaned_roles, total_deleted


def main():
    dataset_path = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset"
    max_count = 300

    print("=" * 60)
    print(f"清理超过 {max_count} 张图片的角色")
    print("=" * 60)

    cleaned_roles, total_deleted = clean_excess_images(dataset_path, max_count)

    print("=" * 60)
    print(f"清理完成！")
    print(f"处理角色数: {len(cleaned_roles)}")
    print(f"删除图片总数: {total_deleted}")
    print("=" * 60)

    if cleaned_roles:
        print("\\n详细清理记录:")
        for role, before, after, deleted in cleaned_roles:
            print(f"  {role}: {before} -> {after} (-{deleted})")


if __name__ == "__main__":
    main()
