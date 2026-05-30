#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Tagger - 集成标签模型
输出高质量的动漫图像标签
"""

import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 需要过滤的不当标签
FILTERED_TAGS = {
    "bone",
    "bone nail",
    "bone nails",
    "skeleton",
    "skull",
    "gore",
    "blood",
    "violence",
    "death",
    "nsfw",
    "nudity",
    "explicit",
    "porn",
    "hentai",
    "offensive",
    "hateful",
    "racist",
    "sexist",
    "self-harm",
    "suicide",
    "depression",
    "drug",
    "alcohol",
    "smoking",
    "cigarette",
    "weapon",
    "gun",
    "knife",
    "sword",
    "explosion",
    "rifle",
    "spear",
    "bow",
    "arrow",
}

# 安全的角色特征标签
SAFE_CHARACTER_FEATURES = {
    "Tsukiyo": ["blue hair", "long hair", "blue eyes", "school uniform", "serafuku", "calm"],
    "Hina": ["pink hair", "long hair", "pink eyes", "school uniform", "gentle", "smile"],
    "Madoka": ["pink hair", "twintails", "pink eyes", "magical girl", "pink dress"],
    "Homura": ["black hair", "long hair", "purple eyes", "school uniform", "serious"],
    "Sayaka": ["blue hair", "ponytail", "blue eyes", "magical girl"],
    "Mami": ["blonde hair", "twin drills", "yellow eyes", "magical girl"],
    "Kyoko": ["red hair", "ponytail", "orange eyes", "magical girl"],
    "Arona": ["blue hair", "short hair", "blue eyes", "school uniform", "robot", "halo"],
    "Shiroko": ["white hair", "short hair", "blue eyes", "school uniform"],
    "Default": ["anime", "character", "portrait"],
}


def is_inappropriate(tag):
    """检查标签是否不当"""
    tag_lower = tag.lower()
    for forbidden in FILTERED_TAGS:
        if forbidden in tag_lower:
            return True
    return False


class AdvancedTagger:
    """高级标签生成器"""

    def __init__(self):
        self.tagger = None
        self.models_loaded = False

    def load_models(self):
        """加载标签模型"""
        print("📦 加载标签模型...")

        # 加载WD Vit Tagger
        try:
            from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger

            self.tagger = WDViTV3Tagger()
            self.tagger.load_model()
            print("✅ WD Vit Tagger 加载成功")
        except Exception as e:
            print(f"⚠️ WD Vit Tagger 加载失败: {e}")

        self.models_loaded = True
        print("📋 模型加载完成")

    def generate_tags(self, image_path, role_name="Default", threshold=0.3):
        """生成图像标签"""
        all_tags = set()

        # 使用WD Vit Tagger
        if self.tagger:
            try:
                wd_tags = self.tagger.generate_tags(image_path, threshold=threshold)
                for tag_info in wd_tags:
                    tag = tag_info["tag"]
                    if not is_inappropriate(tag):
                        all_tags.add(tag)
            except Exception as e:
                print(f"⚠️ 生成标签失败: {e}")

        # 添加角色特征标签
        if role_name in SAFE_CHARACTER_FEATURES:
            for safe_tag in SAFE_CHARACTER_FEATURES[role_name]:
                if not is_inappropriate(safe_tag):
                    all_tags.add(safe_tag)

        # 添加默认标签
        for safe_tag in SAFE_CHARACTER_FEATURES["Default"]:
            all_tags.add(safe_tag)

        return list(all_tags)


def process_data_directory(data_dir, output_dir, tagger):
    """处理数据目录并生成标签"""
    os.makedirs(output_dir, exist_ok=True)

    all_tags = {}
    processed_count = 0
    failed_count = 0

    print(f"\n📁 处理目录: {data_dir}")

    for role_name in tqdm(os.listdir(data_dir), desc="处理角色"):
        role_dir = os.path.join(data_dir, role_name)
        if not os.path.isdir(role_dir):
            continue

        out_role_dir = os.path.join(output_dir, role_name)
        os.makedirs(out_role_dir, exist_ok=True)

        role_tags = {}

        for filename in os.listdir(role_dir):
            if not filename.lower().endswith((".jpg", ".png", ".webp", ".jpeg")):
                continue

            in_path = os.path.join(role_dir, filename)
            out_filename = os.path.splitext(filename)[0] + ".jpg"
            out_path = os.path.join(out_role_dir, out_filename)

            try:
                Image.open(in_path).convert("RGB").save(out_path, "JPEG", quality=95)
                tags = tagger.generate_tags(out_path, role_name)
                role_tags[out_filename] = tags
                processed_count += 1
            except Exception as e:
                print(f"❌ 处理失败 {in_path}: {e}")
                failed_count += 1

        all_tags[role_name] = role_tags

    # 保存标签文件
    tags_file = os.path.join(output_dir, "image_tags.json")
    with open(tags_file, "w", encoding="utf-8") as f:
        json.dump(all_tags, f, ensure_ascii=False, indent=2)

    # 生成标签统计报告
    stats_file = os.path.join(output_dir, "tag_statistics.json")
    stats = generate_statistics(all_tags)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return processed_count, failed_count, tags_file, stats_file


def generate_statistics(tag_data):
    """生成标签统计信息"""
    all_tags = {}
    role_tag_counts = {}
    total_images = 0

    for role_name, images in tag_data.items():
        total_images += len(images)
        role_tag_counts[role_name] = len(images)

        for tags in images.values():
            for tag in tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1

    sorted_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)

    return {
        "total_images": total_images,
        "total_roles": len(tag_data),
        "total_unique_tags": len(all_tags),
        "role_image_counts": role_tag_counts,
        "top_tags": [(tag, count) for tag, count in sorted_tags[:30]],
        "tag_frequency": dict(sorted_tags),
    }


def main():
    parser = argparse.ArgumentParser(description="Advanced Tagger - Enhanced Tagging")
    parser.add_argument(
        "--data-dir", type=str, default="./data/merged_dataset", help="输入数据目录"
    )
    parser.add_argument("--output-dir", type=str, default="./data_tagged", help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.3, help="置信度阈值")
    args = parser.parse_args()

    print("🚀 Advanced Tagger - Enhanced Tagging")
    print("=" * 60)

    tagger = AdvancedTagger()
    tagger.load_models()

    processed, failed, tags_file, stats_file = process_data_directory(
        args.data_dir, args.output_dir, tagger
    )

    print(f"\n✅ 处理完成!")
    print(f"   成功: {processed} 张")
    print(f"   失败: {failed} 张")
    print(f"   标签文件: {tags_file}")
    print(f"   统计文件: {stats_file}")

    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)

    print(f"\n📊 标签统计:")
    print(f"   图片总数: {stats['total_images']}")
    print(f"   角色总数: {stats['total_roles']}")
    print(f"   标签种类: {stats['total_unique_tags']}")
    print(f"\n🏷️ 前20个高频标签:")
    for tag, count in stats["top_tags"][:20]:
        print(f"   {tag}: {count} 次")


if __name__ == "__main__":
    main()
