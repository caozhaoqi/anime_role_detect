#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用增强版CharacterClassifier对所有角色进行分类
"""

import sys
import os

# 添加脚本所在目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CharacterClassifier_v2 import CharacterClassifier


def load_all_character_files():
    """
    读取所有角色文件中的角色名

    Returns:
        list: 所有角色名列表
    """
    # 定义所有角色文件路径
    character_files = [
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/1_genshin_chinese_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/1_p_top_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/3_star_rail_chinese_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/6_honkai3_chinese_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/0307_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/blda_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/cbjq_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/ht_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/ll_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/lsxy_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/mc_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/new_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/qlwh_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/zzz_spider_img_keyword.txt",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/spider_img_keyword.txt",
    ]

    all_characters = set()

    for file_path in character_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    characters = [line.strip() for line in f if line.strip()]
                    all_characters.update(characters)
                print(f"从 {os.path.basename(file_path)} 加载了 {len(characters)} 个角色")
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
        else:
            print(f"文件不存在: {file_path}")

    # 去重
    all_characters = list(all_characters)

    print(f"\n总共加载了 {len(all_characters)} 个唯一角色")
    return all_characters


def classify_all_characters():
    """
    读取所有角色文件中的角色，使用增强版CharacterClassifier进行分类
    """
    # 加载所有角色
    characters = load_all_character_files()

    if not characters:
        print("没有找到角色")
        return

    # 初始化增强版分类器
    classifier = CharacterClassifier(cache_file="character_cache.json")

    # 分类结果统计
    results = {
        "萝莉": 0,
        "可能具有部分萝莉特征": 0,
        "不属于萝莉": 0,
        "未找到角色": 0,
        "无明显特征": 0,
    }

    # 按分类存储角色
    classified_characters = {
        "萝莉": [],
        "可能具有部分萝莉特征": [],
        "不属于萝莉": [],
        "未找到角色": [],
        "无明显特征": [],
    }

    # 对每个角色进行分类
    for name in characters:
        result, tags, category = classifier.classify(name)
        print(result)
        print("-" * 40)

        # 统计结果
        results[category] += 1
        classified_characters[category].append((name, tags))

    # 输出统计结果
    print("=" * 80)
    print("分类结果统计:")
    for key, value in results.items():
        print(f"{key}: {value}")
    print("=" * 80)

    # 将结果输出为txt文件
    output_dir = (
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/classified_all_v2"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 输出总结果
    with open(
        os.path.join(output_dir, "all_classification_results.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("分类结果统计:\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        f.write("\n详细分类结果:\n")
        for category, chars in classified_characters.items():
            f.write(f"\n{category}:\n")
            for char_name, tags in chars:
                f.write(f"  {char_name} - 特征: {tags if tags else '无'}\n")

    # 按分类输出到不同文件
    for category, chars in classified_characters.items():
        if chars:
            with open(os.path.join(output_dir, f"{category}.txt"), "w", encoding="utf-8") as f:
                for char_name, tags in chars:
                    f.write(f"{char_name}\n")

    print(f"分类结果已输出到 {output_dir} 目录")
    print(f"缓存数据已保存到 character_cache.json")


if __name__ == "__main__":
    classify_all_characters()
