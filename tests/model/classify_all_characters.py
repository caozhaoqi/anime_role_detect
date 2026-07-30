#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CharacterClassifier
"""

import sys
import os

#  CharacterClassifier Python
_script_dir = os.path.dirname(os.path.abspath(__file__))

from CharacterClassifier import CharacterClassifier


def load_all_character_files():
    """
    

    Returns:
        list: 
    """
    # 
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
                print(f" {os.path.basename(file_path)}  {len(characters)} ")
            except Exception as e:
                print(f" {file_path} : {e}")
        else:
            print(f": {file_path}")

    # 
    all_characters = list(all_characters)

    print(f"\n {len(all_characters)} ")
    return all_characters


def classify_all_characters():
    """
    CharacterClassifier
    """
    # 
    characters = load_all_character_files()

    if not characters:
        print("")
        return

    # 
    classifier = CharacterClassifier()

    # 
    results = {"": 0, "": 0, "": 0, "": 0}

    # 
    classified_characters = {
        "": [],
        "": [],
        "": [],
        "": [],
    }

    # 
    for name in characters:
        result, tags, category = classifier.classify(name)
        print(result)
        print("-" * 40)

        # 
        results[category] += 1
        classified_characters[category].append((name, tags))

    # 
    print("=" * 80)
    print(":")
    for key, value in results.items():
        print(f"{key}: {value}")
    print("=" * 80)

    # txt
    output_dir = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/classified_all"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 
    with open(
        os.path.join(output_dir, "all_classification_results.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(":\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        f.write("\n:\n")
        for category, chars in classified_characters.items():
            f.write(f"\n{category}:\n")
            for char_name, tags in chars:
                f.write(f"  {char_name} - : {tags if tags else ''}\n")

    # 
    for category, chars in classified_characters.items():
        if chars:
            with open(os.path.join(output_dir, f"{category}.txt"), "w", encoding="utf-8") as f:
                for char_name, tags in chars:
                    f.write(f"{char_name}\n")

    print(f" {output_dir} ")


if __name__ == "__main__":
    classify_all_characters()
