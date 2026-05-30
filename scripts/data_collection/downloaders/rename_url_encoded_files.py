#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将URL编码的中文文件名重命名为拼音格式
"""

import os
import urllib.parse
from pypinyin import lazy_pinyin, Style

# 配置
URL_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url"


def rename_url_encoded_files():
    """重命名URL编码的文件"""
    renamed_count = 0
    skipped_count = 0

    print("=== 开始重命名URL编码的文件 ===")

    for filename in os.listdir(URL_DIR):
        # 检查是否是URL编码的文件（包含%且以_img.txt结尾）
        if "%" in filename and filename.endswith("_img.txt"):
            try:
                # 解码URL编码的文件名
                decoded_name = urllib.parse.unquote(filename)
                print(f"\n发现URL编码文件: {filename}")
                print(f"  解码后: {decoded_name}")

                # 提取角色中文名（去掉_img.txt后缀）
                chinese_name = decoded_name.replace("_img.txt", "")
                print(f"  角色名: {chinese_name}")

                # 转换为拼音格式
                pinyin = "".join(lazy_pinyin(chinese_name, style=Style.TONE3))
                new_filename = f"{pinyin}_img.txt"
                print(f"  新文件名: {new_filename}")

                # 检查新文件名是否已存在
                old_path = os.path.join(URL_DIR, filename)
                new_path = os.path.join(URL_DIR, new_filename)

                if os.path.exists(new_path):
                    print(f"  ⚠️ 目标文件已存在，跳过")
                    skipped_count += 1
                    continue

                # 重命名文件
                os.rename(old_path, new_path)
                print(f"  ✅ 重命名成功")
                renamed_count += 1

            except Exception as e:
                print(f"  ❌ 重命名失败: {e}")
                skipped_count += 1

    print(f"\n=== 重命名完成 ===")
    print(f"已重命名: {renamed_count} 个文件")
    print(f"已跳过: {skipped_count} 个文件")


if __name__ == "__main__":
    rename_url_encoded_files()
