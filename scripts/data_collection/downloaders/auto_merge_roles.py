#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动定时合并中英文角色名脚本
按照 loli-role.txt 中的内容进行合并
"""
import os
import shutil
import time
import schedule
from pypinyin import pinyin, Style

# 配置
ROLE_LIST_FILE = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt"
DATASET_PATH = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset"
SPIDER_DATA_DIR = (
    "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url"
)


def load_role_list():
    """从角色名单文件中读取角色列表"""
    roles = {}
    if os.path.exists(ROLE_LIST_FILE):
        with open(ROLE_LIST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(" ")
                    if len(parts) >= 3:
                        cn_name = parts[0]
                        en_name = parts[2]
                        roles[cn_name] = en_name
    return roles


def generate_pinyin_variants(cn_name):
    """生成中文名字的各种拼音变体"""
    variants = set()

    # 带声调的拼音
    pinyin_with_tone = pinyin(cn_name, style=Style.TONE3)
    pinyin_str = "".join([p[0] for p in pinyin_with_tone])
    variants.add(pinyin_str)

    # 不带声调的拼音
    pinyin_without_tone = pinyin(cn_name, style=Style.NORMAL)
    pinyin_str_no_tone = "".join([p[0] for p in pinyin_without_tone])
    variants.add(pinyin_str_no_tone)

    return variants


def merge_pinyin_to_english():
    """合并拼音目录到英文目录"""
    roles = load_role_list()
    merged_count = 0
    moved_files = 0

    print(f"\n=== 开始自动合并 [{time.strftime('%Y-%m-%d %H:%M:%S')}] ===")
    print(f"读取到 {len(roles)} 个角色")

    for cn_name, en_name in roles.items():
        # 生成可能的拼音变体
        pinyin_variants = generate_pinyin_variants(cn_name)

        for pinyin_name in pinyin_variants:
            pinyin_dir = os.path.join(DATASET_PATH, pinyin_name)
            english_dir = os.path.join(DATASET_PATH, en_name)

            # 跳过自己映射到自己的情况
            if pinyin_name == en_name:
                continue

            # 检查拼音目录是否存在
            if os.path.isdir(pinyin_dir):
                # 确保英文目录存在
                if not os.path.isdir(english_dir):
                    os.makedirs(english_dir)
                    print(f"创建目录: {english_dir}")

                # 获取拼音目录中的所有图片文件
                files = [f for f in os.listdir(pinyin_dir) if f.lower().endswith(".jpg")]

                if files:
                    print(f"\n合并: {pinyin_name} -> {en_name} ({cn_name})")
                    print(f"  源目录文件数: {len(files)}")

                    for filename in files:
                        src_path = os.path.join(pinyin_dir, filename)
                        dst_path = os.path.join(english_dir, filename)

                        # 如果目标文件已存在，添加后缀
                        counter = 1
                        while os.path.exists(dst_path):
                            name, ext = os.path.splitext(filename)
                            dst_path = os.path.join(english_dir, f"{name}_{counter}{ext}")
                            counter += 1

                        shutil.move(src_path, dst_path)
                        moved_files += 1

                    # 删除空目录
                    if not os.listdir(pinyin_dir):
                        os.rmdir(pinyin_dir)
                        print(f"  删除空目录: {pinyin_dir}")

                    merged_count += 1
                    print(f"  成功移动 {len(files)} 个文件")

    print(f"\n=== 合并完成 ===")
    print(f"合并目录数: {merged_count}")
    print(f"移动文件数: {moved_files}")

    return merged_count, moved_files


def scheduled_merge():
    """定时合并任务"""
    try:
        merge_pinyin_to_english()
    except Exception as e:
        print(f"合并任务出错: {e}")


def run_scheduler(interval_minutes=30):
    """启动定时调度器"""
    print(f"启动自动合并调度器，每 {interval_minutes} 分钟执行一次")
    print(f"按 Ctrl+C 停止")

    # 立即执行一次
    merge_pinyin_to_english()

    # 设置定时任务
    schedule.every(interval_minutes).minutes.do(scheduled_merge)

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n定时调度器已停止")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="自动合并中英文角色名")
    parser.add_argument(
        "-s", "--schedule", action="store_true", help="启动定时调度器（每30分钟执行一次）"
    )
    parser.add_argument(
        "-i", "--interval", type=int, default=30, help="定时执行间隔（分钟），默认30分钟"
    )
    parser.add_argument("-o", "--once", action="store_true", help="仅执行一次合并（默认）")

    args = parser.parse_args()

    if args.schedule:
        run_scheduler(args.interval)
    else:
        merge_pinyin_to_english()
