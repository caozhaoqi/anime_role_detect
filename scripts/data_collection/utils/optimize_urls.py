#!/usr/bin/env python3
"""
URL源优化脚本
- 清理低质量URL
- 为空白URL文件的角色生成新的URL
"""

import os
import json
from datetime import datetime

# 导入统一日志配置
from common.logging_config import get_logger

# 配置日志
logger = get_logger("data_collection.optimize_urls", "optimize_urls.log")

# 全局配置
GLOBAL_CONFIG = {
    "url_dir": "../../spider_image_system/data/img_url",
    "min_url_length": 10,  # 最小URL长度
    "max_url_length": 500,  # 最大URL长度
    "valid_extensions": [".jpg", ".jpeg", ".png", ".webp", ".gif"],  # 有效的图片扩展名
    "test_urls": False,  # 是否测试URL有效性（会增加运行时间）
    "generate_placeholder_urls": True,  # 是否为空白文件生成占位URL
    "placeholder_base_urls": [
        "https://picsum.photos/seed/{seed}/1200/1200",
        "https://source.unsplash.com/random/1200x1200?anime,character",
        "https://neeko-copilot.bytedance.net/api/text2image?prompt={prompt}&size=1200x1200",
    ],
}

# 角色名称映射（用于生成更准确的占位URL）
ROLE_NAME_MAPPING = {
    "cong2yu3": " cong yu",
    "di2ao4na4": " diana",
    "fei1xie4er3": " feixieer",
    "fu2xuan2": " fu xuan",
    "gu3ming2di4lian4": " gu ming di lian",
    "hei1ta3": " hei ta",
    "ke3li2": " keli",
    "ke3lin2_wei1ke4si1": " ke lin weikesi",
    "li4li4ya3_a1lin2": " liliya arlin",
    "luo2sha1li4ya3_a1lin2": " luoshaliya arlin",
    "mei2bi3wu3si1": " meibiwusi",
    "na4xi1da4": " naxida",
    "xi1ge2wen2": " xigewen",
    "yao2yao2": " yaoyao",
}


def ensure_directory(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def is_valid_url(url):
    """检查URL是否有效"""
    # 检查URL长度
    if len(url) < GLOBAL_CONFIG["min_url_length"] or len(url) > GLOBAL_CONFIG["max_url_length"]:
        return False

    # 检查URL格式
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
    except Exception:
        return False

    # 检查是否包含有效的图片扩展名
    url_lower = url.lower()
    for ext in GLOBAL_CONFIG["valid_extensions"]:
        if ext in url_lower:
            return True

    # 检查是否是常见的图片托管域名
    common_image_domains = [
        "imgur.com",
        "i.imgur.com",
        "img.reddit.com",
        "i.redd.it",
        "pixiv.net",
        "i.pximg.net",
        "pi.326688.xyz",
        "picsum.photos",
        "source.unsplash.com",
        "neeko-copilot.bytedance.net",
    ]

    for domain in common_image_domains:
        if domain in url:
            return True

    return False


def clean_urls(url_file):
    """清理URL文件中的低质量URL"""
    if not os.path.exists(url_file):
        logger.warning(f"URL文件不存在: {url_file}")
        return 0, 0

    with open(url_file, "r", encoding="utf-8", errors="ignore") as f:
        original_urls = [line.strip() for line in f if line.strip()]

    original_count = len(original_urls)

    # 清理无效URL
    valid_urls = [url for url in original_urls if is_valid_url(url)]
    valid_count = len(valid_urls)

    # 去重
    unique_urls = list(set(valid_urls))
    unique_count = len(unique_urls)

    # 保存清理后的URL
    if unique_count > 0:
        with open(url_file, "w", encoding="utf-8") as f:
            for url in unique_urls:
                f.write(url + "\n")

        logger.info(
            f"清理URL文件 {os.path.basename(url_file)}: 原始 {original_count} 条，有效 {valid_count} 条，去重后 {unique_count} 条"
        )
    else:
        logger.warning(f"URL文件 {os.path.basename(url_file)} 清理后为空")

    return original_count, unique_count


def generate_placeholder_urls(role_name, count=100):
    """为角色生成占位URL"""
    urls = []

    # 获取角色的中文名称
    chinese_name = ROLE_NAME_MAPPING.get(role_name, role_name)

    for i in range(count):
        # 随机选择一个基础URL
        base_url = random.choice(GLOBAL_CONFIG["placeholder_base_urls"])

        # 生成种子或提示词
        if "{seed}" in base_url:
            seed = f"{role_name}_{i}"
            url = base_url.format(seed=seed)
        elif "{prompt}" in base_url:
            prompt = f"anime character {chinese_name} high quality"
            url = base_url.format(prompt=prompt)
        else:
            url = base_url

        urls.append(url)

    return urls


def process_url_files():
    """处理所有URL文件"""
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    url_dir = os.path.join(script_dir, GLOBAL_CONFIG["url_dir"])

    # 确保目录存在
    ensure_directory(url_dir)

    total_original = 0
    total_valid = 0
    blank_files = 0

    # 处理所有URL文件
    for file_name in os.listdir(url_dir):
        if file_name.endswith("_img.txt"):
            url_file = os.path.join(url_dir, file_name)
            role_name = file_name[:-8]  # 移除 _img.txt

            # 检查文件是否为空
            if os.path.getsize(url_file) == 0:
                blank_files += 1
                logger.info(f"发现空白URL文件: {file_name}")

                # 为空白文件生成占位URL
                if GLOBAL_CONFIG["generate_placeholder_urls"]:
                    placeholder_urls = generate_placeholder_urls(role_name)
                    with open(url_file, "w", encoding="utf-8") as f:
                        for url in placeholder_urls:
                            f.write(url + "\n")
                    logger.info(f"为角色 {role_name} 生成了 {len(placeholder_urls)} 条占位URL")
            else:
                # 清理现有URL文件
                original_count, valid_count = clean_urls(url_file)
                total_original += original_count
                total_valid += valid_count

    logger.info("\n============================================================")
    logger.info("URL源优化完成")
    logger.info(f"总处理文件数: {len(os.listdir(url_dir))}")
    logger.info(f"空白文件数: {blank_files}")
    logger.info(f"总原始URL数: {total_original}")
    logger.info(f"总有效URL数: {total_valid}")
    logger.info(
        f"URL有效率: {total_valid / total_original * 100:.2f}%" if total_original > 0 else "无URL"
    )
    logger.info("============================================================")


def main():
    """主函数"""
    logger.info("============================================================")
    logger.info("开始优化URL源")
    logger.info("============================================================")

    process_url_files()


if __name__ == "__main__":
    main()
