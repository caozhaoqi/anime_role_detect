#!/usr/bin/env python3
"""
优化版蔚蓝档案角色图片采集脚本
为分类效果差的角色收集更多更准确的图片样本
"""
import os
import shutil

from tests.data.collect_test_data import collect_single_character_data


def collect_optimized_blue_archive(output_base_dir):
    """优化采集蔚蓝档案角色图片"""
    print("=== 优化版蔚蓝档案角色图片采集 ===")

    # 确保输出目录存在
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # 角色采集配置
    # 为分类效果差的角色增加采集数量
    characters_config = [
        # (角色名称, 采集数量, 优先级标签)
        (
            "蔚蓝档案_日奈",
            10,
            ["hina_(blue_archive)", "hina_(blue_archive)_solo", "blue_archive_hina"],
        ),
        (
            "蔚蓝档案_宫子",
            10,
            ["miyako_(blue_archive)", "miyako_(blue_archive)_solo", "blue_archive_miyako"],
        ),
        (
            "蔚蓝档案_星野",
            10,
            ["hoshino_(blue_archive)", "hoshino_(blue_archive)_solo", "blue_archive_hoshino"],
        ),
        (
            "蔚蓝档案_白子",
            8,
            ["shiroko_(blue_archive)", "shiroko_(blue_archive)_solo", "blue_archive_shiroko"],
        ),
        (
            "蔚蓝档案_阿罗娜",
            6,
            ["arona_(blue_archive)", "arona_(blue_archive)_solo", "blue_archive_arona"],
        ),
        (
            "蔚蓝档案_优花梨",
            6,
            ["yuuka_(blue_archive)", "yuuka_(blue_archive)_solo", "blue_archive_yuuka"],
        ),
    ]

    downloaded_characters = {}

    for character, image_limit, priority_tags in characters_config:
        output_dir = os.path.join(output_base_dir, character)

        # 清理旧目录
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        print(f"\n=== 采集角色 '{character}' 的图片 ===")
        print(f"目标: {image_limit} 张图片")
        print(f"优先级标签: {priority_tags}")

        # 尝试使用不同的标签组合
        total_downloaded = 0

        for tag in priority_tags:
            if total_downloaded >= image_limit:
                break

            print(f"\n尝试标签: {tag}")

            # 计算还需要下载的数量
            remaining = image_limit - total_downloaded

            # 使用自定义标签采集
            count = collect_single_character_data(tag, remaining, output_dir)
            total_downloaded += count

            print(f"  此标签下载: {count} 张，累计: {total_downloaded} 张")

        # 如果还不够，使用默认采集
        if total_downloaded < image_limit:
            remaining = image_limit - total_downloaded
            print(f"\n使用默认采集方法补充 {remaining} 张图片")
            count = collect_single_character_data(character, remaining, output_dir)
            total_downloaded += count

        if total_downloaded > 0:
            downloaded_characters[character] = {"directory": output_dir, "count": total_downloaded}
            print(f"✅ 成功采集 {total_downloaded} 张图片")
        else:
            print(f"❌ 无法采集图片")

    # 汇总报告
    print(f"\n=== 采集完成 ===")
    print(f"成功采集 {len(downloaded_characters)} 个角色的图片")
    print("\n采集结果:")
    for char, info in downloaded_characters.items():
        print(f"- {char}: {info['count']} 张图片")

    return downloaded_characters


def main():
    """主函数"""
    print("优化版蔚蓝档案角色图片采集脚本")
    print("============================")
    print("目标：为分类效果差的角色收集更多更准确的图片样本")

    # 配置
    output_base_dir = "data/blue_archive_optimized_v2"

    # 采集角色图片
    downloaded_characters = collect_optimized_blue_archive(output_base_dir)

    if downloaded_characters:
        print(f"\n🎉 采集完成！")
        print(f"所有图片已保存到: {output_base_dir}")
        print("\n下一步建议:")
        print("1. 运行 test_blue_archive_classification.py 脚本进行分类测试")
        print("2. 分析分类结果，进一步调整采集策略")
        print("3. 考虑添加更多角色或增加采集数量")
    else:
        print("\n❌ 采集失败，未下载到任何图片")


if __name__ == "__main__":
    main()
