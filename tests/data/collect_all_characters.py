#!/usr/bin/env python3
"""
采集所有二次元角色数据脚本
支持多种游戏和动漫角色
"""
import os
import sys
import argparse
from time import sleep

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.collect_test_data import collect_smart_images

def collect_all_characters(limit, output_dir, api_key=None, user=None):
    """
    采集所有支持的二次元角色数据
    """
    # 游戏和动漫角色配置
    characters_config = [
        # 原神 (Genshin Impact)
        {
            "game": "原神",
            "game_en": "genshin_impact",
            "characters": [
                {"name": "荧", "tag": "lumine_(genshin_impact)"},
                {"name": "空", "tag": "aether_(genshin_impact)"},
                {"name": "琴", "tag": "jean_(genshin_impact)"},
                {"name": "丽莎", "tag": "lisa_(genshin_impact)"},
                {"name": "芭芭拉", "tag": "barbara_(genshin_impact)"},
                {"name": "温迪", "tag": "venti_(genshin_impact)"},
                {"name": "迪卢克", "tag": "diluc_(genshin_impact)"},
                {"name": "凯亚", "tag": "kaeya_(genshin_impact)"},
                {"name": "安柏", "tag": "amber_(genshin_impact)"},
                {"name": "雷泽", "tag": "razor_(genshin_impact)"},
            ]
        },
        # 鸣潮 (Wuthering Waves)
        {
            "game": "鸣潮",
            "game_en": "wuthering_waves",
            "characters": [
                {"name": "守岸人", "tag": "shorekeeper_(wuthering_waves)"},
                {"name": "椿", "tag": "camellya_(wuthering_waves)"},
                {"name": "卡提西亚", "tag": "katya_(wuthering_waves)"},
                {"name": "安比", "tag": "anby_(wuthering_waves)"},
                {"name": "比安卡", "tag": "bianca_(wuthering_waves)"},
                {"name": "科林", "tag": "corin_(wuthering_waves)"},
                {"name": "林", "tag": "lin_(wuthering_waves)"},
                {"name": "莉拉", "tag": "lyra_(wuthering_waves)"},
                {"name": "塞斯", "tag": "seth_(wuthering_waves)"},
            ]
        },
        # 绝区零 (Zenless Zone Zero)
        {
            "game": "绝区零",
            "game_en": "zenless_zone_zero",
            "characters": [
                {"name": "安比", "tag": "anby_(zenless_zone_zero)"},
                {"name": "夏娃", "tag": "eve_(zenless_zone_zero)"},
                {"name": "妮可", "tag": "nico_(zenless_zone_zero)"},
                {"name": "诺伊斯", "tag": "noise_(zenless_zone_zero)"},
                {"name": "雷文", "tag": "raven_(zenless_zone_zero)"},
                {"name": "杰克", "tag": "jack_(zenless_zone_zero)"},
            ]
        },
        # 崩坏三 (Honkai Impact 3rd)
        {
            "game": "崩坏三",
            "game_en": "honkai_impact_3rd",
            "characters": [
                {"name": "琪亚娜", "tag": "kiana_kaslana_(honkai_impact_3rd)"},
                {"name": "芽衣", "tag": "raiden_mei_(honkai_impact_3rd)"},
                {"name": "布洛妮娅", "tag": "bronya_zaychik_(honkai_impact_3rd)"},
                {"name": "符华", "tag": "fu_hua_(honkai_impact_3rd)"},
                {"name": "德丽莎", "tag": "theresa_apocalypse_(honkai_impact_3rd)"},
            ]
        },
        # 崩坏星穹铁道 (Honkai: Star Rail)
        {
            "game": "崩坏星穹铁道",
            "game_en": "honkai_star_rail",
            "characters": [
                {"name": "三月七", "tag": "march_7th_(honkai_star_rail)"},
                {"name": "丹恒", "tag": "dan_heng_(honkai_star_rail)"},
                {"name": "姬子", "tag": "himeko_(honkai_star_rail)"},
                {"name": "瓦尔特", "tag": "welt_(honkai_star_rail)"},
                {"name": "青雀", "tag": "qingque_(honkai_star_rail)"},
            ]
        },
        # 崩坏二 (Guns GirlZ / Honkai Impact)
        {
            "game": "崩坏二",
            "game_en": "honkai_impact",
            "characters": [
                {"name": "琪亚娜", "tag": "kiana_kaslana_(honkai_impact)"},
                {"name": "芽衣", "tag": "raiden_mei_(honkai_impact)"},
                {"name": "布洛妮娅", "tag": "bronya_zaychik_(honkai_impact)"},
            ]
        },
        # 幻塔 (Tower of Fantasy)
        {
            "game": "幻塔",
            "game_en": "tower_of_fantasy",
            "characters": [
                {"name": "莎莉", "tag": "sally_(tower_of_fantasy)"},
                {"name": "米娅", "tag": "mia_(tower_of_fantasy)"},
                {"name": "凛夜", "tag": "lin_(tower_of_fantasy)"},
            ]
        },
        # 明日方舟 (Arknights)
        {
            "game": "明日方舟",
            "game_en": "arknights",
            "characters": [
                {"name": "阿米娅", "tag": "amiya_(arknights)"},
                {"name": "能天使", "tag": "exusiai_(arknights)"},
                {"name": "德克萨斯", "tag": "texas_(arknights)"},
                {"name": "陈", "tag": "ch'en_(arknights)"},
                {"name": "银灰", "tag": "silver_ash_(arknights)"},
            ]
        },
        # 终末地 (The End Earth)
        {
            "game": "终末地",
            "game_en": "the_end_earth",
            "characters": [
                {"name": "艾琳", "tag": "irene_(the_end_earth)"},
                {"name": "露西亚", "tag": "lucia_(the_end_earth)"},
            ]
        },
        # 我推的孩子 (Oshi no Ko)
        {
            "game": "我推的孩子",
            "game_en": "oshi_no_ko",
            "characters": [
                {"name": "星野爱", "tag": "hoshino_ai_(oshi_no_ko)"},
                {"name": "星野瑠美衣", "tag": "hoshino_ruby_(oshi_no_ko)"},
                {"name": "星野泉", "tag": "hoshino_aqua_(oshi_no_ko)"},
            ]
        },
        # 间谍过家家 (Spy x Family)
        {
            "game": "间谍过家家",
            "game_en": "spy_x_family",
            "characters": [
                {"name": "阿尼亚", "tag": "anya_forger_(spy_x_family)"},
                {"name": "洛伊德", "tag": "loid_forger_(spy_x_family)"},
                {"name": "约尔", "tag": "yor_forger_(spy_x_family)"},
            ]
        },
        # 蔚蓝档案 (Blue Archive)
        {
            "game": "蔚蓝档案",
            "game_en": "blue_archive",
            "characters": [
                {"name": "星野", "tag": "hoshino_(blue_archive)"},
                {"name": "白子", "tag": "shiroko_(blue_archive)"},
                {"name": "阿罗娜", "tag": "arona_(blue_archive)"},
                {"name": "宫子", "tag": "miyako_(blue_archive)"},
                {"name": "日奈", "tag": "hina_(blue_archive)"},
                {"name": "优花梨", "tag": "yuuka_(blue_archive)"},
            ]
        },
    ]

    total_downloaded = 0
    total_characters = 0

    # 计算总角色数
    for config in characters_config:
        total_characters += len(config["characters"])

    print(f"=== 开始采集所有二次元角色数据 ===")
    print(f"总游戏数: {len(characters_config)}")
    print(f"总角色数: {total_characters}")
    print(f"每个角色目标采集: {limit} 张图片")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 遍历所有游戏和角色
    for game_config in characters_config:
        game = game_config["game"]
        game_en = game_config["game_en"]
        characters = game_config["characters"]

        print(f"\n>>> 开始采集游戏: {game} ({game_en})")
        print(f"角色数: {len(characters)}")

        for char_info in characters:
            char_name = char_info["name"]
            char_tag = char_info["tag"]
            
            # 为每个角色创建独立文件夹
            char_output_dir = os.path.join(output_dir, f"{game}_{char_name}")
            
            print(f"\n--- 处理角色: {char_name} (标签: {char_tag}) ---")
            print(f"输出目录: {char_output_dir}")
            
            try:
                downloaded = collect_smart_images(char_tag, limit, char_output_dir, api_key, user)
                total_downloaded += downloaded
                print(f"✓ 完成采集: {downloaded} 张图片")
            except Exception as e:
                print(f"✗ 采集失败: {e}")
            
            # 礼貌延时
            sleep(3)

    print("\n" + "=" * 60)
    print(f"=== 采集完成 ===")
    print(f"总下载图片数: {total_downloaded}")
    print(f"总角色数: {total_characters}")
    print(f"平均每个角色: {total_downloaded / total_characters:.1f} 张图片")
    print("=" * 60)

    return total_downloaded

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="采集所有二次元角色数据脚本")
    parser.add_argument("--limit", type=int, default=20, help="每个角色采集图片数量")
    parser.add_argument("--output_dir", default="data/all_characters", help="输出目录")
    parser.add_argument("--api_key", help="Danbooru API密钥 (可选)")
    parser.add_argument("--user", help="Danbooru用户名 (可选)")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 执行采集
    collect_all_characters(args.limit, args.output_dir, args.api_key, args.user)

if __name__ == "__main__":
    main()
