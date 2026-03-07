#!/usr/bin/env python3
"""
蔚蓝档案角色图片采集脚本
使用标准Booru标签格式，确保下载到准确的角色图片
"""
import os
import sys
import shutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.collect_test_data import collect_single_character_data

def collect_blue_archive_characters(output_base_dir, image_limit=6):
    """采集蔚蓝档案角色图片"""
    print("=== 蔚蓝档案角色图片采集 ===")
    
    # 确保输出目录存在
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # 蔚蓝档案角色列表（使用标准格式）
    blue_archive_characters = [
        "蔚蓝档案_星野",      # 星野
        "蔚蓝档案_白子",      # 白子
        "蔚蓝档案_一之濑明日奈", # 一之濑明日奈
        "蔚蓝档案_黑子",      # 黑子
        "蔚蓝档案_阿罗娜",     # 阿罗娜
        "蔚蓝档案_宫子",      # 宫子
        "蔚蓝档案_日奈",      # 日奈
        "蔚蓝档案_优花梨",     # 优花梨
    ]
    
    downloaded_characters = {}
    
    for character in blue_archive_characters:
        output_dir = os.path.join(output_base_dir, character)
        
        # 清理旧目录
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        print(f"\n=== 采集角色 '{character}' 的图片 ===")
        print(f"目标: {image_limit} 张图片")
        
        # 使用优化后的采集函数
        count = collect_single_character_data(character, image_limit, output_dir)
        
        if count > 0:
            downloaded_characters[character] = {
                "directory": output_dir,
                "count": count
            }
            print(f"✅ 成功采集 {count} 张图片")
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
    print("蔚蓝档案角色图片采集脚本")
    print("====================")
    
    # 配置
    output_base_dir = "data/blue_archive_optimized"
    image_limit = 6  # 每个角色采集6张图片
    
    # 采集角色图片
    downloaded_characters = collect_blue_archive_characters(output_base_dir, image_limit)
    
    if downloaded_characters:
        print(f"\n🎉 采集完成！")
        print(f"所有图片已保存到: {output_base_dir}")
        print("\n下一步建议:")
        print("1. 检查采集的图片是否与角色匹配")
        print("2. 运行分类测试脚本评估效果")
        print("3. 如有需要，调整image_limit参数采集更多图片")
    else:
        print("\n❌ 采集失败，未下载到任何图片")

if __name__ == "__main__":
    main()
