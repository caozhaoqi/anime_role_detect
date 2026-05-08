#!/usr/bin/env python3
"""检查未匹配的文件夹是否为新角色"""
from pathlib import Path
import sys
sys.path.insert(0, 'spider_image_system/src/run')
from constants import PINYIN_MAPPING

def main():
    # 未匹配的文件夹
    new_folders = {
        'luo4ke3ke3': '洛可可',
        'mi2dou4zi': '弥豆子',
        'shen2le4': '神乐'
    }
    
    print("检查未匹配的文件夹:")
    print("=" * 60)
    
    # 检查这些拼音是否已存在于映射中
    for pinyin, guess_name in new_folders.items():
        exists = False
        for name, py in PINYIN_MAPPING.items():
            if py == pinyin:
                print(f"  • {pinyin} -> {name} (已在映射中)")
                exists = True
                break
        if not exists:
            print(f"  • {pinyin} -> {guess_name} (新角色，需要添加)")
    
    print("\n" + "=" * 60)
    print("建议操作:")
    print("1. 将这些新角色添加到角色名单")
    print("2. 更新拼音映射")
    
    # 统计图片数量
    img_dir = Path('data/organized_images')
    print("\n图片数量统计:")
    for pinyin, name in new_folders.items():
        folder = img_dir / pinyin
        if folder.exists():
            count = len(list(folder.glob('*.jpg'))) + len(list(folder.glob('*.png'))) + len(list(folder.glob('*.webp')))
            print(f"  • {name} ({pinyin}): {count}张")
    
    # 添加到名单
    add_to_list = input("\n是否添加这些角色到名单? (y/n): ")
    if add_to_list.lower() == 'y':
        # 读取现有名单
        with open('auto_spider_img/loli-role.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 添加新角色
        new_entries = [
            '罗可可 鸣潮 Roccia ロchia\n',
            '蜜豆子 鬼灭之刃 Mitsu ミツ\n',
            '神乐 阴阳师 Kagura カグラ\n'
        ]
        
        with open('auto_spider_img/loli-role.txt', 'w', encoding='utf-8') as f:
            f.writelines(lines)
            f.writelines(new_entries)
        
        print("✅ 已添加到角色名单")
        
        # 更新拼音映射
        with open('spider_image_system/src/run/constants.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 在 PINYIN_MAPPING 字典中添加新映射
        insert_pos = content.find("'菲米莉丝': 'fei1mi3li4si1',") + len("'菲米莉丝': 'fei1mi3li4si1',")
        new_mappings = "\n    '罗可可': 'luo4ke3ke3',\n    '蜜豆子': 'mi2dou4zi',\n    '神乐': 'shen2le4',"
        new_content = content[:insert_pos] + new_mappings + content[insert_pos:]
        
        with open('spider_image_system/src/run/constants.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ 已更新拼音映射")

if __name__ == '__main__':
    main()
