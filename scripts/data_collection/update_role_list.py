#!/usr/bin/env python3
"""核对并更新角色名单"""
from pathlib import Path
import sys
sys.path.insert(0, 'spider_image_system/src/run')
from constants import PINYIN_MAPPING

def main():
    
    # 读取角色名单
    with open('auto_spider_img/loli-role.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 获取实际文件夹
    img_dir = Path('data/organized_images')
    actual_folders = set()
    for item in img_dir.iterdir():
        if item.is_dir() and item.name not in ['trash', 'trash_nsfw', 'trash_multi_face', '其他', '.', '..']:
            actual_folders.add(item.name)
    
    # 分析额外文件夹
    print("分析额外文件夹:")
    extra = ['luo4ke3ke3', 'mi2dou4zi', 'shen2le4', '克萝萝', '小闪', '月千夜', '洛可可', '爱丽儿', '芙丽希娅', '釉壶']
    
    duplicates = []  # 重复的文件夹（已有对应拼音/中文）
    new_chinese = []  # 中文命名但不在名单中的
    
    for folder in extra:
        if folder[0].islower():
            # 拼音文件夹
            matched = False
            for name, pinyin in PINYIN_MAPPING.items():
                if pinyin == folder:
                    print(f'  • {folder} = {name} (已在名单中)')
                    duplicates.append((folder, name))
                    matched = True
                    break
            if not matched:
                print(f'  • {folder} = 未匹配到中文名称')
        else:
            # 中文文件夹
            pinyin = PINYIN_MAPPING.get(folder)
            if pinyin:
                print(f'  • {folder} = {pinyin} (已在名单中，中文命名)')
                duplicates.append((folder, pinyin))
            else:
                print(f'  • {folder} = 不在拼音映射中')
                new_chinese.append(folder)
    
    print("\n结论:")
    print(f"  - 重复文件夹: {len(duplicates)}个")
    print(f"  - 新中文文件夹: {len(new_chinese)}个")
    
    # 询问用户是否删除重复文件夹
    if duplicates:
        print("\n重复文件夹列表:")
        for folder, original in duplicates:
            print(f"  • {folder} -> {original}")
        
        # 删除重复文件夹（中文命名的文件夹应该保留拼音命名的）
        print("\n删除重复的中文命名文件夹...")
        for folder, original in duplicates:
            if folder[0].islower():
                # 拼音文件夹保留，检查是否有对应的中文文件夹
                chinese_folder = None
                for name, pinyin in PINYIN_MAPPING.items():
                    if pinyin == folder:
                        chinese_folder = name
                        break
                if chinese_folder and (img_dir / chinese_folder).exists():
                    print(f"  删除 {chinese_folder} (保留 {folder})")
                    # 删除中文文件夹
                    import shutil
                    shutil.rmtree(img_dir / chinese_folder)
            else:
                # 中文文件夹，检查是否有对应的拼音文件夹
                pinyin_folder = PINYIN_MAPPING.get(folder)
                if pinyin_folder and (img_dir / pinyin_folder).exists():
                    print(f"  删除 {folder} (保留 {pinyin_folder})")
                    import shutil
                    shutil.rmtree(img_dir / folder)
    
    print("\n✅ 角色名单核对完成")

if __name__ == '__main__':
    main()
