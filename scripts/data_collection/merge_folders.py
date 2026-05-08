#!/usr/bin/env python3
"""合并神乐的两个文件夹"""
import shutil
from pathlib import Path

def main():
    img_dir = Path('data/organized_images')
    
    # 神乐的两个文件夹
    src_folder = img_dir / 'shen1yue4'
    dest_folder = img_dir / 'shen2le4'
    
    if not src_folder.exists():
        print("❌ 源文件夹不存在")
        return
    
    if not dest_folder.exists():
        print("❌ 目标文件夹不存在")
        return
    
    # 统计文件数量
    src_count = len(list(src_folder.glob('*.jpg'))) + len(list(src_folder.glob('*.png'))) + len(list(src_folder.glob('*.webp')))
    dest_count = len(list(dest_folder.glob('*.jpg'))) + len(list(dest_folder.glob('*.png'))) + len(list(dest_folder.glob('*.webp')))
    
    print(f"📦 源文件夹 (shen1yue4): {src_count}张图片")
    print(f"📦 目标文件夹 (shen2le4): {dest_count}张图片")
    
    # 合并文件
    merged_count = 0
    skipped_count = 0
    
    for img_file in src_folder.glob('*'):
        if img_file.is_file():
            dest_file = dest_folder / img_file.name
            if dest_file.exists():
                # 文件名冲突，添加后缀
                name = img_file.stem
                ext = img_file.suffix
                counter = 1
                while dest_file.exists():
                    dest_file = dest_folder / f"{name}_{counter}{ext}"
                    counter += 1
                skipped_count += 1
            
            shutil.move(str(img_file), str(dest_file))
            merged_count += 1
    
    # 删除空文件夹
    shutil.rmtree(src_folder)
    
    # 最终统计
    final_count = len(list(dest_folder.glob('*.jpg'))) + len(list(dest_folder.glob('*.png'))) + len(list(dest_folder.glob('*.webp')))
    
    print(f"\n✅ 合并完成")
    print(f"   - 成功合并: {merged_count}张")
    print(f"   - 重命名避免冲突: {skipped_count}张")
    print(f"   - 神乐最终图片数: {final_count}张")
    print(f"   - 删除了空文件夹: shen1yue4")

if __name__ == '__main__':
    main()
