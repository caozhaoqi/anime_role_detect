#!/usr/bin/env python3
"""删除截断的图片文件"""

import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

data_dir = Path('data/danbooru_images')
truncated_files = []

print("🔍 正在扫描截断图片...")

for img_file in data_dir.rglob('*'):
    if not img_file.is_file():
        continue
    
    if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        continue
    
    try:
        with Image.open(img_file) as img:
            img.verify()
    except Exception as e:
        if 'truncated' in str(e).lower() or 'truncated' in str(type(e).__name__).lower():
            truncated_files.append(img_file)
        else:
            # 其他错误也尝试删除
            truncated_files.append(img_file)

print(f"📊 找到 {len(truncated_files)} 个截断图片")

if truncated_files:
    print("\n🗑️  正在删除...")
    for f in truncated_files[:10]:
        print(f"  - {f}")
    if len(truncated_files) > 10:
        print(f"  ... 还有 {len(truncated_files) - 10} 个文件")
    
    for f in truncated_files:
        f.unlink()
    
    print(f"\n✅ 已删除 {len(truncated_files)} 个截断图片")
else:
    print("✅ 未发现截断图片")
