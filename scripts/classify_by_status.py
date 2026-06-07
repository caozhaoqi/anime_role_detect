#!/usr/bin/env python3
"""根据数据库分类将图片文件分类到对应目录（使用硬链接节省空间）"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.database.init_db import init_database, Sample

def main():
    # 初始化数据库
    engine, Session = init_database()
    session = Session()
    
    # 原始数据目录
    source_dir = Path('data/danbooru_images')
    
    # 分类目标目录
    output_base = Path('data/classified_images')
    
    # 分类目录映射
    category_dirs = {
        'annotated': output_base / 'annotated',
        'no_detection': output_base / 'no_detection',
        'duplicate': output_base / 'duplicate',
        'filtered_quality': output_base / 'filtered_quality',
        'filtered_non_anime': output_base / 'filtered_non_anime'
    }
    
    # 创建目录
    for dir_path in category_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 统计
    category_counts = {cat: 0 for cat in category_dirs.keys()}
    skipped_existing = 0
    
    # 查询所有样本
    samples = session.query(Sample).all()
    
    print(f"📊 开始分类 {len(samples)} 个样本...")
    
    for i, sample in enumerate(samples, 1):
        # 获取源文件路径
        src_path = Path(sample.image_path)
        
        if not src_path.exists():
            print(f"⚠️ 文件不存在: {src_path}")
            continue
        
        # 确定目标目录
        status = sample.status
        if status not in category_dirs:
            print(f"⚠️ 未知状态: {status} - {sample.image_path}")
            continue
        
        # 目标路径
        dst_dir = category_dirs[status]
        dst_path = dst_dir / src_path.name
        
        # 如果目标文件已存在，跳过（硬链接不需要编号）
        if dst_path.exists():
            skipped_existing += 1
            continue
        
        # 创建硬链接
        try:
            os.link(src_path, dst_path)
            category_counts[status] += 1
        except Exception as e:
            print(f"⚠️ 创建硬链接失败 {src_path}: {e}")
        
        # 进度显示
        if i % 500 == 0:
            print(f"🚀 已处理 {i}/{len(samples)} 个样本")
    
    # 输出统计
    print("\n📋 分类完成！")
    print("=" * 50)
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} 个")
    print(f"  跳过已存在: {skipped_existing} 个")
    print("=" * 50)
    
    session.close()

if __name__ == '__main__':
    main()