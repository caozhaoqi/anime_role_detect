#!/usr/bin/env python3
"""导入样本图片到数据库"""

import os
import sys
from pathlib import Path
from PIL import Image

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.database.init_db import init_database, Sample, Character


def import_samples_from_directory(data_dir: str = "data/final_dataset"):
    """从目录导入样本图片"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return

    # 初始化数据库
    engine, Session = init_database()
    session = Session()

    imported_count = 0
    skipped_count = 0

    # 遍历每个角色目录
    for character_dir in data_path.iterdir():
        if not character_dir.is_dir():
            continue
        
        character_name = character_dir.name
        
        # 查找对应的角色
        character = session.query(Character).filter_by(name=character_name).first()
        if not character:
            print(f"⚠️ 未找到角色: {character_name}，跳过")
            continue

        # 遍历图片文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
        for image_file in character_dir.rglob('*'):
            if image_file.suffix.lower() not in image_extensions:
                continue

            # 检查是否已存在
            existing = session.query(Sample).filter_by(image_path=str(image_file)).first()
            if existing:
                skipped_count += 1
                continue

            # 获取图片信息
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
            except Exception:
                width, height = None, None

            # 创建样本记录
            sample = Sample(
                image_path=str(image_file),
                character_id=character.id,
                width=width,
                height=height,
                status='pending'
            )
            session.add(sample)
            imported_count += 1

            # 每100条提交一次
            if imported_count % 100 == 0:
                session.commit()
                print(f"📥 已导入 {imported_count} 个样本...")

    # 提交剩余数据
    session.commit()
    
    print(f"\n✅ 导入完成！")
    print(f"   新增: {imported_count} 个样本")
    print(f"   跳过: {skipped_count} 个重复样本")

    session.close()
    engine.dispose()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="导入样本图片到数据库")
    parser.add_argument("-d", "--dir", default="data/final_dataset", help="数据目录")
    args = parser.parse_args()
    
    import_samples_from_directory(args.dir)
