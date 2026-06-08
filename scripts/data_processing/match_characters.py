#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
匹配数据目录角色名和数据库角色名
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.database.init_db import init_database, Character, Sample
from sqlalchemy import text

# 数据目录中的角色名映射到数据库角色名
CHARACTER_NAME_MAP = {
    # 蔚蓝档案
    'aris_(blue_archive)': 'aris',
    'arona_(blue_archive)': 'arona',
    'ayane_(blue_archive)': 'ayane',
    'doris_(blue_archive)': 'dori',  # 注意：doris -> dori
    'fermi_(blue_archive)': 'fermi',
    'villhaze_(blue_archive)': 'villhaze',
    'sayo_(blue_archive)': 'sayo',
    
    # 原神
    'yaoyao_(genshin_impact)': 'yaoyao',
    'lumine_(genshin_impact)': 'lumine',
    'qingque_(genshin_impact)': 'qingque',
    'diona_(genshin_impact)': 'diona',
    'collei_(genshin_impact)': 'collei',
    
    # 明日方舟
    'aerial_(arknights)': 'aerial',
    'ren_(arknights)': 'ren',
    'vipula_(arknights)': 'vipula',
    
    # 崩坏星穹铁道
    'bronya_(honkai_star_rail)': 'bronya',
    'koleda_(honkai_star_rail)': 'koleda',
    'ti_bao_(honkai_star_rail)': 'ti_bao',
    
    # Re:从零开始的异世界
    'ram_(re_zero)': 'ram',
    
    # Hololive
    'irys_(hololive)': 'irys',
    
    # 东方Project
    'flandre_scarlet': 'flandre',
    
    # 鬼灭之刃
    'nezuko_(kimetsu_no_yaiba)': 'nezuko',
    
    # 拼音目录名映射
    'ke3li4': 'klee',
    'qi1qi1': 'qiqi',
    'na4xi1da2': 'nahida',
    'niji_douji': 'niji',
    'mao1gong1you4nai4': 'mao_gong',
    'yue4qian1ye4': 'yue_qian_ye',
    'yin2lang2': 'yin_lang',
    'ka3qi2na4': 'ka_qi_na',
    'xia4ke4li3': 'xia_ke_li',
    'xiaomeiyan': 'xiao_mei_yan',
    'xing4': 'xing',
    'kai3lu4': 'kai_lu',
    'ke4luo2luo2': 'ke_luo_luo',
}


def match_and_import():
    """匹配角色名并导入样本"""
    engine, Session = init_database()
    session = Session()
    
    data_dir = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset')
    
    # 获取数据库中的所有角色
    db_characters = {char.name: char for char in session.query(Character).all()}
    print(f"数据库中的角色数: {len(db_characters)}")
    
    # 统计
    stats = {
        'total_dirs': 0,
        'matched_dirs': 0,
        'imported_samples': 0,
        'skipped_samples': 0,
        'unmatched_dirs': [],
    }
    
    # 遍历数据目录
    for character_dir in data_dir.iterdir():
        if not character_dir.is_dir():
            continue
        
        stats['total_dirs'] += 1
        dir_name = character_dir.name
        
        # 查找匹配的数据库角色名
        db_name = CHARACTER_NAME_MAP.get(dir_name)
        
        if not db_name:
            # 尝试从标准格式提取（如 aris_(blue_archive) -> aris）
            if '_(' in dir_name:
                db_name = dir_name.rsplit('_(', 1)[0]
        
        if db_name and db_name in db_characters:
            stats['matched_dirs'] += 1
            character = db_characters[db_name]
            
            # 导入图片
            image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
            for img_file in character_dir.rglob('*'):
                if img_file.suffix.lower() not in image_extensions:
                    continue
                
                # 检查是否已存在
                existing = session.query(Sample).filter_by(image_path=str(img_file)).first()
                if existing:
                    stats['skipped_samples'] += 1
                    continue
                
                # 创建样本记录
                sample = Sample(
                    character_id=character.id,
                    image_path=str(img_file),
                    status='pending'
                )
                session.add(sample)
                stats['imported_samples'] += 1
            
            print(f"✅ 匹配成功: {dir_name} -> {db_name}")
        else:
            stats['unmatched_dirs'].append(dir_name)
            print(f"❌ 未匹配: {dir_name}")
    
    # 提交更改
    session.commit()
    session.close()
    
    # 打印统计
    print("\n" + "="*60)
    print("导入统计")
    print("="*60)
    print(f"总目录数: {stats['total_dirs']}")
    print(f"匹配成功: {stats['matched_dirs']}")
    print(f"导入样本: {stats['imported_samples']}")
    print(f"跳过样本: {stats['skipped_samples']}")
    print(f"未匹配目录: {len(stats['unmatched_dirs'])}")
    
    if stats['unmatched_dirs']:
        print("\n未匹配的目录:")
        for dir_name in stats['unmatched_dirs'][:10]:
            print(f"  - {dir_name}")
        if len(stats['unmatched_dirs']) > 10:
            print(f"  ... 还有 {len(stats['unmatched_dirs']) - 10} 个")


if __name__ == '__main__':
    match_and_import()
