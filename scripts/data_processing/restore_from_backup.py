#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从备份恢复有效图片数据
处理拼音目录名到正确角色名的映射
"""

import os
import shutil
import json
from pathlib import Path
from loguru import logger

# 拼音到角色名的映射
PINYIN_MAP = {
    'a1luo4na4': 'arona_(blue_archive)',
    'a1ni4ya4': 'ayane_(blue_archive)',
    'ai4li4er3': 'aerial_(arknights)',
    'an1ke3': 'an_(blue_archive)',
    'an1ka3xi1ya3': 'arknights',
    'Aris': 'aris_(blue_archive)',
    'Aris wei4lan2dang4an4': 'aris_(blue_archive)',
    'bai2shang4chui1xue3': 'bronya_(honkai_star_rail)',
    'bu4luo4ni2ya4': 'bronya_(honkai_star_rail)',
    'cong2yu3': 'kokomi_(genshin_impact)',
    'de2li4sha1': 'doris_(blue_archive)',
    'di2ao4na4': 'diona_(genshin_impact)',
    'duo1li4': 'dori_(genshin_impact)',
    'fei1mi3li4si1': 'fermi_(blue_archive)',
    'fei1xie4er3': 'fischl_(genshin_impact)',
    'fu2lan2': 'flandre_scarlet',
    'fu2li4xi1ya4': 'furina_(genshin_impact)',
    'fu2xuan2': 'fu_xuan_(honkai_star_rail)',
    'gu3ming2di4lian4': 'guinaifen_(honkai_star_rail)',
    'hei1ta3': 'heitai',
    'hua1huo3': 'sparkle_(honkai_star_rail)',
    'ji3ke4': 'klee_(genshin_impact)',
    'jia1na4': 'jean_(genshin_impact)',
    'kang1na4': 'kanna_kamui',
    'Kagura': 'kagura_(blue_archive)',
    'ke3li2': 'klee_(genshin_impact)',
    'ke3lin2_wei1ke4si1': 'kallen_(honkai_impact_3rd)',
    'ke4la1la1': 'collei_(genshin_impact)',
    'ke4luo2li4ke1': 'coroika',
    'ke4xie4ni2ya4': 'kshatriya_(arknights)',
    'kou4er3fu2': 'koleda_(honkai_star_rail)',
    'la1mu3': 'ram_(re_zero)',
    'lei2mu3': 'rem_(re_zero)',
    'lei3bei4': 'rebe_(tower_of_fantasy)',
    'li4li4ya3_a1lin2': 'liliya_olenyeva',
    'li4li4ya4·a1lin2': 'liliya_olenyeva',
    'li4ta3la1': 'lyla_(genshin_impact)',
    'luo2sha1li4ya3_a1lin2': 'rozaliya_olenyeva',
    'luo4ke3ke3': 'raki_(blue_archive)',
    'luo4ke4ke4': 'raki_(blue_archive)',
    'lu4mu4yuan2': 'lumine_(genshin_impact)',
    'mi3dou4zi5': 'miko_(blue_archive)',
    'mei2bi3wu3si1': 'mephistopheles_(arknights)',
    'na4gan1': 'nahida_(genshin_impact)',
    'na4xi1da4': 'nahida_(genshin_impact)',
    'Nezuko': 'nezuko_(kimetsu_no_yaiba)',
    'ni2dou4zi5': 'niji_douji',
    'pe4li3ti2ya4': 'pelagia_(blue_archive)',
    'qi2ta3': 'other',
    'qing1que4': 'qingque_(genshin_impact)',
    'ren3ye3ren3': 'ren_(arknights)',
    'san1yue4qi1': 'march_7th_(honkai_star_rail)',
    'sha1wu4': 'sasha_(blue_archive)',
    'shen2le4 yin1yang2shi1': 'shinobu_(genshin_impact)',
    'si4mi4nai3': 'simulacrum_(blue_archive)',
    'tian1tong2ai4li4si1': 'tien_(blue_archive)',
    'ti2bao3': 'ti_bao_(honkai_star_rail)',
    'wei2li3nai4': 'villhaze_(blue_archive)',
    'wei2pu3lei3': 'vipula_(arknights)',
    'wu4yu3mo2li3sha1': 'noelle_(genshin_impact)',
    'xi1er3': 'sier_(blue_archive)',
    'xi1ge2wen2': 'sigewinne_(genshin_impact)',
    'xiao3mai2': 'wheat_(blue_archive)',
    'xiao3mei3yan4': 'xiaomeiyan',
    'xiao3shan3': 'shan_(blue_archive)',
    'xin1xi1': 'sins_(blue_archive)',
    'xin1ye4': 'xinye',
    'ya1si4te3la1': 'asteria_(blue_archive)',
    'yao2yao2': 'yaoyao_(genshin_impact)',
    'yao2yao2 yuan2shen2': 'yaoyao_(genshin_impact)',
    'Yaoyao': 'yaoyao_(genshin_impact)',
    'Yaoyao yuan2shen2': 'yaoyao_(genshin_impact)',
    'yi1li4ya3': 'illya_(fate)',
    'yi1se4lin2': 'irys_(hololive)',
    'zao3wu4': 'zao_(blue_archive)',
    'zao3you4': 'sayo_(blue_archive)',
    'xue4xiao3ban3': 'school_uniform',
    'ありす': 'aris_(blue_archive)',
    'ユウホ': 'yuho_(blue_archive)',
}


def restore_backup(backup_dir, target_dir):
    """从备份恢复图片"""
    backup_path = Path(backup_dir)
    target_path = Path(target_dir)
    
    if not backup_path.exists():
        logger.error(f"备份目录不存在: {backup_dir}")
        return
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_restored': 0,
        'total_roles': 0,
        'skipped_dirs': 0,
        'mapped_roles': 0,
        'direct_copy': 0,
    }
    
    logger.info(f"开始从备份恢复: {backup_dir}")
    
    for pinyin_dir in backup_path.iterdir():
        if not pinyin_dir.is_dir():
            continue
        
        pinyin_name = pinyin_dir.name
        
        # 确定目标角色名
        if pinyin_name in PINYIN_MAP:
            target_role_name = PINYIN_MAP[pinyin_name]
            stats['mapped_roles'] += 1
        else:
            target_role_name = pinyin_name
            stats['direct_copy'] += 1
        
        # 创建目标目录
        target_role_dir = target_path / target_role_name
        target_role_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制图片
        image_count = 0
        for img_file in pinyin_dir.iterdir():
            if not img_file.is_file():
                continue
            if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
                continue
            
            # 验证图片有效性
            if is_valid_image(img_file):
                target_file = target_role_dir / img_file.name
                shutil.copy2(img_file, target_file)
                image_count += 1
                stats['total_restored'] += 1
        
        if image_count > 0:
            stats['total_roles'] += 1
            logger.info(f"恢复角色 '{pinyin_name}' -> '{target_role_name}': {image_count} 张图片")
        else:
            stats['skipped_dirs'] += 1
            logger.debug(f"跳过空目录: {pinyin_name}")
    
    logger.info("\n恢复完成!")
    logger.info(f"总恢复图片: {stats['total_restored']}")
    logger.info(f"总恢复角色: {stats['total_roles']}")
    logger.info(f"映射角色: {stats['mapped_roles']}")
    logger.info(f"直接复制: {stats['direct_copy']}")
    logger.info(f"跳过空目录: {stats['skipped_dirs']}")
    
    return stats


def is_valid_image(file_path):
    """检查图片是否有效"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(2)
        
        if header[:2] == b'\xff\xd8':  # JPEG
            return True
        if header[:4] == b'\x89PNG':  # PNG
            return True
        if header[:4] == b'RIFF':  # WebP
            return True
        
        # 检查是否是HTML
        if header[:5] in [b'<!DOC', b'<html']:
            return False
        
        return True
    except:
        return False


def main():
    backup_dir = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/spider_image_system/src/run/data/downloaded_images'
    target_dir = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset'
    
    restore_backup(backup_dir, target_dir)


if __name__ == '__main__':
    main()
