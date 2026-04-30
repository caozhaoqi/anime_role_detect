#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接调用爬虫功能采集缺失角色的URL
"""

import os
import sys
import time
import logging

# 添加爬虫系统路径
sys.path.insert(0, '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/src')

from ui_event.get_url import spider_artworks_url

# 配置
ROLE_LIST_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_existing_url_files():
    """获取已有的URL文件名"""
    existing = set()
    if os.path.exists(URL_DIR):
        for filename in os.listdir(URL_DIR):
            if filename.endswith('_img.txt'):
                role_pinyin = filename.replace('_img.txt', '')
                existing.add(role_pinyin)
    return existing


def get_all_roles():
    """获取完整角色列表"""
    roles = []
    with open(ROLE_LIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ')
                chinese_name = parts[0]
                roles.append(chinese_name)
    return roles


def get_missing_roles():
    """获取需要爬取的角色列表"""
    all_roles = get_all_roles()
    existing_files = get_existing_url_files()
    
    # 拼音映射表
    pinyin_mapping = {
        '阿洛娜': ['a1lu4', 'a1luo2na4', 'a1luo4na4'],
        '普拉娜': ['pu3la1na4'],
        '纳西妲': ['na4xi1da2', 'na4xi1da4'],
        '缇宝': ['ti2bao3'],
        '可莉': ['ke3li4', 'ke3li2'],
        '迪奥娜': ['di2ao4na4'],
        '瑶瑶': ['yao2yao2'],
        '希格雯': ['xi1ge2wen2'],
        '蕾贝': ['lei3bei4'],
        '黑塔': ['hei1ta3'],
        '符玄': ['fu2xuan2'],
        '七七': ['qi1qi1'],
        '早柚': ['zao3you4'],
        '多莉': ['duo1li4'],
        '卡齐娜': ['ka3qi2na4'],
        '三月七': ['san1yue4qi1'],
        '花火': ['hua1huo3'],
        '银狼': ['yin2lang2'],
        '天童爱丽丝': ['tian1tong2ai4li4si1'],
        '早雾': ['zao3wu4'],
        '维里奈': ['wei2li3nai4'],
        '安可': ['an1ke3'],
        '釉壶': ['you4hu2'],
        '洛可可': ['luo4ke4ke4'],
        '鹿目圆': ['lu4mu4yuan2'],
        '晓美焰': ['xiao3mei3yan4'],
        '血小板': ['xue3xiao3ban3'],
        '雷姆': ['lei2mu3'],
        '拉姆': ['la1mu3'],
        '康娜': ['kang1na4'],
        '四糸乃': ['si4mi4nai3'],
        '凯露': ['kai3lu4'],
        '克萝萝': ['ke4luo2luo2'],
        '小闪': ['xiao3shan3'],
        '伊莉雅': ['yi1li4ya3'],
        '忍野忍': ['ren3ye3ren3'],
        '智乃': ['zhi4nai3'],
        '小埋': ['xiao3mai2'],
        '纱雾': ['sha1wu4'],
        '猫宫又奈': ['mao1gong1you4nai4'],
        '德丽莎': ['de2li4sha1'],
        '布洛妮娅': ['bu4luo4ni2ya4'],
        '可琳': ['ke3lin2'],
        '爱丽儿': ['ai4li4er3'],
        '神乐': ['shen1yue4'],
        '白上吹雪': ['bai2shang4chui1xue3'],
        '月千夜': ['yue4qian1ye4'],
        '芙丽希娅': ['fu2li4xi1ya4'],
        '莉塔拉': ['li4ta3la1'],
        '维普蕾': ['wei2pu3lei3'],
        '夏克里': ['xia4ke4li3'],
        '纳甘': ['na4gan1'],
        '科谢尼娅': ['ke1xie4ni2ya4'],
        '奇塔': ['qi2ta3'],
        '寇尔芙': ['kou4er3fu2'],
        '克罗丽科': ['ke4luo2li4ke1'],
        '佩里缇亚': ['pei4li3ti2ya4'],
        '阿尼亚': ['a1ni4ya4'],
        '洛茜': ['luo4qian4'],
        '祢豆子': ['ni2dou4zi5'],
        '希儿': ['xi1er3'],
        '杏': ['xing4'],
        '伊瑟琳': ['yi1se4lin2'],
        '芙兰': ['fu2lan2'],
        '菲米莉丝': ['fei1mi3li4si1'],
    }
    
    missing = []
    for role in all_roles:
        pinyins = pinyin_mapping.get(role, [role])
        has_url = False
        for pinyin in pinyins:
            if f"{pinyin}_img.txt" in existing_files:
                has_url = True
                break
        if not has_url:
            missing.append(role)
    return missing


def spider_roles():
    """爬取缺失角色的URL"""
    missing_roles = get_missing_roles()
    
    if not missing_roles:
        logger.info("所有角色的URL都已采集完成！")
        return
    
    logger.info(f"找到 {len(missing_roles)} 个缺失角色需要爬取")
    
    success_count = 0
    for i, role in enumerate(missing_roles, 1):
        logger.info(f"[{i}/{len(missing_roles)}] 开始爬取: {role}")
        
        try:
            # 调用爬虫主函数
            spider_artworks_url(key_word=role, page_count=5)
            success_count += 1
            logger.info(f"✓ 爬取完成: {role}")
        except Exception as e:
            logger.error(f"✗ 爬取失败: {role} - {str(e)}")
        
        # 间隔10秒再爬取下一个
        time.sleep(10)
    
    logger.info(f"\n=== 爬取完成 ===")
    logger.info(f"成功: {success_count}/{len(missing_roles)}")


if __name__ == '__main__':
    spider_roles()
