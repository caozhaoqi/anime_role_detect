#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展采集脚本 - 将每个角色补充到100张图片
"""

import os
import sys
import time
import json
import requests

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("spider_extend_to_100")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("spider_extend_to_100")

DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
API_BASE = "http://localhost:33333/api/v1.2.5.260305/sis"
TARGET_COUNT = 100

PINYIN_MAPPING = {
    'a1luo4na4': '阿洛娜',
    'a1ni4ya4': '阿妮娅',
    'an1ke3': '安可',
    'bai2shang4chui1xue3': '白上吹雪',
    'bu4luo4ni2ya4': '布洛妮娅',
    'de2li4sha1': '德丽莎',
    'di2ao4na4': '迪奥娜',
    'duo1li4': '多莉',
    'fei1mi3li4si1': '菲米莉丝',
    'fu2lan2': '芙兰',
    'fu2xuan2': '符玄',
    'hei1ta3': '黑塔',
    'hua1huo3': '火花',
    'ka3qi2na4': '卡琪娜',
    'kai3lu4': '开萝',
    'kang1na4': '康娜',
    'ke1xie4ni2ya4': '克谢尼娅',
    'ke3li4': '刻晴',
    'ke3lin2': '克林',
    'ke4la1la1': '克拉拉',
    'ke4luo2li4ke1': '克罗丽克',
    'kou4er3fu2': '蔻尔芙',
    'la1mu3': '拉姆',
    'lei2mu3': '雷姆',
    'lei3bei4': '蕾贝',
    'li4ta3la1': '莉塔菈',
    'lu4mu4yuan2': '鹿目圆',
    'luo4qian4': '洛茜',
    'mao1gong1you4nai4': '猫宫又奈',
    'mi2dou4zi': '蜜豆子',
    'na4gan1': '娜甘',
    'na4xi1da2': '纳西妲',
    'pei4li3ti2ya4': '佩丽缇娅',
    'pu3la1na4': '普拉娜',
    'qi1qi1': '七七',
    'ren3ye3ren3': '人外人',
    'san1yue4qi1': '三月七',
    'sha1wu4': '砂狼',
    'shen1yue4': '神乐',
    'si4mi4nai3': '四宫奈',
    'ti2bao3': '提宝',
    'tian1tong2ai4li4si1': '天音爱莉丝',
    'wei2li3nai4': '薇莉奈',
    'wei2pu3lei3': '薇普蕾',
    'xi1er3': '希尔',
    'xi1ge2wen2': '希格雯',
    'xia4ke4li3': '夏可莉',
    'xiao3mai2': '小麦',
    'xiao3mei3yan4': '小美焰',
    'xing4': '星',
    'xue4xiao3ban3': '学校版',
    'yao2yao2': '瑶瑶',
    'yi1li4ya3': '伊莉雅',
    'yi1se4lin2': '伊瑟琳',
    'yin2lang2': '银狼',
    'you4hu2': '釉瑚',
    'yue4qian1ye4': '月千夜',
    'zao3wu4': '早雾',
    'zao3you4': '早柚',
    'zhi4nai3': '智乃'
}

def get_current_count(role_pinyin):
    """获取角色当前图片数量"""
    role_dir = os.path.join(DATA_DIR, role_pinyin)
    if not os.path.exists(role_dir):
        return 0
    return len([f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])

def spider_single_role(role_name):
    """调用API爬取单个角色的URL"""
    url = f"{API_BASE}/spider_start/single?key_word={role_name}"
    try:
        response = requests.post(url, timeout=30)
        result = response.json()
        if result.get('code') == 200:
            logger.info(f"✅ 开始爬取 {role_name}: {result.get('msg')}")
            return True
        else:
            logger.error(f"❌ 爬取 {role_name} 失败: {result.get('msg')}")
            return False
    except Exception as e:
        logger.error(f"❌ 爬取 {role_name} 异常: {str(e)}")
        return False

def check_spider_status():
    """检查爬虫状态"""
    url = f"{API_BASE}/spider/status"
    try:
        response = requests.get(url, timeout=10)
        result = response.json()
        if result.get('code') == 200:
            data = result.get('data', {})
            return data.get('is_running', False), data.get('current_keyword', '')
        return False, ''
    except Exception as e:
        logger.warning(f"⚠️ 检查状态异常: {str(e)}")
        return False, ''

def wait_for_spider():
    """等待当前爬虫任务完成"""
    while True:
        is_running, keyword = check_spider_status()
        if not is_running:
            break
        logger.info(f"⏳ 正在爬取 {keyword}，等待完成...")
        time.sleep(15)
    logger.info("✅ 当前爬虫任务完成")

def download_images(role_pinyin):
    """下载采集到的图片"""
    download_script = os.path.join(project_root, 'scripts/data_collection/download_low_count.py')
    
    cmd = ['python3', download_script, '--role', role_pinyin]
    
    try:
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info(f"✅ {role_pinyin}: 下载完成")
            return True
        else:
            logger.error(f"❌ {role_pinyin}: 下载失败 - {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {role_pinyin}: 下载超时")
        return False
    except Exception as e:
        logger.error(f"❌ {role_pinyin}: 下载异常 - {e}")
        return False

def main():
    logger.info("=" * 70)
    logger.info("🚀 开始扩展采集 - 目标: 每个角色100张图片")
    logger.info("=" * 70)
    
    # 获取需要采集的角色列表（按当前数量排序）
    roles_need = []
    for pinyin, name in PINYIN_MAPPING.items():
        current = get_current_count(pinyin)
        if current < TARGET_COUNT:
            roles_need.append((current, pinyin, name))
    
    # 按当前数量升序排列（先采集数量少的）
    roles_need.sort()
    
    logger.info(f"需要补充的角色: {len(roles_need)} 个")
    for current, pinyin, name in roles_need[:10]:  # 只显示前10个
        logger.info(f"  {name}: {current}/{TARGET_COUNT}")
    if len(roles_need) > 10:
        logger.info(f"  ... (还有 {len(roles_need) - 10} 个)")
    
    # 逐个采集
    for current, pinyin, name in roles_need:
        need_count = TARGET_COUNT - current
        logger.info(f"\n📦 处理 {name} ({pinyin}): 需要补充 {need_count} 张")
        
        # 等待前一个任务完成
        wait_for_spider()
        
        # 采集URL
        success = spider_single_role(name)
        
        if success:
            # 等待爬虫启动
            time.sleep(5)
            
            # 等待爬取完成
            wait_for_spider()
            
            # 下载图片
            download_images(pinyin)
            
            # 等待一下，避免请求过快
            time.sleep(3)
    
    # 最终统计
    logger.info("\n" + "=" * 70)
    logger.info("📊 扩展采集完成")
    logger.info("=" * 70)
    
    total_images = 0
    completed_count = 0
    remaining_roles = []
    
    for pinyin, name in PINYIN_MAPPING.items():
        current = get_current_count(pinyin)
        total_images += current
        if current >= TARGET_COUNT:
            completed_count += 1
        else:
            remaining_roles.append((name, current))
    
    logger.info(f"已完成: {completed_count}/{len(PINYIN_MAPPING)} 角色")
    logger.info(f"总图片数: {total_images}")
    logger.info(f"平均每角色: {total_images / len(PINYIN_MAPPING):.1f} 张")
    
    if remaining_roles:
        logger.info(f"\n⚠️ 未达标的角色 ({len(remaining_roles)}个):")
        for name, count in remaining_roles[:10]:
            logger.info(f"  {name}: {count}/{TARGET_COUNT}")
        if len(remaining_roles) > 10:
            logger.info(f"  ... (还有 {len(remaining_roles) - 10} 个)")
    else:
        logger.info("\n🎉 所有角色均已达标100张！")

if __name__ == '__main__':
    main()
