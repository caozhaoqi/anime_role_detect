#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多站点智能采集器 - 优化版
支持自动站点切换、失败重试、并发优化
"""

import os
import sys
import time
import random
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 加载飞书配置
config_path = project_root / "scripts" / "notification_config.json"
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        notification_config = json.load(f)
    
    os.environ["NOTIFICATION_ENABLED"] = "true"
    os.environ["NOTIFICATION_PLATFORM"] = notification_config["platform"]
    os.environ["FEISHU_APP_ID"] = notification_config["feishu"]["app_id"]
    os.environ["FEISHU_APP_SECRET"] = notification_config["feishu"]["app_secret"]
    os.environ["FEISHU_RECEIVE_ID"] = notification_config["feishu"]["receive_id"]
    os.environ["FEISHU_RECEIVE_ID_TYPE"] = notification_config["feishu"]["receive_id_type"]
    logger.info(f"✅ 已加载通知配置: {config_path}")

# 导入通知服务
try:
    from src.services.notification_service import get_notification_manager
    notification_manager = get_notification_manager()
    logger.info("✅ 通知服务初始化完成")
except Exception as e:
    notification_manager = None
    logger.warning(f"⚠️ 通知服务未找到: {e}")

def send_notification(message, title=None, level="info"):
    """发送通知"""
    if notification_manager:
        try:
            return notification_manager.send(message, title, level)
        except Exception as e:
            logger.warning(f"发送通知失败: {e}")
    return False

# 作品名映射
WORK_TITLE_MAPPING = {
    '蔚蓝档案': 'blue_archive',
    '原神': 'genshin_impact',
    '崩坏星穹铁道': 'honkai:_star_rail',
    '幻塔': 'tower_of_fantasy',
    '公主连结': 'princess_connect',
    '赛马娘': 'umamusume',
    '明日方舟': 'arknights',
    '碧蓝航线': 'azur_lane',
    'FGO': 'fate/grand_order',
    'Fate': 'fate',
    '偶像大师': 'idolmaster',
    'LoveLive': 'love_live',
    'BanG Dream': 'bang_dream',
    '你的名字': 'kimi_no_na_wa.',
    '声之形': 'koe_no_katachi',
    '天气之子': 'tenki_no_ko',
    '铃芽之旅': 'suzume_no_tojimari',
    '五等分的新娘': '5-toubun_no_hanayome',
    '咒术回战': 'jujutsu_kaisen',
    '鬼灭之刃': 'kimetsu_no_yaiba',
    '间谍过家家': 'spy_x_family',
    '电锯人': 'chainsaw_man',
    '孤独摇滚': 'bocchi_the_rock!',
    '孤独摇滚!': 'bocchi_the_rock!',
    '莉可丽丝': 'lycoris_recoil',
    '无职转生': 'mushoku_tensei',
    '关于我转生变成史莱姆这档事': 'tensei_shitara_slime_datta_ken',
    '辉夜大小姐想让我告白': 'kaguya-sama_wa_kokurasetai',
    '知晓天空之蓝的人啊': 'sora_no_aosa_wo_shiru_hito_yo',
    '青春猪头少年不会梦到兔女郎学姐': 'seishun_buta_yarou_series',
    '刀剑神域': 'sword_art_online',
    '进击的巨人': 'shingeki_no_kyojin',
    '命运石之门': 'steins;gate',
    '从零开始的异世界生活': 're:zero_kara_hajimeru_isekai_seikatsu',
    '紫罗兰永恒花园': 'violet_evergarden',
    '轻音少女': 'k-on!',
    '未闻花名': 'ano_hi_mita_hana_no_namae_wo_bokutachi_wa_mada_shiranai.',
    '魔卡少女樱': 'cardcaptor_sakura',
    '魔法少女小圆': 'mahou_shoujo_madoka_magica',
    '东方Project': 'touhou',
}

def format_danbooru_tag(english_name: str, work_title: str) -> str:
    """构建Danbooru标签"""
    work_tag = WORK_TITLE_MAPPING.get(work_title, work_title.replace(' ', '_'))
    return f"{english_name.lower()}_({work_tag})"

class MultiSiteSpider:
    """多站点智能采集器"""
    
    SITES = {
        'yande.re': {
            'name': 'Yande.re',
            'api_url': 'https://yande.re/post.json',
            'format': 'json',
            'priority': 1,
        },
        'gelbooru': {
            'name': 'Gelbooru',
            'api_url': 'https://gelbooru.com/index.php?page=dapi&s=post&q=index',
            'format': 'xml',
            'priority': 2,
        },
        'konachan': {
            'name': 'Konachan',
            'api_url': 'https://konachan.com/post.json',
            'format': 'json',
            'priority': 3,
        },
    }
    
    def __init__(self, max_workers: int = 16, include_nsfw: bool = False):
        self.max_workers = max_workers
        self.include_nsfw = include_nsfw
        self.session = self._create_session()
        self._lock = threading.Lock()
        
    def _create_session(self) -> requests.Session:
        """创建请求会话"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/xml, */*',
        })
        return session
    
    def search_site(self, site: str, tags: str, max_posts: int = 100) -> List[Dict]:
        """搜索单个站点"""
        site_info = self.SITES[site]
        api_url = site_info['api_url']
        
        all_posts = []
        page = 1
        
        try:
            while len(all_posts) < max_posts:
                if site_info['format'] == 'json':
                    params = {'tags': tags, 'page': page, 'limit': min(100, max_posts)}
                    response = self.session.get(api_url, params=params, timeout=30)
                    response.raise_for_status()
                    posts = response.json()
                else:  # XML
                    params = {'tags': tags, 'pid': page - 1, 'limit': min(100, max_posts)}
                    response = self.session.get(api_url, params=params, timeout=30)
                    response.raise_for_status()
                    posts = self._parse_xml(response.text)
                
                if not posts:
                    break
                    
                all_posts.extend(posts[:max_posts - len(all_posts)])
                page += 1
                time.sleep(random.uniform(0.5, 1.0))
                
            logger.info(f"[{site_info['name']}] 找到 {len(all_posts)} 张图片")
            return all_posts
            
        except Exception as e:
            logger.error(f"[{site_info['name']}] 搜索失败: {e}")
            return []
    
    def _parse_xml(self, xml_text: str) -> List[Dict]:
        """解析XML响应"""
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(xml_text)
            return [{attr: post.attrib[attr] for attr in post.attrib} 
                    for post in root.findall('post')]
        except:
            return []
    
    def search_all_sites(self, tags: str, max_posts: int = 100) -> List[Dict]:
        """搜索所有站点，返回合并结果"""
        all_posts = []
        seen_ids = set()
        
        # 按优先级搜索站点
        for site in sorted(self.SITES.keys(), key=lambda s: self.SITES[s]['priority']):
            posts = self.search_site(site, tags, max_posts)
            
            # 去重
            for post in posts:
                post_id = post.get('id', post.get('md5', ''))
                if post_id and post_id not in seen_ids:
                    seen_ids.add(post_id)
                    all_posts.append(post)
            
            if len(all_posts) >= max_posts:
                break
                
            time.sleep(random.uniform(1.0, 2.0))
        
        return all_posts[:max_posts]
    
    def download_image(self, post: Dict, save_dir: Path) -> bool:
        """下载单张图片"""
        # 提取图片URL
        url_fields = ['file_url', 'source', 'image', 'url']
        image_url = None
        for field in url_fields:
            if field in post and post[field]:
                image_url = post[field]
                break
        
        if not image_url:
            return False
        
        # 生成文件名
        ext = image_url.split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'gif']:
            ext = 'jpg'
        
        post_id = post.get('id', post.get('md5', random.randint(1, 9999)))
        file_path = save_dir / f"{post_id}.{ext}"
        
        if file_path.exists():
            return True
        
        try:
            response = self.session.get(image_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            logger.debug(f"下载失败: {e}")
            return False
    
    def download_character(self, tag: str, save_dir: Path, max_count: int = 100) -> Tuple[int, int]:
        """下载单个角色的图片"""
        # 构建搜索标签
        tags = tag
        if not self.include_nsfw:
            tags += ' rating:safe'
        
        # 搜索所有站点
        posts = self.search_all_sites(tags, max_count)
        
        if not posts:
            logger.warning(f"角色 '{tag}' 未找到图片")
            return (0, 0)
        
        # 创建保存目录
        character_dir = save_dir / self._sanitize(tag)
        character_dir.mkdir(parents=True, exist_ok=True)
        
        # 并发下载
        success = 0
        fail = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_image, post, character_dir): post 
                      for post in posts}
            
            for future in as_completed(futures):
                try:
                    if future.result():
                        success += 1
                    else:
                        fail += 1
                except:
                    fail += 1
        
        logger.success(f"{tag}: 成功 {success}, 失败 {fail}")
        return (success, fail)
    
    def _sanitize(self, filename: str) -> str:
        """清理文件名"""
        import re
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
        return sanitized.strip().strip('_')[:100]

def load_character_list(file_path: str) -> List[Tuple[str, str, str]]:
    """加载角色列表"""
    characters = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3:
                    characters.append((parts[0], parts[2], parts[1]))
    return characters

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='多站点智能采集器')
    parser.add_argument('--character-file', type=str, required=True, help='角色列表文件')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--max-count', type=int, default=100, help='每个角色最大下载数')
    parser.add_argument('--workers', type=int, default=16, help='并发线程数')
    parser.add_argument('--include-nsfw', action='store_true', help='包含非安全内容')
    parser.add_argument('--delay', type=float, default=2.0, help='角色间延迟')
    parser.add_argument('--start-from', type=int, default=0, help='从第几个角色开始')
    
    args = parser.parse_args()
    
    # 创建采集器
    spider = MultiSiteSpider(max_workers=args.workers, include_nsfw=args.include_nsfw)
    
    # 加载角色列表
    characters = load_character_list(args.character_file)
    logger.info(f"加载了 {len(characters)} 个角色")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 发送开始通知
    start_msg = f"""**🚀 多站点智能采集开始**

**配置信息**:
- 角色总数: {len(characters)}
- 每角色数量: {args.max_count}
- 并发线程: {args.workers}
- 采集站点: Yande.re, Gelbooru, Konachan

**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
    send_notification(start_msg, "采集任务开始", "info")
    
    # 开始采集
    total_success = 0
    total_fail = 0
    
    for i, (chinese_name, english_name, work_title) in enumerate(characters[args.start_from:], start=args.start_from):
        tag = format_danbooru_tag(english_name, work_title)
        logger.info(f"[{i+1}/{len(characters)}] {chinese_name} ({tag})")
        
        try:
            success, fail = spider.download_character(tag, output_dir, args.max_count)
            total_success += success
            total_fail += fail
            
            # 每10个角色发送进度通知
            if (i + 1) % 10 == 0:
                progress_msg = f"""**📊 采集进度更新**

**当前进度**: {i+1}/{len(characters)} 个角色
**本批次**: {chinese_name} ✅ {success}张

**累计统计**:
- 成功: {total_success} 张
- 失败: {total_fail} 张

**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
                send_notification(progress_msg, "采集进度", "info")
                
        except Exception as e:
            logger.error(f"处理 {chinese_name} 失败: {e}")
        
        time.sleep(random.uniform(args.delay * 0.5, args.delay * 1.5))
    
    # 发送完成通知
    complete_msg = f"""**✅ 采集任务完成**

**统计信息**:
- 总角色数: {len(characters)}
- 成功下载: {total_success} 张
- 失败: {total_fail} 张

**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
    send_notification(complete_msg, "采集完成", "success")
    
    logger.success("采集完成！")

if __name__ == '__main__':
    main()
