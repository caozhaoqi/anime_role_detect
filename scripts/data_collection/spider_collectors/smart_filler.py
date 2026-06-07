#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""智能补全采集器 - 并行采集不足角色，自动去重"""

import os
import sys
import time
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
from loguru import logger
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加项目根目录
project_root = Path(__file__).parent.parent
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
    logger.info(f"✅ 已加载通知配置")

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

class SmartFillerSpider:
    """智能补全采集器"""
    
    SITES = {
        'yande.re': {
            'name': 'Yande.re',
            'api_url': 'https://yande.re/post.json',
            'format': 'json',
            'priority': 1,
        },
    }
    
    # 中文作品名到英文标签映射
    WORK_MAPPING = {
        # 游戏
        '蔚蓝档案': 'blue_archive',
        '原神': 'genshin_impact',
        '崩坏星穹铁道': 'honkai:_star_rail',
        '崩坏3': 'honkai_impact_3rd',
        '崩坏学园2': 'honkai_academy_2',
        '鸣潮': 'wuthering_waves',
        '异环': 'arknights:_endfield',
        '明日方舟': 'arknights',
        '碧蓝航线': 'azur_lane',
        '公主连接': 'princess_connect',
        
        # 动漫
        '魔法少女小圆': 'puella_magi_madoka_magica',
        're:从零开始的异世界生活': 're:zero_kara_hajimeru_isekai_seikatsu',
        '小林家的龙女仆': 'kobayashi-san_chi_no_maidragon',
        '约会大作战': 'date_a_live',
        'fate/kaleid liner prisma illya': 'fate/kaleid_liner_prisma_illya',
        '物语系列': 'monogatari_series',
        '请问您今天要来点兔子吗': 'gochuumon_wa_usagi_desu_ka',
        '干物妹小埋': 'himouto!_umaru-chan',
        '埃罗芒阿老师': 'eromanga_sensei',
        '间谍过家家': 'spy_x_family',
        
        # 别名处理
        'honkai__star_rail': 'honkai:_star_rail',
        're:zero': 're:zero_kara_hajimeru_isekai_seikatsu',
        'fate/kaleid': 'fate/kaleid_liner_prisma_illya',
    }
    
    def __init__(self, max_workers: int = 8, include_nsfw: bool = True):
        self.max_workers = max_workers
        self.include_nsfw = include_nsfw
        self.session = self._create_session()
        self._lock = threading.Lock()
        self._seen_ids = set()  # 全局去重
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        })
        return session
    
    def search_all(self, tags: str, max_posts: int = 200) -> List[Dict]:
        """搜索并合并所有站点结果，自动去重"""
        all_posts = []
        
        for site in self.SITES:
            posts = self._search_site(site, tags, max_posts)
            
            for post in posts:
                post_id = post.get('id', post.get('md5', ''))
                if post_id and post_id not in self._seen_ids:
                    self._seen_ids.add(post_id)
                    all_posts.append(post)
            
            if len(all_posts) >= max_posts:
                break
            
            time.sleep(random.uniform(0.5, 1.0))
        
        return all_posts[:max_posts]
    
    def _search_site(self, site: str, tags: str, max_posts: int) -> List[Dict]:
        """搜索单个站点"""
        site_info = self.SITES[site]
        api_url = site_info['api_url']
        all_posts = []
        page = 1
        
        try:
            while len(all_posts) < max_posts:
                params = {'tags': tags, 'page': page, 'limit': min(100, max_posts)}
                response = self.session.get(api_url, params=params, timeout=30)
                response.raise_for_status()
                posts = response.json()
                
                if not posts:
                    break
                
                all_posts.extend(posts[:max_posts - len(all_posts)])
                page += 1
                time.sleep(random.uniform(0.3, 0.7))
            
            return all_posts
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def download_image(self, post: Dict, save_dir: Path) -> bool:
        """下载单张图片（自动跳过已存在）"""
        url_fields = ['file_url', 'source', 'image', 'url']
        image_url = None
        
        for field in url_fields:
            if field in post and post[field]:
                image_url = post[field]
                break
        
        if not image_url:
            return False
        
        ext = image_url.split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'gif']:
            ext = 'jpg'
        
        post_id = post.get('id', post.get('md5', random.randint(1, 9999)))
        file_path = save_dir / f"{post_id}.{ext}"
        
        # 检查是否已存在（去重）
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
            return False
    
    def fill_character(self, tag: str, save_dir: Path, target_count: int = 100) -> Tuple[int, int]:
        """补全单个角色到目标数量"""
        # 检查当前已有数量
        character_dir = save_dir / tag
        if character_dir.exists():
            current_count = len(list(character_dir.glob("*.jpg"))) + len(list(character_dir.glob("*.png")))
        else:
            current_count = 0
            character_dir.mkdir(parents=True, exist_ok=True)
        
        if current_count >= target_count:
            logger.info(f"[{tag}] 已达标 ({current_count}张)")
            return (0, 0)
        
        need_count = target_count - current_count
        logger.info(f"[{tag}] 需补充 {need_count} 张 (当前 {current_count}张)")
        
        # 解析标签，提取角色名和作品名
        character_name = tag
        work_name = ""
        
        if '(' in tag and ')' in tag:
            # 格式: name_(work)
            idx = tag.rfind('(')
            character_name = tag[:idx].strip('_')
            work_name = tag[idx+1:-1]
        
        # 尝试多种搜索策略
        all_posts = []
        strategies = []
        
        # 策略1: 原始标签
        strategies.append(tag)
        
        # 策略2: 只角色名
        strategies.append(character_name)
        
        # 策略3: 尝试修复作品名
        if work_name:
            mapped_work = self.WORK_MAPPING.get(work_name, work_name)
            if mapped_work != work_name:
                strategies.append(f"{character_name}_({mapped_work})")
        
        # 策略4: 简化角色名
        if '_' in character_name:
            strategies.append(character_name.replace('_', ' '))
        
        # 策略5: 移除作品限制的宽松搜索
        strategies.append(character_name.replace('_', ' '))
        
        # 执行搜索
        for strategy in strategies:
            if len(all_posts) >= need_count:
                break
            
            search_tags = strategy
            if not self.include_nsfw:
                search_tags += ' rating:safe'
            
            posts = self.search_all(search_tags, need_count - len(all_posts))
            all_posts.extend(posts)
            time.sleep(random.uniform(0.5, 1.0))
        
        # 下载
        success = 0
        fail = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_image, post, character_dir): post 
                      for post in all_posts}
            
            for future in as_completed(futures):
                try:
                    if future.result():
                        success += 1
                    else:
                        fail += 1
                except:
                    fail += 1
        
        logger.success(f"[{tag}] 补充完成: +{success}张")
        return (success, fail)

def get_insufficient_roles(data_dir: Path, min_count: int = 100) -> List[str]:
    """获取不足目标数量的角色列表"""
    insufficient = []
    
    for role_dir in data_dir.iterdir():
        if role_dir.is_dir():
            count = len(list(role_dir.glob("*.jpg"))) + len(list(role_dir.glob("*.png")))
            if count < min_count:
                insufficient.append((role_dir.name, count))
    
    # 按数量排序（最少的先处理）
    insufficient.sort(key=lambda x: x[1])
    return [role for role, _ in insufficient]

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='智能补全采集器')
    parser.add_argument('--output-dir', type=str, default='data/danbooru_images', help='数据目录')
    parser.add_argument('--target-count', type=int, default=100, help='目标数量')
    parser.add_argument('--workers', type=int, default=8, help='并发线程数')
    parser.add_argument('--delay', type=float, default=3.0, help='角色间延迟')
    
    args = parser.parse_args()
    
    # 创建采集器（默认启用NSFW）
    spider = SmartFillerSpider(max_workers=args.workers, include_nsfw=True)
    
    # 获取不足角色列表
    data_dir = Path(args.output_dir)
    insufficient_roles = get_insufficient_roles(data_dir, args.target_count)
    
    logger.info(f"找到 {len(insufficient_roles)} 个不足角色")
    
    # 发送开始通知
    start_msg = f"""**🔄 智能补全采集开始**

**配置信息**:
- 需要补全角色: {len(insufficient_roles)} 个
- 目标数量: {args.target_count}张/角色
- 并发线程: {args.workers}
- 启用NSFW: 是

**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
    send_notification(start_msg, "补全采集开始", "info")
    
    # 开始补全
    total_added = 0
    
    for i, tag in enumerate(insufficient_roles, 1):
        logger.info(f"[{i}/{len(insufficient_roles)}] 补全: {tag}")
        
        try:
            added, _ = spider.fill_character(tag, data_dir, args.target_count)
            total_added += added
            
            # 每5个角色发送进度
            if i % 5 == 0:
                progress_msg = f"""**📊 补全进度更新**

**当前进度**: {i}/{len(insufficient_roles)} 个角色
**本批次新增**: {added}张

**累计新增**: {total_added} 张

**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
                send_notification(progress_msg, "补全进度", "info")
                
        except Exception as e:
            logger.error(f"补全 {tag} 失败: {e}")
        
        time.sleep(random.uniform(args.delay * 0.5, args.delay * 1.5))
    
    # 发送完成通知
    complete_msg = f"""**✅ 智能补全采集完成**

**统计信息**:
- 处理角色: {len(insufficient_roles)} 个
- 新增图片: {total_added} 张

**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
    send_notification(complete_msg, "补全完成", "success")
    
    logger.success("补全采集完成！")

if __name__ == '__main__':
    main()
