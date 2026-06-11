#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多站点角色图片采集器 - 增强版
支持多个镜像站点轮换、日文名搜索、别名搜索策略
"""

import os
import sys
import time
import random
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加项目根目录到sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

class MultiSiteSpider:
    """支持多站点轮换的角色图片采集器"""
    
    # MD5索引文件路径
    MD5_INDEX_FILE = None
    MD5_SET = set()
    
    # 支持的镜像站点列表（按优先级排序）
    MIRROR_SITES = {
        'lolibooru': {
            'name': 'Lolibooru',
            'api_url': 'https://lolibooru.moe/post.json',
            'requires_auth': False,
            'rate_limit': 2,
            'format': 'json',
            'nsfw': False,  # 默认安全
        },
        'yande.re': {
            'name': 'Yande.re',
            'api_url': 'https://yande.re/post.json',
            'requires_auth': False,
            'rate_limit': 2,
            'format': 'json',
            'nsfw': True,
        },
        'konachan': {
            'name': 'Konachan',
            'api_url': 'https://konachan.com/post.json',
            'requires_auth': False,
            'rate_limit': 2,
            'format': 'json',
            'nsfw': True,
        },
        'gelbooru': {
            'name': 'Gelbooru',
            'api_url': 'https://gelbooru.com/index.php?page=dapi&s=post&q=index',
            'requires_auth': False,
            'rate_limit': 1,
            'format': 'xml',
            'nsfw': True,
        },
        'safebooru': {
            'name': 'Safebooru',
            'api_url': 'https://safebooru.org/index.php?page=dapi&s=post&q=index',
            'requires_auth': False,
            'rate_limit': 1,
            'format': 'xml',
            'nsfw': False,
        },
    }
    
    # 角色别名映射（日文名、常见别名）
    CHARACTER_ALIASES = {
        # 蔚蓝档案
        '阿洛娜': ['アロナ', 'arona'],
        '普拉娜': ['プラナ', 'plana'],
        '砂狼白子': ['シロコ', 'shiroko', 'sunaookami_shiroko'],
        '圣园未花': ['ミカ', 'mika', 'mika_misono'],
        '空崎日奈': ['ヒナ', 'hina', 'hina_sorasaki'],
        '小鸟游星野': ['ホシノ', 'hoshino', 'hoshino_takanashi'],
        
        # 原神
        '纳西妲': ['ナヒダ', 'nahida'],
        '可莉': ['クレー', 'klee'],
        '七七': ['チチ', 'qiqi'],
        '早柚': ['サユ', 'sayu'],
        '胡桃': ['hutao', 'hu_tao'],
        '芙宁娜': ['フリーナ', 'furina'],
        
        # 崩坏星穹铁道
        '三月七': ['マーチ', 'march', 'march_7th'],
        '花火': ['スパークル', 'sparkle'],
        '克拉拉': ['クララ', 'clara'],
        '白露': ['bailu'],
        
        # 东方Project
        '琪露诺': ['チルノ', 'cirno'],
        '芙兰朵露': ['フラン', 'flandre', 'flandre_scarlet'],
        '蕾米莉亚': ['レミリア', 'remilia', 'remilia_scarlet'],
        '古明地恋': ['こいし', 'koishi', 'koishi_komeiji'],
        '洩矢诹访子': ['すわこ', 'suwako', 'suwako_moriya'],
        '铃仙·优昙华院·因幡': ['れいせん', 'reisen', 'reisen_udongein_inaba'],
        
        # 其他
        '雷姆': ['レム', 'rem'],
        '拉姆': ['ラム', 'ram'],
        '阿尼亚': ['アーニャ', 'anya', 'anya_forger'],
        '香风智乃': ['チノ', 'chino', 'kafuu_chino'],
        '康娜': ['カンナ', 'kanna', 'kanna_kamui'],
        '鹿目圆': ['まどか', 'madoka', 'kaname_madoka'],
        '晓美焰': ['ほむら', 'homura', 'akemi_homura'],
    }
    
    def __init__(self, sites: List[str] = None, max_workers: int = 8, 
                 delay: float = 2.0, timeout: int = 30,
                 md5_index_file: str = None):
        """
        初始化采集器
        
        Args:
            sites: 站点列表，按优先级排序
            max_workers: 最大并发下载线程数
            delay: 请求间隔延迟（秒）
            timeout: 请求超时时间（秒）
            md5_index_file: MD5索引文件路径（用于去重）
        """
        self.sites = sites or ['lolibooru', 'yande.re', 'konachan']
        self.max_workers = max_workers
        self.delay = delay
        self.timeout = timeout
        self._download_lock = threading.Lock()
        self._progress_counter = 0
        
        # 验证站点
        for site in self.sites:
            if site not in self.MIRROR_SITES:
                logger.warning(f"不支持的站点: {site}，已忽略")
                self.sites.remove(site)
        
        logger.info(f"使用采集站点: {[self.MIRROR_SITES[s]['name'] for s in self.sites]}")
        
        # 加载MD5索引
        if md5_index_file:
            self.load_md5_index(md5_index_file)
    
    def load_md5_index(self, index_file: str):
        """加载MD5索引用于去重"""
        import json
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            self.MD5_SET = {v['md5'] for v in index.values() if 'md5' in v}
            logger.info(f"加载MD5索引: {len(self.MD5_SET)} 个已存在文件")
        except Exception as e:
            logger.warning(f"加载MD5索引失败: {e}")
            self.MD5_SET = set()
    
    def _create_session(self) -> requests.Session:
        """创建请求会话"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/xml, */*',
            'Referer': 'https://example.com/',
        })
        return session
    
    def _parse_xml_response(self, xml_text: str) -> List[Dict]:
        """解析XML响应为字典列表"""
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(xml_text)
            posts = []
            for post in root.findall('post'):
                post_dict = {}
                for attr in post.attrib:
                    post_dict[attr] = post.attrib[attr]
                posts.append(post_dict)
            return posts
        except Exception as e:
            logger.error(f"XML解析失败: {e}")
            return []
    
    def search_posts(self, site: str, tags: str, page: int = 1, limit: int = 20) -> List[Dict]:
        """
        在指定站点搜索帖子
        
        Args:
            site: 站点名称
            tags: 搜索标签
            page: 页码
            limit: 每页数量
            
        Returns:
            List[Dict]: 帖子列表
        """
        site_info = self.MIRROR_SITES[site]
        api_url = site_info['api_url']
        session = self._create_session()
        
        try:
            if site_info.get('format') == 'xml':
                params = {
                    'tags': tags,
                    'pid': page - 1,
                    'limit': min(limit, 100),
                }
                response = session.get(api_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return self._parse_xml_response(response.text)
            else:
                params = {
                    'tags': tags,
                    'page': page,
                    'limit': min(limit, 200),
                }
                response = session.get(api_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"[{site_info['name']}] 搜索失败 [{tags}]: {e}")
            return []
    
    def get_all_posts(self, site: str, tags: str, max_posts: int = 50) -> List[Dict]:
        """
        获取所有匹配的帖子
        
        Args:
            site: 站点名称
            tags: 搜索标签
            max_posts: 最大获取数量
            
        Returns:
            List[Dict]: 帖子列表
        """
        all_posts = []
        page = 1
        limit = min(100, max_posts)
        
        while len(all_posts) < max_posts:
            posts = self.search_posts(site, tags, page, limit)
            
            if not posts:
                break
                
            for post in posts:
                if len(all_posts) >= max_posts:
                    break
                all_posts.append(post)
            
            page += 1
            time.sleep(random.uniform(0.5, 1.5))
        
        logger.info(f"[{self.MIRROR_SITES[site]['name']}] 搜索 '{tags}' 共找到 {len(all_posts)} 条结果")
        return all_posts
    
    def get_image_url(self, post: Dict, site: str) -> Optional[str]:
        """
        从帖子中提取图片URL
        
        Args:
            post: 帖子信息
            site: 站点名称
            
        Returns:
            Optional[str]: 图片URL
        """
        url_fields = ['file_url', 'source', 'image', 'url', 'preview_url', 'sample_url']
        
        for field in url_fields:
            if field in post:
                url = post[field]
                if url and not url.startswith('http'):
                    url = f"https://{site}.com{url}"
                return url
        
        return None
    
    def download_image(self, post: Dict, save_dir: str) -> bool:
        """
        下载单张图片（带MD5去重）
        
        Args:
            post: 帖子信息
            save_dir: 保存目录
            
        Returns:
            bool: 是否下载成功
        """
        image_url = self.get_image_url(post, self.sites[0])
        if not image_url:
            return False
        
        ext = image_url.split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            ext = 'jpg'
        
        post_id = post.get('id', post.get('md5', f"unknown_{random.randint(1, 9999)}"))
        file_path = Path(save_dir) / f"{post_id}.{ext}"
        
        if file_path.exists():
            return True
        
        session = self._create_session()
        try:
            response = session.get(image_url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            # 计算MD5用于去重
            content = response.content
            import hashlib
            md5 = hashlib.md5(content).hexdigest()
            
            # 检查是否已存在
            if md5 in self.MD5_SET:
                logger.debug(f"跳过重复文件 (MD5: {md5[:8]}...)")
                return False
            
            # 保存文件
            with open(file_path, 'wb') as f:
                f.write(content)
            
            # 更新MD5集合
            self.MD5_SET.add(md5)
            
            logger.debug(f"下载成功: {file_path.name}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"下载失败 [{image_url}]: {e}")
            return False
    
    def download_character_with_fallback(self, character_info: Dict, save_dir: str, 
                                        max_count: int = 30) -> Tuple[int, int]:
        """
        使用多站点和多标签策略下载角色图片
        
        Args:
            character_info: 角色信息字典
            save_dir: 保存目录
            max_count: 最大下载数量
            
        Returns:
            Tuple[int, int]: (成功数量, 失败数量)
        """
        chinese_name = character_info['chinese_name']
        danbooru_tag = character_info.get('danbooru_tag', '')
        work_en = character_info.get('work_en', '')
        
        # 构建搜索标签列表（按优先级）
        search_tags = []
        
        # 1. 优先使用已知的Danbooru标签
        if danbooru_tag:
            search_tags.append(danbooru_tag)
        
        # 2. 添加别名搜索（日文名、英文名等）
        if chinese_name in self.CHARACTER_ALIASES:
            for alias in self.CHARACTER_ALIASES[chinese_name]:
                search_tags.append(alias)
                # 带作品名的别名
                if work_en:
                    work_tag = work_en.lower().replace(' ', '_').replace(':', '_').replace('!', '').replace('?', '')
                    search_tags.append(f"{alias}_({work_tag})")
        
        # 3. 使用中文名拼音
        pinyin_tag = chinese_name.lower().replace(' ', '_').replace('·', '_')
        search_tags.append(pinyin_tag)
        
        # 去重
        search_tags = list(dict.fromkeys(search_tags))
        logger.info(f"[{chinese_name}] 搜索策略: {search_tags}")
        
        # 创建保存目录
        character_dir = Path(save_dir) / self._sanitize_filename(chinese_name)
        character_dir.mkdir(parents=True, exist_ok=True)
        
        all_posts = []
        downloaded_ids = set()
        
        # 遍历站点和标签进行搜索
        for site in self.sites:
            for tag in search_tags:
                if len(all_posts) >= max_count:
                    break
                
                # 添加安全过滤
                safe_tag = tag
                if not self.MIRROR_SITES[site].get('nsfw', False):
                    safe_tag = f"{tag} rating:safe"
                
                posts = self.get_all_posts(site, safe_tag, max_count - len(all_posts))
                
                for post in posts:
                    post_id = post.get('id', post.get('md5', str(random.randint(1, 99999))))
                    if post_id not in downloaded_ids:
                        downloaded_ids.add(post_id)
                        all_posts.append(post)
                
                if len(all_posts) >= max_count:
                    break
            
            if len(all_posts) >= max_count:
                break
        
        if not all_posts:
            logger.warning(f"角色 '{chinese_name}' 在所有站点都未找到匹配的图片")
            return (0, 0)
        
        # 下载图片
        success_count = 0
        fail_count = 0
        
        def download_with_progress(post_info: Tuple[int, Dict]) -> bool:
            idx, post = post_info
            result = self.download_image(post, str(character_dir))
            with self._download_lock:
                self._progress_counter += 1
                if self._progress_counter % 5 == 0 or self._progress_counter == len(all_posts):
                    logger.info(f"下载进度: [{self._progress_counter}/{len(all_posts)}]")
            return result
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(download_with_progress, (i, post)) 
                      for i, post in enumerate(all_posts, 1)]
            
            for future in as_completed(futures):
                try:
                    if future.result():
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
        
        logger.success(f"{chinese_name}: 成功 {success_count}, 失败 {fail_count}")
        return (success_count, fail_count)
    
    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名中的非法字符"""
        import re
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
        sanitized = sanitized.strip().strip('_')
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized


def load_roles_from_json(file_path: str) -> List[Dict]:
    """从roles.json加载角色列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持两种格式：{"characters": [...]} 或直接是数组
        if isinstance(data, dict):
            return data.get('characters', [])
        elif isinstance(data, list):
            return data
        else:
            return []
    except Exception as e:
        logger.error(f"加载roles.json失败: {e}")
        return []


def load_roles_from_csv(file_path: str) -> List[Dict]:
    """从roles.csv加载角色列表"""
    characters = []
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                characters.append({
                    'chinese_name': row['中文名'],
                    'work_title': row['作品名'],
                    'work_en': row['作品英文名'],
                    'english_name': row['角色英文名'],
                    'danbooru_tag': row['Danbooru标签'],
                })
        return characters
    except Exception as e:
        logger.error(f"加载roles.csv失败: {e}")
        return []


def load_roles_from_formatted(file_path: str) -> List[Dict]:
    """从formatted_roles.txt加载角色列表"""
    characters = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        characters.append({
                            'chinese_name': parts[0],
                            'work_en': parts[1],
                            'danbooru_tag': parts[2],
                        })
        return characters
    except Exception as e:
        logger.error(f"加载formatted_roles.txt失败: {e}")
        return []


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='多站点角色图片采集器 - 支持日文名和别名搜索')
    parser.add_argument('--site', type=str, nargs='+', default=['lolibooru', 'yande.re', 'konachan'],
                        help=f"镜像站点列表（按优先级）: {list(MultiSiteSpider.MIRROR_SITES.keys())}")
    parser.add_argument('--input-file', type=str, 
                        default='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/roles.json',
                        help='角色列表文件路径（支持json/csv/txt格式）')
    parser.add_argument('--output-dir', type=str, 
                        default='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/multi_site_images',
                        help='图片保存目录')
    parser.add_argument('--max-count', type=int, default=30,
                        help='每个角色最大下载数量')
    parser.add_argument('--start-from', type=int, default=0,
                        help='从第几个角色开始')
    parser.add_argument('--workers', type=int, default=8,
                        help='并发下载线程数')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='请求间隔延迟（秒）')
    parser.add_argument('--timeout', type=int, default=30,
                        help='请求超时时间（秒）')
    parser.add_argument('--md5-index', type=str, default=None,
                        help='MD5索引文件路径（用于去重）')
    
    args = parser.parse_args()
    
    # 加载角色列表
    input_path = Path(args.input_file)
    if input_path.suffix == '.json':
        characters = load_roles_from_json(args.input_file)
    elif input_path.suffix == '.csv':
        characters = load_roles_from_csv(args.input_file)
    elif input_path.suffix == '.txt':
        characters = load_roles_from_formatted(args.input_file)
    else:
        logger.error(f"不支持的文件格式: {input_path.suffix}")
        return
    
    if not characters:
        logger.error("未加载到角色列表")
        return
    
    logger.info(f"加载了 {len(characters)} 个角色")
    
    # 创建采集器
    spider = MultiSiteSpider(
        sites=args.site,
        max_workers=args.workers,
        delay=args.delay,
        timeout=args.timeout,
        md5_index_file=args.md5_index
    )
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 开始采集
    total_success = 0
    total_fail = 0
    
    for i, character in enumerate(characters[args.start_from:], start=args.start_from):
        chinese_name = character.get('chinese_name', '未知角色')
        logger.info(f"========== [{i+1}/{len(characters)}] 正在处理角色: {chinese_name} ==========")
        
        try:
            success, fail = spider.download_character_with_fallback(
                character, args.output_dir, args.max_count
            )
            total_success += success
            total_fail += fail
        except Exception as e:
            logger.error(f"处理角色 {chinese_name} 时发生错误: {e}")
        
        # 添加延迟
        time.sleep(random.uniform(args.delay * 0.5, args.delay * 1.5))
    
    # 输出汇总
    logger.info("========== 采集完成 ==========")
    logger.success(f"总计: 成功 {total_success}, 失败 {total_fail}")


if __name__ == '__main__':
    main()