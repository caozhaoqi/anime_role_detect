#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版多站点角色图片采集器
功能：
1. 支持多个镜像站点轮换采集
2. 保存图片URL到txt文件（预防数据丢失）
3. 创建SQLite数据库存储采集记录
4. 支持日文名和别名搜索策略
5. MD5去重功能
"""

import os
import sys
import time
import random
import json
import csv
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


class EnhancedSpider:
    """增强版多站点角色图片采集器"""
    
    MIRROR_SITES = {
        'lolibooru': {
            'name': 'Lolibooru',
            'api_url': 'https://lolibooru.moe/post.json',
            'requires_auth': False,
            'rate_limit': 2,
            'format': 'json',
            'nsfw': False,
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
    
    CHARACTER_ALIASES = {
        '阿洛娜': ['アロナ', 'arona'],
        '普拉娜': ['プラナ', 'plana'],
        '砂狼白子': ['シロコ', 'shiroko', 'sunaookami_shiroko'],
        '圣园未花': ['ミカ', 'mika', 'mika_misono'],
        '空崎日奈': ['ヒナ', 'hina', 'hina_sorasaki'],
        '小鸟游星野': ['ホシノ', 'hoshino', 'hoshino_takanashi'],
        '纳西妲': ['ナヒダ', 'nahida'],
        '可莉': ['クレー', 'klee'],
        '七七': ['チチ', 'qiqi'],
        '早柚': ['サユ', 'sayu'],
        '胡桃': ['hutao', 'hu_tao'],
        '芙宁娜': ['フリーナ', 'furina'],
        '三月七': ['マーチ', 'march', 'march_7th'],
        '花火': ['スパークル', 'sparkle'],
        '克拉拉': ['クララ', 'clara'],
        '白露': ['bailu'],
        '琪露诺': ['チルノ', 'cirno'],
        '芙兰朵露': ['フラン', 'flandre', 'flandre_scarlet'],
        '蕾米莉亚': ['レミリア', 'remilia', 'remilia_scarlet'],
        '古明地恋': ['こいし', 'koishi', 'koishi_komeiji'],
        '洩矢诹访子': ['すわこ', 'suwako', 'suwako_moriya'],
        '铃仙·优昙华院·因幡': ['れいせん', 'reisen', 'reisen_udongein_inaba'],
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
                 md5_index_file: str = None, output_dir: str = None):
        self.sites = sites or ['lolibooru', 'yande.re', 'konachan']
        self.max_workers = max_workers
        self.delay = delay
        self.timeout = timeout
        self.output_dir = Path(output_dir or '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset')
        self._download_lock = threading.Lock()
        self._progress_counter = 0
        self.MD5_SET = set()
        
        # 验证站点
        self.sites = [s for s in self.sites if s in self.MIRROR_SITES]
        logger.info(f"使用采集站点: {[self.MIRROR_SITES[s]['name'] for s in self.sites]}")
        
        # 加载MD5索引
        if md5_index_file:
            self.load_md5_index(md5_index_file)
        
        # 初始化数据库和URL文件
        self._init_database()
        self._init_url_files()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        db_dir = self.output_dir.parent / 'database'
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / 'spider_records.db'
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 创建角色表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chinese_name TEXT NOT NULL UNIQUE,
                work_title TEXT,
                work_en TEXT,
                english_name TEXT,
                danbooru_tag TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建图片记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER,
                post_id TEXT,
                image_url TEXT NOT NULL,
                file_path TEXT,
                md5_hash TEXT,
                site TEXT,
                status TEXT DEFAULT 'pending',
                downloaded_at TIMESTAMP,
                FOREIGN KEY (character_id) REFERENCES characters(id),
                UNIQUE (image_url)
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_character_id ON images(character_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_md5_hash ON images(md5_hash)')
        
        conn.commit()
        conn.close()
        logger.info(f"数据库初始化完成: {self.db_path}")
    
    def _init_url_files(self):
        """初始化URL文件目录"""
        self.url_dir = self.output_dir.parent / 'urls'
        self.url_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建主URL文件
        self.all_urls_file = self.url_dir / 'all_image_urls.txt'
        if not self.all_urls_file.exists():
            self.all_urls_file.write_text('')
    
    def _get_character_id(self, chinese_name: str) -> int:
        """获取或创建角色ID"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM characters WHERE chinese_name = ?', (chinese_name,))
        result = cursor.fetchone()
        
        if result:
            conn.close()
            return result[0]
        
        # 创建新角色
        cursor.execute('INSERT INTO characters (chinese_name) VALUES (?)', (chinese_name,))
        conn.commit()
        char_id = cursor.lastrowid
        conn.close()
        return char_id
    
    def _save_image_record(self, character_id: int, post_id: str, image_url: str, 
                          site: str, file_path: str = None, md5_hash: str = None, 
                          status: str = 'pending'):
        """保存图片记录到数据库"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO images 
                (character_id, post_id, image_url, file_path, md5_hash, site, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (character_id, post_id, image_url, file_path, md5_hash, site, status))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"数据库写入失败: {e}")
        finally:
            conn.close()
    
    def _save_url_to_file(self, chinese_name: str, image_url: str):
        """保存URL到文本文件"""
        # 保存到角色专属文件
        char_url_file = self.url_dir / f"{self._sanitize_filename(chinese_name)}.txt"
        with open(char_url_file, 'a', encoding='utf-8') as f:
            f.write(f"{image_url}\n")
        
        # 保存到总URL文件
        with open(self.all_urls_file, 'a', encoding='utf-8') as f:
            f.write(f"{chinese_name}|{image_url}\n")
    
    def load_md5_index(self, index_file: str):
        """加载MD5索引用于去重"""
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            self.MD5_SET = {v['md5'] for v in index.values() if 'md5' in v}
            logger.info(f"加载MD5索引: {len(self.MD5_SET)} 个已存在文件")
        except Exception as e:
            logger.warning(f"加载MD5索引失败: {e}")
            self.MD5_SET = set()
    
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    ]
    
    def _create_session(self) -> requests.Session:
        """创建请求会话"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://lolibooru.moe/',
            'Connection': 'keep-alive',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"macOS"',
            'Sec-Fetch-Dest': 'image',
            'Sec-Fetch-Mode': 'no-cors',
            'Sec-Fetch-Site': 'cross-site',
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
        """在指定站点搜索帖子"""
        site_info = self.MIRROR_SITES[site]
        api_url = site_info['api_url']
        session = self._create_session()
        
        try:
            if site_info.get('format') == 'xml':
                params = {'tags': tags, 'pid': page - 1, 'limit': min(limit, 100)}
                response = session.get(api_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return self._parse_xml_response(response.text)
            else:
                params = {'tags': tags, 'page': page, 'limit': min(limit, 200)}
                response = session.get(api_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"[{site_info['name']}] 搜索失败 [{tags}]: {e}")
            return []
    
    def get_all_posts(self, site: str, tags: str, max_posts: int = 50) -> List[Dict]:
        """获取所有匹配的帖子"""
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
        """从帖子中提取图片URL"""
        url_fields = ['file_url', 'source', 'image', 'url', 'preview_url', 'sample_url']
        
        for field in url_fields:
            if field in post:
                url = post[field]
                if url and not url.startswith('http'):
                    url = f"https://{site}.com{url}"
                return url
        return None
    
    def _is_valid_image(self, content: bytes) -> bool:
        """检查内容是否为有效图片"""
        if len(content) < 100:
            return False
        
        # JPEG 文件头: FF D8
        if content[:2] == b'\xff\xd8':
            return True
        # PNG 文件头: 89 50 4E 47
        if content[:4] == b'\x89PNG':
            return True
        # WebP 文件头: RIFF....WEBP
        if content[:4] == b'RIFF' and content[8:12] == b'WEBP':
            return True
        # GIF 文件头: GIF87a 或 GIF89a
        if content[:6] in [b'GIF87a', b'GIF89a']:
            return True
        # BMP 文件头: BM
        if content[:2] == b'BM':
            return True
        
        # 检查是否是HTML（反爬返回的网页）
        if content[:5] in [b'<!DOC', b'<html', b'<!htm']:
            logger.debug("检测到HTML内容，可能是反爬拦截")
            return False
        
        return False
    
    def download_image(self, post: Dict, save_dir: str, character_id: int, site: str) -> bool:
        """下载单张图片（带MD5去重、图片验证和记录保存）"""
        image_url = self.get_image_url(post, site)
        if not image_url:
            return False
        
        ext = image_url.split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']:
            ext = 'jpg'
        
        post_id = post.get('id', post.get('md5', f"unknown_{random.randint(1, 9999)}"))
        file_path = Path(save_dir) / f"{post_id}.{ext}"
        
        if file_path.exists():
            return True
        
        session = self._create_session()
        try:
            response = session.get(image_url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            content = response.content
            
            # 验证图片有效性
            if not self._is_valid_image(content):
                logger.debug(f"无效图片内容，跳过: {image_url}")
                self._save_image_record(character_id, str(post_id), image_url, site, status='invalid')
                return False
            
            md5 = hashlib.md5(content).hexdigest()
            
            if md5 in self.MD5_SET:
                logger.debug(f"跳过重复文件 (MD5: {md5[:8]}...)")
                return False
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            self.MD5_SET.add(md5)
            
            # 保存记录到数据库和文件
            self._save_image_record(character_id, str(post_id), image_url, site, 
                                   str(file_path), md5, 'success')
            self._save_url_to_file(os.path.basename(save_dir), image_url)
            
            logger.debug(f"下载成功: {file_path.name} (大小: {len(content)//1024}KB)")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"下载失败 [{image_url}]: {e}")
            self._save_image_record(character_id, str(post_id), image_url, site, status='failed')
            return False
    
    def download_character_with_fallback(self, character_info: Dict, max_count: int = 30) -> Tuple[int, int]:
        """使用多站点和多标签策略下载角色图片"""
        chinese_name = character_info['chinese_name']
        danbooru_tag = character_info.get('danbooru_tag', '')
        work_en = character_info.get('work_en', '')
        
        # 获取角色ID
        character_id = self._get_character_id(chinese_name)
        
        # 构建搜索标签列表
        search_tags = []
        if danbooru_tag:
            search_tags.append(danbooru_tag)
        
        if chinese_name in self.CHARACTER_ALIASES:
            for alias in self.CHARACTER_ALIASES[chinese_name]:
                search_tags.append(alias)
                if work_en:
                    work_tag = work_en.lower().replace(' ', '_').replace(':', '_').replace('!', '').replace('?', '')
                    search_tags.append(f"{alias}_({work_tag})")
        
        pinyin_tag = chinese_name.lower().replace(' ', '_').replace('·', '_')
        search_tags.append(pinyin_tag)
        search_tags = list(dict.fromkeys(search_tags))
        
        logger.info(f"[{chinese_name}] 搜索策略: {search_tags}")
        
        # 创建保存目录
        character_dir = self.output_dir / self._sanitize_filename(chinese_name)
        character_dir.mkdir(parents=True, exist_ok=True)
        
        all_posts = []
        downloaded_ids = set()
        
        for site in self.sites:
            for tag in search_tags:
                if len(all_posts) >= max_count:
                    break
                
                safe_tag = tag
                if not self.MIRROR_SITES[site].get('nsfw', False):
                    safe_tag = f"{tag} rating:safe"
                
                posts = self.get_all_posts(site, safe_tag, max_count - len(all_posts))
                
                for post in posts:
                    post_id = post.get('id', post.get('md5', str(random.randint(1, 99999))))
                    if post_id not in downloaded_ids:
                        downloaded_ids.add(post_id)
                        all_posts.append((post, site))
                
                if len(all_posts) >= max_count:
                    break
            
            if len(all_posts) >= max_count:
                break
        
        if not all_posts:
            logger.warning(f"角色 '{chinese_name}' 在所有站点都未找到匹配的图片")
            return (0, 0)
        
        success_count = 0
        fail_count = 0
        
        def download_with_progress(post_info: Tuple[int, Tuple[Dict, str]]) -> bool:
            idx, (post, site) = post_info
            result = self.download_image(post, str(character_dir), character_id, site)
            with self._download_lock:
                self._progress_counter += 1
                if self._progress_counter % 5 == 0 or self._progress_counter == len(all_posts):
                    logger.info(f"下载进度: [{self._progress_counter}/{len(all_posts)}]")
            return result
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(download_with_progress, (i, post_info)) 
                      for i, post_info in enumerate(all_posts, 1)]
            
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
    
    def get_stats(self) -> Dict:
        """获取采集统计"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM characters')
        char_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM images')
        img_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM images WHERE status = ?', ('success',))
        success_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'characters': char_count,
            'total_images': img_count,
            'success_images': success_count,
            'database_path': str(self.db_path),
            'url_files_dir': str(self.url_dir)
        }


def load_roles_from_json(file_path: str) -> List[Dict]:
    """从roles.json加载角色列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get('characters', [])
        elif isinstance(data, list):
            return data
        else:
            return []
    except Exception as e:
        logger.error(f"加载roles.json失败: {e}")
        return []


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='增强版多站点角色图片采集器 - 支持URL保存和数据库存储')
    parser.add_argument('--site', type=str, nargs='+', default=['lolibooru', 'yande.re', 'konachan'],
                        help=f"镜像站点列表: {list(EnhancedSpider.MIRROR_SITES.keys())}")
    parser.add_argument('--input-file', type=str, 
                        default='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/roles.json',
                        help='角色列表文件路径')
    parser.add_argument('--output-dir', type=str, 
                        default='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset',
                        help='图片保存目录')
    parser.add_argument('--max-count', type=int, default=30, help='每个角色最大下载数量')
    parser.add_argument('--start-from', type=int, default=0, help='从第几个角色开始')
    parser.add_argument('--workers', type=int, default=8, help='并发下载线程数')
    parser.add_argument('--delay', type=float, default=2.0, help='请求间隔延迟')
    parser.add_argument('--timeout', type=int, default=30, help='请求超时时间')
    parser.add_argument('--md5-index', type=str, default=None, help='MD5索引文件路径')
    
    args = parser.parse_args()
    
    # 加载角色列表
    input_path = Path(args.input_file)
    if input_path.suffix == '.json':
        characters = load_roles_from_json(args.input_file)
    else:
        logger.error(f"不支持的文件格式: {input_path.suffix}")
        return
    
    if not characters:
        logger.error("角色列表为空")
        return
    
    logger.info(f"加载角色列表: {len(characters)} 个角色")
    
    # 创建采集器
    spider = EnhancedSpider(
        sites=args.site,
        max_workers=args.workers,
        delay=args.delay,
        timeout=args.timeout,
        md5_index_file=args.md5_index,
        output_dir=args.output_dir
    )
    
    # 开始采集
    total_success = 0
    total_fail = 0
    start_idx = args.start_from
    characters_to_collect = characters[start_idx:]
    
    logger.info(f"开始采集，从第 {start_idx + 1} 个角色开始，共 {len(characters_to_collect)} 个角色")
    
    for i, char in enumerate(characters_to_collect, start=start_idx):
        logger.info(f"\n=== [{i + 1}/{len(characters)}] 正在采集: {char['chinese_name']} ===")
        
        success, fail = spider.download_character_with_fallback(char, max_count=args.max_count)
        total_success += success
        total_fail += fail
        
        # 添加延迟避免请求过快
        if i < len(characters_to_collect) - 1:
            time.sleep(random.uniform(1.0, 2.0))
    
    # 输出统计
    logger.info("\n" + "="*60)
    logger.info("采集完成！")
    logger.info(f"总成功: {total_success}")
    logger.info(f"总失败: {total_fail}")
    
    stats = spider.get_stats()
    logger.info(f"数据库: {stats['database_path']}")
    logger.info(f"URL文件目录: {stats['url_files_dir']}")
    logger.info(f"已记录角色: {stats['characters']} 个")
    logger.info(f"已记录图片: {stats['total_images']} 张")
    logger.info("="*60)


if __name__ == '__main__':
    main()
