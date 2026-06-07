#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多数据源高质量图片采集器"""

import os
import sys
import time
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from loguru import logger
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib

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
    if notification_manager:
        try:
            return notification_manager.send(message, title, level)
        except Exception as e:
            logger.warning(f"发送通知失败: {e}")
    return False

class MultiSourceSpider:
    """多数据源采集器"""
    
    # 多数据源配置
    SITES = {
        'yande.re': {
            'name': 'Yande.re',
            'api_url': 'https://yande.re/post.json',
            'format': 'json',
            'priority': 1,
            'requires_auth': False,
        },
        'safebooru': {
            'name': 'Safebooru',
            'api_url': 'https://safebooru.org/index.php?page=dapi&s=post&q=index',
            'format': 'xml',
            'priority': 2,
            'requires_auth': False,
        },
        'zerochan': {
            'name': 'Zerochan',
            'api_url': 'https://zerochan.net/',
            'format': 'html',
            'priority': 3,
            'requires_auth': False,
        },
        'anime-pictures': {
            'name': 'Anime-Pictures',
            'api_url': 'https://anime-pictures.net/api',
            'format': 'json',
            'priority': 4,
            'requires_auth': False,
        },
        'e-shuushuu': {
            'name': 'E-shuushuu',
            'api_url': 'https://e-shuushuu.net/search.php',
            'format': 'html',
            'priority': 5,
            'requires_auth': False,
        },
    }
    
    # 中文作品名到英文标签映射
    WORK_MAPPING = {
        '蔚蓝档案': 'blue_archive',
        '原神': 'genshin_impact',
        '崩坏星穹铁道': 'honkai:_star_rail',
        '崩坏3': 'honkai_impact_3rd',
        '崩坏学园2': 'honkai_academy_2',
        '鸣潮': 'wuthering_waves',
        '异环': 'endfield',
        '明日方舟': 'arknights',
        '碧蓝航线': 'azur_lane',
        '公主连接': 'princess_connect',
        '魔法少女小圆': 'madoka_magica',
        're:从零开始的异世界生活': 're:zero',
        '小林家的龙女仆': 'maidragon',
        '约会大作战': 'date_a_live',
        'fate/kaleid liner': 'fate/kaleid',
        '物语系列': 'monogatari',
        '请问您今天要来点兔子吗': 'gochuumon_usagi',
        '干物妹小埋': 'himouto_umaru',
        '埃罗芒阿老师': 'eromanga',
        '间谍过家家': 'spy_family',
        '偶像荣耀': 'idolmaster',
        '绝区零': 'zenless_zone_zero',
        '为美好的世界献上祝福': 'konosuba',
        '悠哉日常大王': 'yuru_yuri',
        '你的名字': 'kimi_no_na_wa',
        '声之形': 'koe_no_katachi',
        '葬送的芙莉莲': 'frieren',
        '我推的孩子': 'oshi_no_ko',
        '间谍过家家': 'spy_x_family',
    }
    
    def __init__(self, max_workers: int = 8, include_nsfw: bool = True, timeout: int = 30):
        self.max_workers = max_workers
        self.include_nsfw = include_nsfw
        self.timeout = timeout
        self.session = self._create_session()
        self._lock = threading.Lock()
        self._seen_hashes = set()  # 全局去重（基于MD5）
        self._total_downloaded = 0
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        return session
    
    def search_all(self, tags: str, max_posts: int = 100) -> List[Dict]:
        """从所有可用站点搜索"""
        all_posts = []
        
        for site_id in self.SITES:
            if len(all_posts) >= max_posts:
                break
            
            try:
                posts = self._search_site(site_id, tags, max_posts - len(all_posts))
                
                for post in posts:
                    # 基于多个字段生成唯一hash去重
                    unique_key = self._get_unique_key(post)
                    if unique_key and unique_key not in self._seen_hashes:
                        self._seen_hashes.add(unique_key)
                        all_posts.append(post)
                
                logger.info(f"[{self.SITES[site_id]['name']}] 找到 {len(posts)} 张")
                
            except Exception as e:
                logger.warning(f"[{self.SITES[site_id]['name']}] 搜索失败: {e}")
            
            # 站点间延迟
            time.sleep(random.uniform(0.5, 1.5))
        
        return all_posts[:max_posts]
    
    def _get_unique_key(self, post: Dict) -> Optional[str]:
        """生成唯一标识"""
        # 优先使用md5
        if 'md5' in post and post['md5']:
            return f"md5_{post['md5']}"
        # 其次使用id
        if 'id' in post and post['id']:
            return f"id_{post['id']}"
        # 使用file_url的hash
        if 'file_url' in post and post['file_url']:
            return hashlib.md5(post['file_url'].encode()).hexdigest()
        return None
    
    def _search_site(self, site_id: str, tags: str, max_posts: int) -> List[Dict]:
        """搜索单个站点"""
        site = self.SITES[site_id]
        
        if site_id == 'yande.re':
            return self._search_yandere(tags, max_posts)
        elif site_id == 'safebooru':
            return self._search_safebooru(tags, max_posts)
        elif site_id == 'zerochan':
            return self._search_zerochan(tags, max_posts)
        elif site_id == 'anime-pictures':
            return self._search_anime_pictures(tags, max_posts)
        elif site_id == 'e-shuushuu':
            return self._search_e_shuushuu(tags, max_posts)
        
        return []
    
    def _search_yandere(self, tags: str, max_posts: int) -> List[Dict]:
        """搜索 Yande.re"""
        all_posts = []
        page = 1
        
        # 尝试多种标签格式
        tag_formats = [
            tags,  # 原始格式
            tags.replace('_', ' '),  # 空格分隔
            tags.split('_(')[0] if '_(' in tags else tags,  # 只有角色名
        ]
        
        for tag_format in tag_formats:
            if len(all_posts) >= max_posts:
                break
            
            params = {
                'tags': tag_format,
                'page': page,
                'limit': min(100, max_posts)
            }
            
            try:
                response = self.session.get(
                    self.SITES['yande.re']['api_url'],
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                posts = response.json()
                
                if posts:
                    all_posts.extend(posts)
                    logger.info(f"[Yande.re] 标签'{tag_format}'找到 {len(posts)} 张")
                    break
                
            except requests.exceptions.Timeout:
                logger.warning("[Yande.re] 请求超时")
                break
            except Exception as e:
                logger.warning(f"[Yande.re] 搜索失败: {e}")
                break
            
            time.sleep(random.uniform(0.3, 0.8))
        
        return all_posts[:max_posts]
    
    def _search_safebooru(self, tags: str, max_posts: int) -> List[Dict]:
        """搜索 Safebooru (只返回安全内容)"""
        all_posts = []
        
        # 尝试多种标签格式
        tag_formats = [
            tags.replace('_', ' '),  # 空格分隔
            tags.split('_(')[0] if '_(' in tags else tags,  # 只有角色名
            tags,  # 原始格式
        ]
        
        for tag_format in tag_formats:
            if len(all_posts) >= max_posts:
                break
            
            params = {
                'tags': tag_format,
                'pid': 0,
                'limit': min(100, max_posts)
            }
            
            try:
                response = self.session.get(
                    self.SITES['safebooru']['api_url'],
                    params=params,
                    timeout=self.timeout
                )
                
                if response.status_code == 403 or response.status_code == 404:
                    break
                
                if response.status_code != 200:
                    continue
                
                # 解析XML
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
                
                posts_found = root.findall('post')
                if posts_found:
                    for post in posts_found:
                        post_dict = {
                            'id': post.get('id'),
                            'file_url': post.get('file_url'),
                            'preview_url': post.get('preview_url'),
                            'tags': post.get('tags'),
                            'md5': post.get('md5'),
                        }
                        all_posts.append(post_dict)
                    
                    logger.info(f"[Safebooru] 标签'{tag_format}'找到 {len(posts_found)} 张")
                    break
                    
            except Exception as e:
                logger.warning(f"[Safebooru] 搜索失败: {e}")
                continue
            
            time.sleep(random.uniform(0.5, 1.0))
        
        return all_posts[:max_posts]
    
    def _search_zerochan(self, tags: str, max_posts: int) -> List[Dict]:
        """搜索 Zerochan"""
        # Zerochan需要从HTML解析
        all_posts = []
        
        try:
            # 提取角色名作为搜索词
            search_term = tags.split('(')[0].strip().replace('_', '+')
            
            response = self.session.get(
                f"{self.SITES['zerochan']['api_url']}{search_term}",
                params={'s': 'id'},
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                return []
            
            # 简单的HTML解析
            import re
            # 匹配图片URL模式
            img_pattern = r'src="(https://s[0-9]+\.zerochan\.net/[^"]+\.(jpg|png|jpeg|gif))"'
            matches = re.findall(img_pattern, response.text)
            
            for i, (url, _) in enumerate(matches[:max_posts]):
                all_posts.append({
                    'id': f'zerochan_{i}',
                    'file_url': url,
                    'preview_url': url.replace('.img', '.thumb.img'),
                })
                
        except Exception as e:
            logger.warning(f"[Zerochan] 搜索失败: {e}")
        
        return all_posts[:max_posts]
    
    def _search_anime_pictures(self, tags: str, max_posts: int) -> List[Dict]:
        """搜索 Anime-Pictures.net"""
        all_posts = []
        
        try:
            params = {
                'search_tag': tags,
                'order_by': 'date',
                'ldate': '0',
                'page': '0'
            }
            
            response = self.session.get(
                f"{self.SITES['anime-pictures']['api_url']}/posts",
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                for post in data.get('posts', [])[:max_posts]:
                    all_posts.append({
                        'id': post.get('id'),
                        'file_url': post.get('file_url'),
                        'preview_url': post.get('preview_url'),
                        'tags': ' '.join(post.get('tags', [])),
                        'md5': post.get('md5'),
                    })
                    
        except Exception as e:
            logger.warning(f"[Anime-Pictures] 搜索失败: {e}")
        
        return all_posts[:max_posts]
    
    def _search_e_shuushuu(self, tags: str, max_posts: int) -> List[Dict]:
        """搜索 E-shuushuu"""
        all_posts = []
        
        try:
            params = {
                'keywords': tags.split('(')[0].strip(),
                'search': 'Search'
            }
            
            response = self.session.get(
                self.SITES['e-shuushuu']['api_url'],
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                import re
                # 匹配图片链接
                img_pattern = r'src="(https://e-shuushuu\.net/images/[^"]+\.(jpg|png|jpeg))"'
                matches = re.findall(img_pattern, response.text)
                
                for i, (url, _) in enumerate(matches[:max_posts]):
                    all_posts.append({
                        'id': f'eshuushuu_{i}',
                        'file_url': url,
                        'preview_url': url,
                    })
                    
        except Exception as e:
            logger.warning(f"[E-shuushuu] 搜索失败: {e}")
        
        return all_posts[:max_posts]
    
    def download_image(self, post: Dict, save_dir: Path) -> Tuple[bool, str]:
        """下载单张图片"""
        # 获取图片URL
        url_fields = ['file_url', 'source', 'image', 'url']
        image_url = None
        
        for field in url_fields:
            if field in post and post[field]:
                image_url = post[field]
                break
        
        if not image_url:
            return False, "无URL"
        
        # 获取扩展名
        ext = 'jpg'
        if '.' in image_url:
            possible_ext = image_url.split('.')[-1].lower()
            if possible_ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                ext = possible_ext
        
        # 生成文件名
        post_id = post.get('id', hashlib.md5(image_url.encode()).hexdigest()[:8])
        file_path = save_dir / f"{post_id}.{ext}"
        
        # 检查是否已存在
        if file_path.exists():
            return True, "已存在"
        
        try:
            response = self.session.get(image_url, stream=True, timeout=self.timeout)
            
            if response.status_code == 403 or response.status_code == 404:
                return False, f"HTTP {response.status_code}"
            
            response.raise_for_status()
            
            # 验证内容类型
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                return False, "HTML响应"
            
            # 写入文件
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 验证文件
            if file_path.stat().st_size < 1000:  # 小于1KB可能是错误图片
                file_path.unlink()
                return False, "文件过小"
            
            self._total_downloaded += 1
            return True, "成功"
            
        except Exception as e:
            return False, str(e)[:30]
    
    def collect_character(self, tag: str, save_dir: Path, target_count: int = 100) -> Tuple[int, int, int]:
        """采集单个角色"""
        character_dir = save_dir / tag
        
        if character_dir.exists():
            current_count = len(list(character_dir.glob("*.jpg"))) + len(list(character_dir.glob("*.png")))
        else:
            current_count = 0
            character_dir.mkdir(parents=True, exist_ok=True)
        
        if current_count >= target_count:
            return (0, 0, current_count)
        
        need_count = target_count - current_count
        logger.info(f"[{tag}] 当前{current_count}张，补充{need_count}张")
        
        # 获取搜索标签列表
        search_tags_list = self._prepare_search_tags(tag)
        
        # 从所有站点搜索
        all_posts = []
        for search_tag in search_tags_list:
            if len(all_posts) >= need_count:
                break
            
            posts = self.search_all(search_tag, need_count - len(all_posts))
            all_posts.extend(posts)
            
            if posts:
                logger.info(f"[{tag}] 标签'{search_tag}'找到 {len(posts)} 张")
        
        logger.info(f"[{tag}] 共找到 {len(all_posts)} 张图片")
        
        if not all_posts:
            return (0, 0, current_count)
        
        # 下载
        success = 0
        fail = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_image, post, character_dir): post 
                for post in all_posts
            }
            
            for future in as_completed(futures):
                ok, msg = future.result()
                if ok:
                    success += 1
                else:
                    fail += 1
        
        final_count = current_count + success
        logger.success(f"[{tag}] 补充完成: +{success}张 (总计{final_count}张)")
        
        return (success, fail, final_count)
    
    def _prepare_search_tags(self, tag: str) -> List[str]:
        """准备搜索标签列表"""
        tags_list = []
        
        if '(' in tag and ')' in tag:
            idx = tag.rfind('(')
            char_name = tag[:idx].strip('_')
            work_name = tag[idx+1:-1]
            
            # 转换作品名
            mapped_work = self.WORK_MAPPING.get(work_name, work_name)
            
            # 生成多种搜索组合
            tags_list = [
                char_name,  # 只角色名
                f"{char_name} {mapped_work}",  # 角色+作品
                char_name.replace('_', ' '),  # 角色名（空格）
            ]
        else:
            tags_list = [
                tag,
                tag.replace('_', ' '),
            ]
        
        return tags_list

def get_insufficient_roles(data_dir: Path, min_count: int = 100) -> List[str]:
    """获取不足目标数量的角色列表"""
    insufficient = []
    
    for role_dir in sorted(data_dir.iterdir(), key=lambda x: len(list(x.glob("*.*")))):
        if role_dir.is_dir():
            count = len(list(role_dir.glob("*.jpg"))) + len(list(role_dir.glob("*.png")))
            if count < min_count:
                insufficient.append((role_dir.name, count))
    
    return [role for role, _ in insufficient]

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='多数据源采集器')
    parser.add_argument('--character-file', type=str, help='角色名单文件')
    parser.add_argument('--output-dir', type=str, default='data/danbooru_images', help='数据目录')
    parser.add_argument('--target-count', type=int, default=100, help='目标数量')
    parser.add_argument('--workers', type=int, default=8, help='并发线程数')
    parser.add_argument('--delay', type=float, default=2.0, help='角色间延迟')
    parser.add_argument('--min-count', type=int, default=0, help='只处理小于此数量的角色')
    
    args = parser.parse_args()
    
    # 创建采集器
    spider = MultiSourceSpider(max_workers=args.workers, include_nsfw=True)
    
    # 获取角色列表
    if args.character_file:
        with open(args.character_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 解析角色名单
        characters = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                char_name = parts[0]  # 中文名
                work_name = parts[1]  # 作品名
                
                # 生成标签格式
                char_en = parts[2] if len(parts) > 2 else char_name.lower()
                work_en = spider.WORK_MAPPING.get(work_name, work_name.lower())
                
                tag = f"{char_en}_({work_en})"
                characters.append(tag)
    else:
        # 从现有目录获取不足角色
        characters = get_insufficient_roles(Path(args.output_dir), args.target_count)
    
    # 过滤已达标角色
    if args.min_count > 0:
        characters = [c for c in characters if get_insufficient_roles(Path(args.output_dir), 999)[0] if get_insufficient_roles(Path(args.output_dir), 999)[1] < args.min_count]
    
    logger.info(f"待采集角色: {len(characters)} 个")
    
    # 发送开始通知
    start_msg = f"""**🔍 多数据源采集开始**

**配置**:
- 数据源: Yande.re + Safebooru + Zerochan + Anime-Pictures + E-shuushuu
- 并发线程: {args.workers}
- 目标: {args.target_count}张/角色
- 角色数: {len(characters)} 个

**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
    send_notification(start_msg, "多数据源采集", "info")
    
    # 开始采集
    total_added = 0
    target_dir = Path(args.output_dir)
    
    for i, tag in enumerate(characters, 1):
        logger.info(f"[{i}/{len(characters)}] {tag}")
        
        try:
            added, failed, final = spider.collect_character(tag, target_dir, args.target_count)
            total_added += added
            
            # 进度通知
            if i % 5 == 0:
                progress_msg = f"""**📊 多数据源采集进度**

**进度**: {i}/{len(characters)} ({i/len(characters)*100:.1f}%)
**本批次新增**: +{added}张
**累计新增**: {total_added}张

**时间**: {time.strftime('%H:%M:%S')}"""
                send_notification(progress_msg, "采集进度", "info")
                
        except Exception as e:
            logger.error(f"采集 {tag} 失败: {e}")
        
        time.sleep(random.uniform(args.delay * 0.5, args.delay * 1.5))
    
    # 完成通知
    complete_msg = f"""**✅ 多数据源采集完成**

**统计**:
- 处理角色: {len(characters)} 个
- 新增图片: {total_added} 张

**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
    send_notification(complete_msg, "采集完成", "success")
    
    logger.success(f"采集完成！新增 {total_added} 张图片")

if __name__ == '__main__':
    main()
