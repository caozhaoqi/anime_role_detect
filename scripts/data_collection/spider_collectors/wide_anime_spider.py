#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""广泛二次元角色数据采集器 - 不限萝莉，采集各类二次元角色"""

import os
import sys
import time
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
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

class WideAnimeSpider:
    """广泛二次元角色采集器"""
    
    # 多数据源配置
    SITES = {
        'yande.re': {
            'name': 'Yande.re',
            'api_url': 'https://yande.re/post.json',
            'format': 'json',
            'priority': 1,
        },
        'safebooru': {
            'name': 'Safebooru',
            'api_url': 'https://safebooru.org/index.php?page=dapi&s=post&q=index',
            'format': 'xml',
            'priority': 2,
        },
        'konachan': {
            'name': 'Konachan',
            'api_url': 'https://konachan.com/post.json',
            'format': 'json',
            'priority': 3,
            'enabled': False,  # 已禁用：持续403错误
        },
    }
    
    # 扩展作品映射
    WORK_MAPPING = {
        # 游戏
        '蔚蓝档案': 'blue_archive',
        '原神': 'genshin_impact',
        '崩坏星穹铁道': 'honkai_star_rail',
        '崩坏3': 'honkai_impact_3rd',
        '崩坏学园2': 'honkai_academy_2',
        '鸣潮': 'wuthering_waves',
        '异环': 'endfield',
        '明日方舟': 'arknights',
        '碧蓝航线': 'azur_lane',
        '公主连接': 'princess_connect',
        '赛马娘': 'pretty_derby',
        'FGO': 'fate/grand_order',
        '战舰少女': 'kancolle',
        '少女前线': 'girls_frontline',
        '阴阳师': 'onmyoji',
        '绝区零': 'zenless_zone_zero',
        '蓝色协议': 'blue_protocol',
        '塔罗斯的法则': 'the_witness',
        
        # 动漫
        '进击的巨人': 'shingeki_no_kyojin',
        '咒术回战': 'jujutsu_kaisen',
        '鬼灭之刃': 'kimetsu_no_yaiba',
        '刀剑神域': 'sword_art_online',
        '莉可丽丝': 'lycoris_recoil',
        '孤独摇滚': 'bocchi_the_rock',
        'bang_dream': 'banG_Dream',
        '偶像大师': 'the_idolmaster',
        'LoveLive': 'love_live',
        '赛马娘': 'umamusume',
        'pop子': 'pop_team_epic',
        'jojo': 'jojo_no_kimyou_na_bouken',
        '电锯人': 'chainsaw_man',
        '夏日大作战': 'summer_wars',
        '龙珠': 'dragon_ball',
        '火影忍者': 'naruto',
        '海贼王': 'one_piece',
        '死神': 'bleach',
        '妖精的尾巴': 'fairy_tail',
        '美食的俘虏': 'toriko',
        '银魂': 'gin Tama',
        '家教': 'katekyo_hitman_reborn',
        '驱魔少年': 'dnangel',
        '滑头鬼之孙': 'natsume_yuujinchou',
        '结界师': 'kekkaishi',
        '犬夜叉': 'inuyasha',
        '翼·年代记': 'xxxholic',
        '四月是你的谎言': 'shigatsu_kimi_no_uso',
        '未闻花名': 'ano_hana',
        '四月新番': 'spring_2024',
        '七月新番': 'summer_2024',
        '十月新番': 'fall_2024',
        '一月新番': 'winter_2024',
        'Re:从零开始的异世界生活': 're_zero',
        '小林家的龙女仆': 'maidragon',
        '约会大作战': 'date_a_live',
        'Fate系列': 'fate_series',
        '魔法禁书目录': 'toaru_majutsu_no_index',
        '某科学的超电磁炮': 'toaru_kagaku_no_railgun',
        '魔法少女小圆': 'madoka_magica',
        '甘城光辉乐园': 'amagi_brilliant_park',
        '我的青春恋爱物语': 'yahari_ore_no_shaburi',
        '路人女主的养成方法': 'sakura_soft',
        '我的妹妹不可能这么可爱': 'ore_no_imouto',
        '埃罗芒阿老师': 'eromanga_sensei',
        '干物妹小埋': 'himouto_umaru',
        '请问您今天要来点兔子吗': 'gochuumon_usagi',
        '悠哉日常大王': 'yuru_yuri',
        '向阳素描': 'hidamari_sketch',
        '黄金拼图': 'kiniro_mosaic',
        '樱花庄的宠物女孩': 'sakura_soft',
        '冰果': 'hyouka',
        '轻音少女': 'k_on',
        '吹响吧上低音号': 'hibike_euphonium',
        '冰室理鸣': 'himari',
        '点兔': 'gochuumon',
    }
    
    def __init__(self, max_workers: int = 8, include_nsfw: bool = True, timeout: int = 30):
        self.max_workers = max_workers
        self.include_nsfw = include_nsfw
        self.timeout = timeout
        self.session = self._create_session()
        self._lock = threading.Lock()
        self._seen_hashes = set()
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
        """从所有站点搜索"""
        all_posts = []
        
        for site_id in self.SITES:
            if len(all_posts) >= max_posts:
                break
            
            try:
                posts = self._search_site(site_id, tags, max_posts - len(all_posts))
                
                for post in posts:
                    # 添加女角色过滤
                    if not self._is_female_character(post):
                        continue
                    
                    unique_key = self._get_unique_key(post)
                    if unique_key and unique_key not in self._seen_hashes:
                        self._seen_hashes.add(unique_key)
                        all_posts.append(post)
                
                if posts:
                    logger.info(f"[{self.SITES[site_id]['name']}] 找到 {len(posts)} 张")
                    
            except Exception as e:
                logger.warning(f"[{self.SITES[site_id]['name']}] 搜索失败: {e}")
            
            time.sleep(random.uniform(0.5, 1.5))
        
        return all_posts[:max_posts]
    
    def _is_female_character(self, post: Dict) -> bool:
        """检查是否为女角色"""
        # 检查post中的tags字段
        tags_str = ""
        
        # 优先使用tags字段
        if 'tags' in post and post['tags']:
            tags_str = str(post['tags']).lower()
        
        # 对于Yande.re/Danbooru格式，检查tag_list
        if 'tag_string' in post and post['tag_string']:
            tags_str = str(post['tag_string']).lower()
        
        # 检查是否为女角色标签
        female_indicators = [
            'female',
            '1girl',
            'solo_female',
            'girl',
            'female_only',
        ]
        
        male_indicators = [
            'male',
            '1boy',
            'solo_male',
            'boy',
            'male_only',
            'gender_swap',
        ]
        
        # 如果有男性标签，排除
        for indicator in male_indicators:
            if indicator in tags_str:
                return False
        
        # 如果有女性标签，包含
        for indicator in female_indicators:
            if indicator in tags_str:
                return True
        
        # 如果没有明确标签，默认包含（可能是女角色）
        return True
    
    def _get_unique_key(self, post: Dict) -> Optional[str]:
        if 'md5' in post and post['md5']:
            return f"md5_{post['md5']}"
        if 'id' in post and post['id']:
            return f"id_{post['id']}"
        if 'file_url' in post and post['file_url']:
            return hashlib.md5(post['file_url'].encode()).hexdigest()
        return None
    
    def _scan_existing_images(self, char_dir: Path) -> Set[str]:
        """扫描已存在的图片，返回图片ID集合"""
        existing_ids = set()
        
        if not char_dir.exists():
            return existing_ids
        
        for img_file in char_dir.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                # 从文件名提取ID（格式: id_{post_id}_{hash}.jpg）
                stem = img_file.stem
                parts = stem.split('_')
                if len(parts) >= 2 and parts[0] == 'id':
                    post_id = parts[1]
                    existing_ids.add(f"id_{post_id}")
        
        return existing_ids
    
    def _search_site(self, site_id: str, tags: str, max_posts: int) -> List[Dict]:
        site = self.SITES[site_id]
        
        # 跳过禁用的站点
        if not site.get('enabled', True):
            return []
        
        if site_id == 'yande.re':
            return self._search_json_api(site['api_url'], tags, max_posts)
        elif site_id == 'safebooru':
            return self._search_xml_api(site['api_url'], tags, max_posts)
        elif site_id == 'konachan':
            return self._search_json_api(site['api_url'], tags, max_posts)
        
        return []
    
    def _search_json_api(self, api_url: str, tags: str, max_posts: int) -> List[Dict]:
        """搜索 JSON API"""
        all_posts = []
        page = 1
        
        while len(all_posts) < max_posts:
            params = {
                'tags': tags,
                'page': page,
                'limit': min(100, max_posts)
            }
            
            try:
                response = self.session.get(api_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                posts = response.json()
                
                if not posts:
                    break
                
                all_posts.extend(posts)
                page += 1
                
            except Exception as e:
                logger.warning(f"[搜索] 请求失败: {e}")
                break
            
            time.sleep(random.uniform(0.3, 0.8))
        
        return all_posts[:max_posts]
    
    def _search_xml_api(self, api_url: str, tags: str, max_posts: int) -> List[Dict]:
        """搜索 XML API"""
        all_posts = []
        
        params = {
            'tags': tags,
            'pid': 0,
            'limit': min(100, max_posts)
        }
        
        try:
            response = self.session.get(api_url, params=params, timeout=self.timeout)
            
            if response.status_code != 200:
                return []
            
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as e:
                logger.warning(f"[Safebooru] XML解析错误: {e}")
                return []
            
            for post in root.findall('post'):
                try:
                    post_dict = {
                        'id': int(post.get('id', 0) or 0),
                        'file_url': post.get('file_url'),
                        'preview_url': post.get('preview_url'),
                        'tags': post.get('tags'),
                        'md5': post.get('md5'),
                    }
                    all_posts.append(post_dict)
                except (ValueError, TypeError):
                    continue
                
        except Exception as e:
            logger.warning(f"[Safebooru] 搜索失败: {e}")
        
        return all_posts[:max_posts]
    
    def download_image(self, post: Dict, save_dir: Path) -> Tuple[bool, str]:
        """下载图片"""
        url_fields = ['file_url', 'source', 'image', 'url']
        image_url = None
        
        for field in url_fields:
            if field in post and post[field]:
                image_url = post[field]
                break
        
        if not image_url:
            return False, "无URL"
        
        ext = 'jpg'
        if '.' in image_url:
            possible_ext = image_url.split('.')[-1].lower()
            if possible_ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                ext = possible_ext
        
        post_id = post.get('id', hashlib.md5(image_url.encode()).hexdigest()[:8])
        file_path = save_dir / f"{post_id}.{ext}"
        
        if file_path.exists():
            return True, "已存在"
        
        try:
            response = self.session.get(image_url, stream=True, timeout=self.timeout)
            
            if response.status_code in [403, 404]:
                return False, f"HTTP {response.status_code}"
            
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            if file_path.stat().st_size < 1000:
                file_path.unlink()
                return False, "文件过小"
            
            self._total_downloaded += 1
            return True, "成功"
            
        except Exception as e:
            return False, str(e)[:30]
    
    def collect_character(self, tag: str, save_dir: Path, target_count: int = 100) -> Tuple[int, int, int]:
        """采集角色"""
        character_dir = save_dir / tag
        
        # 扫描已存在的图片，避免重复下载
        existing_ids = self._scan_existing_images(character_dir)
        
        if character_dir.exists():
            current_count = len(list(character_dir.glob("*.jpg"))) + len(list(character_dir.glob("*.png")))
        else:
            current_count = 0
            character_dir.mkdir(parents=True, exist_ok=True)
        
        if current_count >= target_count:
            logger.info(f"[{tag}] 已达标 ({current_count}张)")
            return (0, 0, current_count)
        
        need_count = target_count - current_count
        logger.info(f"[{tag}] 当前{current_count}张，补充{need_count}张 (已跳过{len(existing_ids)}张本地已有)")
        
        # 生成搜索标签
        search_tags = self._prepare_search_tags(tag)
        
        # 搜索并过滤已存在的图片
        all_posts = []
        for st in search_tags:
            if len(all_posts) >= need_count:
                break
            posts = self.search_all(st, need_count - len(all_posts))
            
            # 过滤掉已存在的图片
            for post in posts:
                post_id = str(post.get('id', ''))
                if post_id and f"id_{post_id}" in existing_ids:
                    continue  # 跳过已存在的
                all_posts.append(post)
            
            if posts:
                logger.info(f"[{tag}] 标签'{st}'找到 {len(posts)} 张 (过滤后{len(all_posts)}张)")
        
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
                ok, _ = future.result()
                if ok:
                    success += 1
                else:
                    fail += 1
        
        final_count = current_count + success
        logger.success(f"[{tag}] 完成: +{success}张 (总计{final_count}张)")
        
        return (success, fail, final_count)
    
    def _prepare_search_tags(self, tag: str) -> List[str]:
        """准备搜索标签"""
        tags_list = []
        
        if '(' in tag and ')' in tag:
            idx = tag.rfind('(')
            char_name = tag[:idx].strip('_')
            work_name = tag[idx+1:-1]
            
            mapped_work = self.WORK_MAPPING.get(work_name, work_name)
            
            tags_list = [
                char_name,
                f"{char_name} {mapped_work}",
                char_name.replace('_', ' '),
            ]
        else:
            tags_list = [
                tag,
                tag.replace('_', ' '),
            ]
        
        return tags_list

def generate_wide_character_list() -> List[Dict]:
    """生成广泛的二次元角色列表"""
    characters = [
        # 原神
        {"name": "雷电将军", "tag": "raiden_shogun", "work": "genshin_impact"},
        {"name": "八重神子", "tag": "yae_miko", "work": "genshin_impact"},
        {"name": "神里绫华", "tag": "ayaka_genshin", "work": "genshin_impact"},
        {"name": "枫原万叶", "tag": "kaedehara_kazuha", "work": "genshin_impact"},
        {"name": "宵宫", "tag": "yoimiya", "work": "genshin_impact"},
        {"name": "珊瑚宫心海", "tag": "sangonomiya_kokomi", "work": "genshin_impact"},
        {"name": "荒泷一斗", "tag": "arataki_itto", "work": "genshin_impact"},
        {"name": "申鹤", "tag": "shenhe", "work": "genshin_impact"},
        {"name": "云堇", "tag": "yun_jin", "work": "genshin_impact"},
        {"name": "夜兰", "tag": "yelan", "work": "genshin_impact"},
        
        # 蔚蓝档案
        {"name": "悠间夏莲", "tag": "herel", "work": "blue_archive"},
        {"name": "黑见报道", "tag": "iroha", "work": "blue_archive"},
        {"name": "白鹤莲", "tag": "renge", "work": "blue_archive"},
        
        # 明日方舟
        {"name": "陈", "tag": "chen_arknights", "work": "arknights"},
        {"name": "年", "tag": "nian", "work": "arknights"},
        {"name": "令", "tag": "ling_arknights", "work": "arknights"},
        {"name": "能天使", "tag": "exusiai", "work": "arknights"},
        {"name": "德克萨斯", "tag": "texas", "work": "arknights"},
        {"name": "拉普兰德", "tag": "lappland", "work": "arknights"},
        {"name": "银灰", "tag": "silverash", "work": "arknights"},
        {"name": "星熊", "tag": "hoshiguma", "work": "arknights"},
        
        # 崩坏星穹铁道
        {"name": "景元", "tag": "jing_yuan", "work": "honkai_star_rail"},
        {"name": "彦卿", "tag": "yanqing", "work": "honkai_star_rail"},
        {"name": "丹恒", "tag": "dan_heng", "work": "honkai_star_rail"},
        {"name": "饮月", "tag": "dan_heng_imbibitor_lunae", "work": "honkai_star_rail"},
        {"name": "符玄", "tag": "fu_xuan", "work": "honkai_star_rail"},
        {"name": "罗刹", "tag": "luocha", "work": "honkai_star_rail"},
        
        # 咒术回战
        {"name": "钉崎野蔷薇", "tag": "nobara_kugisaki", "work": "jujutsu_kaisen"},
        {"name": "伏黑惠", "tag": "megumi_fushiguro", "work": "jujutsu_kaisen"},
        {"name": "真人", "tag": "gojo_satoru", "work": "jujutsu_kaisen"},
        
        # 鬼灭之刃
        {"name": "蝴蝶忍", "tag": "shinobu_kochou", "work": "kimetsu_no_yaiba"},
        {"name": "甘露寺蜜璃", "tag": "mitsuri_kanroji", "work": "kimetsu_no_yaiba"},
        {"name": "嘴平伊助", "tag": "zenitsu_agatsuma", "work": "kimetsu_no_yaiba"},
        
        # 刀剑神域
        {"name": "亚丝娜", "tag": "asuna", "work": "sword_art_online"},
        {"name": "爱丽丝", "tag": "alice_alicization", "work": "sword_art_online"},
        {"name": "莉法", "tag": "leafa", "work": "sword_art_online"},
        
        # 进击的巨人
        {"name": "三笠", "tag": "mikasa_ackerman", "work": "shingeki_no_kyojin"},
        {"name": "阿尔敏", "tag": "armin_arlert", "work": "shingeki_no_kyojin"},
        {"name": "韩吉", "tag": "hange_zoe", "work": "shingeki_no_kyojin"},
        
        # Fate系列
        {"name": "saber", "tag": "saber_fate", "work": "fate_series"},
        {"name": "远坂凛", "tag": "rin_tohsaka", "work": "fate_series"},
        {"name": "间桐樱", "tag": "sakura_matou", "work": "fate_series"},
        {"name": "伊莉雅", "tag": "ilya", "work": "fate_series"},
        
        # 孤独摇滚
        {"name": "后藤一里", "tag": "bocchi", "work": "bocchi_the_rock"},
        {"name": "伊地知虹夏", "tag": "nijika", "work": "bocchi_the_rock"},
        
        # 莉可丽丝
        {"name": "锦木千束", "tag": "chisato", "work": "lycoris_recoil"},
        {"name": "井之上泷奈", "tag": "takina", "work": "lycoris_recoil"},
        
        # 碧蓝航线
        {"name": "企业", "tag": "enterprise_azur_lane", "work": "azur_lane"},
        {"name": "贝尔法斯特", "tag": "belfast", "work": "azur_lane"},
        {"name": "厌战", "tag": "warspite", "work": "azur_lane"},
        
        # 赛马娘
        {"name": "特别周", "tag": "tokai_teio", "work": "pretty_derby"},
        {"name": "黄金船", "tag": "gold_ship", "work": "pretty_derby"},
        {"name": "无声铃鹿", "tag": "silent_suzuka", "work": "pretty_derby"},
        
        # LoveLive
        {"name": "高坂穗乃果", "tag": "honoka_kosaka", "work": "love_live"},
        {"name": "绚濑绘里", "tag": "eli_ayase", "work": "love_live"},
        {"name": "南小鸟", "tag": "kotori_minami", "work": "love_live"},
        
        # 葬送的芙莉莲
        {"name": "芙莉莲", "tag": "frieren", "work": "frieren"},
        {"name": "菲伦", "tag": "fern", "work": "frieren"},
        
        # 蓝色协议
        {"name": "公主连结", "tag": "princess_connect", "work": "princess_connect"},
        
        # 通用动漫角色
        {"name": "鹿目圆", "tag": "madoka_kaname", "work": "puella_magi_madoka_magica"},
        {"name": "晓美焰", "tag": "homura_akemi", "work": "puella_magi_madoka_magica"},
        {"name": "巴麻美", "tag": "mami_tomoe", "work": "puella_magi_madoka_magica"},
        
        # 更多游戏角色
        {"name": "狂三", "tag": "kuronotosaka_kurumi", "work": "date_a_live"},
        {"name": "十香", "tag": "toko_yayoi", "work": "date_a_live"},
        {"name": "折纸", "tag": "tachibana_tokomi", "work": "date_a_live"},
        
        # 鸣潮
        {"name": "吟霖", "tag": "yinlin", "work": "wuthering_waves"},
        {"name": "守岸人", "tag": "shorekeeper", "work": "wuthering_waves"},
        
        # 电锯人
        {"name": "玛奇玛", "tag": "makima", "work": "chainsaw_man"},
        {"name": "早川秋", "tag": "aki_hayakawa", "work": "chainsaw_man"},
        
        # 经典动漫
        {"name": "毛利兰", "tag": "ran_mouri", "work": "detective_conan"},
        {"name": "灰原哀", "tag": "ai_haibara", "work": "detective_conan"},
    ]
    
    return characters

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='广泛二次元角色采集器')
    parser.add_argument('--output-dir', type=str, default='data/wide_anime_images', help='输出目录')
    parser.add_argument('--target-count', type=int, default=100, help='目标数量')
    parser.add_argument('--workers', type=int, default=8, help='并发线程数')
    parser.add_argument('--delay', type=float, default=2.0, help='角色间延迟')
    
    args = parser.parse_args()
    
    spider = WideAnimeSpider(max_workers=args.workers, include_nsfw=True)
    
    # 生成角色列表
    characters = generate_wide_character_list()
    
    # 转换为标签格式
    tags_list = [f"{c['tag']}_({c['work']})" for c in characters]
    
    logger.info(f"待采集角色: {len(tags_list)} 个")
    
    # 发送开始通知
    start_msg = f"""**🌸 广泛二次元角色采集开始**

**配置**:
- 数据源: Yande.re + Safebooru + Konachan
- 角色数: {len(tags_list)} 个
- 目标: {args.target_count}张/角色
- 范围: 原神、蔚蓝档案、明日方舟、崩坏、Fate等

**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
    send_notification(start_msg, "广泛采集", "info")
    
    # 创建输出目录
    target_dir = Path(args.output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 开始采集
    total_added = 0
    
    for i, tag in enumerate(tags_list, 1):
        logger.info(f"[{i}/{len(tags_list)}] {tag}")
        
        try:
            added, failed, final = spider.collect_character(tag, target_dir, args.target_count)
            total_added += added
            
            # 进度通知
            if i % 10 == 0:
                progress_msg = f"""**📊 广泛采集进度**

**进度**: {i}/{len(tags_list)} ({i/len(tags_list)*100:.1f}%)
**本批次新增**: +{added}张
**累计新增**: {total_added}张

**时间**: {time.strftime('%H:%M:%S')}"""
                send_notification(progress_msg, "采集进度", "info")
                
        except Exception as e:
            logger.error(f"采集 {tag} 失败: {e}")
        
        time.sleep(random.uniform(args.delay * 0.5, args.delay * 1.5))
    
    # 完成通知
    complete_msg = f"""**✅ 广泛二次元角色采集完成**

**统计**:
- 处理角色: {len(tags_list)} 个
- 新增图片: {total_added} 张

**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
    send_notification(complete_msg, "采集完成", "success")
    
    logger.success(f"采集完成！新增 {total_added} 张图片")

if __name__ == '__main__':
    main()
