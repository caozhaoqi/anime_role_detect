#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从角色列表整理.md中读取角色并采集数据
"""

import os
import re
import argparse
import requests
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO
import hashlib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('character_expansion')

class CharacterDataExpander:
    def __init__(self, output_dir='data/train', max_workers=5, max_images=300):
        """
        初始化角色数据扩充器
        
        Args:
            output_dir: 输出目录
            max_workers: 最大并发数
            max_images: 每个角色的最大图像数
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.max_images = max_images
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 支持的数据源
        self.data_sources = {
            'safebooru': self._fetch_from_safebooru,
            'danbooru': self._fetch_from_danbooru,
            'gelbooru': self._fetch_from_gelbooru
        }
        
        # 角色系列映射
        self.series_mapping = {
            '崩坏 星穹铁道': 'honkai_star_rail',
            '幻塔': 'tower_of_fantasy',
            '绝区零': 'zenless_zone_zero',
            '轻音少女': 'k_on',
            '间谍过家家': 'spy_x_family',
            '鸣潮': 'wuthering_waves',
            '原神': 'genshin_impact'
        }
        
        # 角色标签映射
        self.character_tags = {}
        
    def parse_character_list(self, character_list_file):
        """
        解析角色列表文件
        
        Args:
            character_list_file: 角色列表文件路径
            
        Returns:
            角色列表
        """
        characters = []
        
        with open(character_list_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取系列和角色
        series_pattern = r'## (\d+)\. (.+)'
        character_pattern = r'- (.+)'
        
        current_series = None
        
        for line in content.split('\n'):
            # 匹配系列
            series_match = re.match(series_pattern, line)
            if series_match:
                current_series = series_match.group(2)
                continue
            
            # 匹配角色
            character_match = re.match(character_pattern, line)
            if character_match and current_series:
                character_name = character_match.group(1).strip()
                
                # 跳过空行和标题
                if not character_name or '角色列表' in character_name:
                    continue
                    
                # 跳过统计和备注部分
                if '角色数量' in character_name or '总角色数量' in character_name:
                    continue
                    
                characters.append({
                    'name': character_name,
                    'series': current_series
                })
        
        return characters
    
    def _fetch_from_safebooru(self, tags, limit=300):
        """
        从Safebooru获取图像
        """
        images = []
        
        try:
            url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&tags={'+'.join(tags)}&limit={limit}&json=1"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    for item in data:
                        if 'file_url' in item:
                            images.append(item['file_url'])
        
        except Exception as e:
            logger.warning(f"Safebooru获取图像失败: {e}")
        
        return images
    
    def _fetch_from_danbooru(self, tags, limit=300):
        """
        从Danbooru获取图像
        """
        images = []
        
        try:
            url = f"https://danbooru.donmai.us/posts.json?tags={'+'.join(tags)}&limit={limit}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    if 'file_url' in item:
                        images.append(item['file_url'])
        
        except Exception as e:
            logger.warning(f"Danbooru获取图像失败: {e}")
        
        return images
    
    def _fetch_from_gelbooru(self, tags, limit=300):
        """
        从Gelbooru获取图像
        """
        images = []
        
        try:
            url = f"https://gelbooru.com/index.php?page=dapi&s=post&q=index&tags={'+'.join(tags)}&limit={limit}&json=1"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'post' in data:
                    data = data['post']
                if isinstance(data, list):
                    for item in data:
                        if 'file_url' in item:
                            images.append(item['file_url'])
        
        except Exception as e:
            logger.warning(f"Gelbooru获取图像失败: {e}")
        
        return images
    
    def _generate_tags(self, character_name, series_name):
        """
        生成角色标签
        
        Args:
            character_name: 角色名称
            series_name: 系列名称
            
        Returns:
            标签列表
        """
        # 基础标签
        tags = []
        
        # 添加角色名称标签（转换为英文格式）
        english_name = self._chinese_to_english(character_name, series_name)
        if english_name:
            tags.append(english_name)
        
        # 添加系列标签
        series_english = self.series_mapping.get(series_name, series_name.lower().replace(' ', '_'))
        tags.append(series_english)
        
        return tags
    
    def _chinese_to_english(self, chinese_name, series_name):
        """
        将中文名称转换为英文标签
        
        Args:
            chinese_name: 中文名称
            series_name: 系列名称
            
        Returns:
            英文标签
        """
        # 常见角色名称映射
        name_mapping = {
            '崩坏 星穹铁道': {
                '三月七': 'march_7th',
                '丹恒': 'dan_heng',
                '丹恒·饮月': 'dan_heng_imbibitor_lunae',
                '丹恒•腾荒': 'dan_heng',
                '卡芙卡': 'kafka',
                '希儿': 'seele',
                '布洛妮娅': 'bronya_zaychik',
                '姬子': 'himeko',
                '瓦尔特': 'welt',
                '景元': 'jing_yuan',
                '银狼': 'silver_wolf',
                '罗刹': 'luocha',
                '符玄': 'fu_xuan',
                '刃': 'blade',
                '流萤': 'firefly',
                '花火': 'sparkle',
                '黑塔': 'herta',
                '停云': 'tingyun',
                '青雀': 'qingque',
                '白露': 'bailu',
                '娜塔莎': 'natasha',
                '克拉拉': 'clara',
                '杰帕德': 'gepard',
                '佩拉': 'pela',
                '桑博': 'sampo',
                '素裳': 'sushang',
                '彦卿': 'yanqing',
                '艾丝妲': 'asta',
                '虎克': 'hook',
                '阿兰': 'arlan',
                '米沙': 'misha',
                '桂乃芬': 'guinaifen',
                '寒鸦': 'raven',
                '雪衣': 'xueyi',
                '真理医生': 'dr_ratio',
                '知更鸟': 'robin',
                '砂金': 'aventurine',
                '黄泉': 'acheron',
                '黑天鹅': 'black_swan',
                '波提欧': 'boothill',
                '椒丘': 'jiaoqiu',
                '飞霄': 'feixiao',
                '云璃': 'yunli',
                '藿藿': 'huohuo',
                '灵砂': 'lingsha',
                '托帕': 'topaz',
                '镜流': 'jingliu',
                '阮·梅': 'ruan_mei',
                '忘归人': 'dr_ratio',
                '加拉赫': 'gallagher',
                '乱破': 'rappa',
                '星期日': 'sunday',
                '大黑塔': 'herta',
                '大丽花': 'dahlia',
                '艾莉丝': 'alice',
                '柯诺': 'cocolia',
                '可可利亚': 'cocolia',
                '希露瓦': 'serval',
                '卢卡': 'luca',
                '银枝': 'argenti',
                '貊泽': 'moze',
                '赛飞儿': 'sophie',
                '遐蝶': 'xiadie',
                '那刻夏': 'naxixia',
                '长夜月': 'longxia',
                '风堇': 'fengjin',
                '驭空': 'yukong',
                '缇宝': 'tibao',
                '翠钰': 'cuiyu',
                '白厄': 'baie',
                '阿格莱雅': 'aglaea',
                '刻律德菈': 'clyde',
                '昔涟': 'xilian',
                '爻光': 'yaoguang',
                '玲可': 'lingke',
                '海瑟音': 'haisiyin'
            },
            '原神': {
                '温迪': 'venti',
                '迪卢克': 'diluc',
                '刻晴': 'keqing',
                '甘雨': 'ganyu',
                '胡桃': 'hu_tao',
                '雷电将军': 'raiden_shogun',
                '神里绫华': 'kamisato_ayaka',
                '荒泷一斗': 'arataki_itto',
                '八重神子': 'yae_miko',
                '公子': 'childe',
                '芙宁娜': 'furina'
            },
            '崩坏三': {
                '琪亚娜': 'kiana_kaslana',
                '芽衣': 'raiden_mei',
                '布洛妮娅': 'bronya_zaychik',
                '德丽莎': 'theresa_apocalypse',
                '雷电芽衣': 'raiden_mei',
                '琪亚娜·卡斯兰娜': 'kiana_kaslana'
            },
            '绝区零': {
                '妮可·德玛拉': 'nicole_demara',
                '安比·德玛拉': 'anby_demara',
                '比利·奇德': 'billy_kidd',
                '格莉丝·霍华德': 'grace_howard',
                '冯·莱卡恩': 'von_lycaon',
                '可琳·威克斯': 'corin_wickes',
                '珂蕾妲·贝洛伯格': 'koleda_belobog',
                '本·比格': 'ben_bigger',
                '安东·伊万诺夫': 'anton_ivanov',
                '星见雅': 'miyabi',
                '简·杜': 'jane_doe',
                '苍角': 'seth_lowell',
                '艾莲·乔': 'ellen_joe',
                '露西': 'lucy',
                '派派·韦尔': 'piper_wheel',
                '猫宫又奈': 'soukaku',
                '青衣': 'qingyi',
                '朱鸢': 'zhu_yuan',
                '莱特': 'lighter',
                '柏妮思·怀特': 'burnice_white',
                '月城柳': 'yuecheng_liu',
                '凯撒·金': 'caesar_king',
                '亚历山德丽娜·莎芭丝缇安': 'alexandrina_sebastiane',
                '伊德海莉': 'idah',
                '伊芙琳': 'evelyn',
                '仪玄': 'yixuan',
                '席德': 'sid',
                '扳机': 'trigger',
                '奥菲丝·马格努森 &「鬼火」': 'ophelia_magnuson',
                '哲&铃': 'zhe_ling',
                '橘福福': 'tangerine',
                '波可娜·费雷尼': 'pocena_ferney',
                '浅羽悠真': 'asaba_yuma',
                '潘引壶': 'panyinhu',
                '琉音': 'liuyin',
                '薇薇安': 'vivian',
                '赛斯·洛威尔': 'seth_lovell',
                '银心锡兵·安比': 'anby_demara',
                '雨果': 'hugo',
                '耀嘉音': 'yaojiayin'
            },
            '幻塔': {
                '克劳迪娅': 'claudia',
                '克劳迪娅·风暴眼': 'claudia',
                '奈美西斯': 'nemesis',
                '奈美西斯·裂空': 'nemesis',
                '夏佐': 'shiro',
                '可可丽特': 'cocoritter',
                '菲欧娜': 'fiona',
                '乌米': 'umi',
                '四枫院羽': 'shikikan_hane',
                '明景': 'mingjing',
                '凌寒': 'linghan',
                '凛夜': 'linye',
                '天琅': 'tianlang',
                '艾莉丝': 'alice',
                '希佩尔': 'hipper',
                '帕洛蒂': 'parody',
                '格诺诺': 'gonono',
                '胡萝贝': 'huluobei',
                '蕾贝': 'leibei',
                '钴蓝': 'cobalt',
                '雅诺': 'yano',
                '默利尼娅': 'melinia',
                '不破咲': 'fusaki',
                '乌丸': 'kumaru',
                '九域': 'jiuyu',
                '亚夏': 'yaxia',
                '伊卡洛斯': 'icarus',
                '修玛': 'xiuma',
                '南音': 'nanyin',
                '启罗': 'qiluo',
                '妃色': 'fese',
                '姬玉': 'jiyu',
                '安可': 'anke',
                '安娜贝拉': 'annabell',
                '安托莉亚': 'antoria',
                '布勒薇': 'blevi',
                '弗丽嘉': 'freya',
                '星寰': 'xinghuan',
                '格网': 'gewang',
                '梅丽尔': 'meriel',
                '榴火': 'liuhuo',
                '洛斯琳': 'loslin',
                '灰狐': 'gray_fox',
                '烟渺': 'yanmiao',
                '玉兰': 'yulan',
                '绫波零': 'rei_ayanami',
                '维拉': 'viera',
                '艾斯特': 'ester',
                '芬璃尔': 'fenlier',
                '茵纳斯': 'yinna',
                '莎莉': 'sally',
                '蕾比莉亚': 'rebilia',
                '西萝': 'xiluo',
                '赛弥尔': 'saimier',
                '阿斯拉达': 'asurada',
                '阿格蕾娅': 'agreya'
            },
            '轻音少女': {
                '平泽唯': 'yui_hirasawa',
                '秋山澪': 'mio akiyama',
                '田井中律': 'ritsu_tainaka',
                '琴吹䌷': 'tsumugi_kotobuki',
                '中野梓': 'azusa_nakano',
                '山中佐和子': 'sawako_yamanaka',
                '平泽忧': 'ui_hirasawa',
                '真锅和': 'nami_minami',
                '曾我部惠': 'megumi_sogabe',
                '铃木纯': 'jun_suzuki'
            },
            '间谍过家家': {
                '阿尼亚·福杰': 'anya_forger',
                '邦德·福杰': 'bond_forger'
            },
            '鸣潮': {
                '洛瑟菈': 'rosetta',
                '若那星火': 'ruona_xinghuo',
                '先行星炬学院校长': 'principal',
                '若那期望满溢': 'ruona_qiwangmany',
                '西格莉卡': 'sigrika'
            }
        }
        
        # 获取对应系列的角色映射
        series_mapping = name_mapping.get(series_name, {})
        
        # 查找角色名称
        for cn_name, en_name in series_mapping.items():
            if cn_name in chinese_name or chinese_name in cn_name:
                return en_name
        
        # 如果找不到映射，返回None
        return None
    
    def _download_image(self, url, save_path):
        """
        下载图像
        
        Args:
            url: 图像URL
            save_path: 保存路径
            
        Returns:
            是否成功
        """
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                
                # 转换为RGB（处理RGBA图像）
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                img.save(save_path, 'JPEG', quality=95)
                return True
        except Exception as e:
            logger.warning(f"下载图像失败 {url}: {e}")
        
        return False
    
    def _get_existing_images(self, character_dir):
        """
        获取已存在的图像数量
        
        Args:
            character_dir: 角色目录
            
        Returns:
            已存在的图像数量
        """
        if not os.path.exists(character_dir):
            return 0
        
        count = 0
        for file in os.listdir(character_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                count += 1
        
        return count
    
    def _get_image_hash(self, image_path):
        """
        计算图像的哈希值
        
        Args:
            image_path: 图像路径
            
        Returns:
            哈希值
        """
        try:
            with Image.open(image_path) as img:
                # 调整大小以加快计算
                img = img.resize((8, 8), Image.LANCZOS)
                # 转换为灰度
                img = img.convert('L')
                # 计算哈希
                pixels = list(img.getdata())
                avg = sum(pixels) / len(pixels)
                bits = ''.join('1' if pixel > avg else '0' for pixel in pixels)
                return bits
        except:
            return None
    
    def _is_duplicate(self, image_path, existing_hashes):
        """
        检查图像是否重复
        
        Args:
            image_path: 图像路径
            existing_hashes: 已存在的哈希值集合
            
        Returns:
            是否重复
        """
        img_hash = self._get_image_hash(image_path)
        if img_hash and img_hash in existing_hashes:
            return True
        return False
    
    def collect_character_images(self, character, series):
        """
        为角色收集图像
        
        Args:
            character: 角色名称
            series: 系列名称
            
        Returns:
            收集的图像数量
        """
        # 创建角色目录
        series_english = self.series_mapping.get(series, series.lower().replace(' ', '_'))
        character_dir = os.path.join(self.output_dir, f"{series_english}_{character}")
        os.makedirs(character_dir, exist_ok=True)
        
        # 获取已存在的图像数量
        existing_count = self._get_existing_images(character_dir)
        needed = self.max_images - existing_count
        
        if needed <= 0:
            logger.info(f"{series}_{character}: 已有 {existing_count} 张图像，无需采集")
            return 0
        
        logger.info(f"开始为 {series}_{character} 收集 {needed} 张图像")
        
        # 生成标签
        tags = self._generate_tags(character, series)
        
        # 获取已存在的图像哈希
        existing_hashes = set()
        for file in os.listdir(character_dir):
            file_path = os.path.join(character_dir, file)
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_hash = self._get_image_hash(file_path)
                if img_hash:
                    existing_hashes.add(img_hash)
        
        # 从各个数据源收集图像
        all_image_urls = []
        for source_name, source_func in self.data_sources.items():
            logger.info(f"从 {source_name} 获取图像")
            image_urls = source_func(tags, limit=needed)
            all_image_urls.extend(image_urls)
            
            if len(all_image_urls) >= needed:
                break
        
        # 下载图像
        downloaded = 0
        for i, url in enumerate(tqdm(all_image_urls[:needed], desc=f"下载 {series}_{character} 图像")):
            if downloaded >= needed:
                break
            
            # 生成临时文件名
            temp_path = os.path.join(character_dir, f"temp_{i}.jpg")
            
            # 下载图像
            if self._download_image(url, temp_path):
                # 检查是否重复
                if not self._is_duplicate(temp_path, existing_hashes):
                    # 生成最终文件名
                    final_path = os.path.join(character_dir, f"{existing_count + downloaded + 1}.jpg")
                    os.rename(temp_path, final_path)
                    
                    # 更新哈希集合
                    img_hash = self._get_image_hash(final_path)
                    if img_hash:
                        existing_hashes.add(img_hash)
                    
                    downloaded += 1
                else:
                    # 删除重复图像
                    os.remove(temp_path)
        
        logger.info(f"{series}_{character} 数据收集完成，当前共有 {existing_count + downloaded} 张图像")
        return downloaded
    
    def expand_dataset(self, character_list_file):
        """
        扩充数据集
        
        Args:
            character_list_file: 角色列表文件路径
        """
        # 解析角色列表
        logger.info(f"开始解析角色列表文件: {character_list_file}")
        characters = self.parse_character_list(character_list_file)
        logger.info(f"共解析出 {len(characters)} 个角色")
        
        # 按系列分组
        series_characters = {}
        for char in characters:
            series = char['series']
            if series not in series_characters:
                series_characters[series] = []
            series_characters[series].append(char['name'])
        
        # 显示统计信息
        logger.info("=" * 60)
        logger.info("角色分布统计")
        logger.info("=" * 60)
        for series, chars in series_characters.items():
            logger.info(f"{series}: {len(chars)} 个角色")
        logger.info("=" * 60)
        
        # 为每个角色收集图像
        total_downloaded = 0
        for char in tqdm(characters, desc="收集角色数据"):
            downloaded = self.collect_character_images(char['name'], char['series'])
            total_downloaded += downloaded
        
        logger.info(f"数据集扩充完成，共下载 {total_downloaded} 张图像")

def main():
    parser = argparse.ArgumentParser(description='从角色列表文件中读取角色并采集数据')
    parser.add_argument('--character-list', type=str, 
                       default='auto_spider_img/characters/角色列表整理.md',
                       help='角色列表文件路径')
    parser.add_argument('--output-dir', type=str, 
                       default='data/train',
                       help='输出目录')
    parser.add_argument('--max-images', type=int, 
                       default=300,
                       help='每个角色的最大图像数')
    parser.add_argument('--max-workers', type=int, 
                       default=5,
                       help='最大并发数')
    
    args = parser.parse_args()
    
    # 创建数据扩充器
    expander = CharacterDataExpander(
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        max_images=args.max_images
    )
    
    # 扩充数据集
    expander.expand_dataset(args.character_list)

if __name__ == '__main__':
    main()