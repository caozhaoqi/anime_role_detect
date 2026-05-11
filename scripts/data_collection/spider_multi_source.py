#!/usr/bin/env python3
"""使用多图片来源采集角色图片URL

支持的图片来源:
1. Pixiv镜像站点 (当前默认)
2. Danbooru (动漫图片站)
3. Gelbooru (动漫图片站)
4. Safebooru (安全动漫图片站)
5. Yande.re (动漫图片站)
"""
import requests
import time
import os
import json
from pathlib import Path
from datetime import datetime

API_BASE = "http://localhost:33333/api/v1.2.5.260305/sis"
REORGANIZED_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
SPIDER_DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data'
TARGET_COUNT = 50

# 需要采集的角色（图片不足50张）
ROLES_NEEDING_MORE = {
    'fu2li4xi1ya4': {'cn': '芙丽希娅', 'en': 'Furishia', 'jp': 'フリシア'},
    'qi2ta3': {'cn': '奇塔', 'en': 'Qita', 'jp': 'チータ'},
    'you4hu2': {'cn': '釉壶', 'en': 'Yuhu', 'jp': 'ユーフ'},
    'xiao3shan3': {'cn': '小闪', 'en': 'Xiao Shan', 'jp': 'シャオシャン'},
    'ke4luo2luo2': {'cn': '克萝萝', 'en': 'Kurolo', 'jp': 'クロロ'},
    'luo4ke4ke4': {'cn': '洛可可', 'en': 'Rokoko', 'jp': 'ロココ'},
    'ai4li4er3': {'cn': '爱丽儿', 'en': 'Ariel', 'jp': 'アリエル'}
}

# 图片来源配置
IMAGE_SOURCES = {
    'pixiv': {
        'name': 'Pixiv镜像',
        'enabled': True,
        'search_url': 'https://www.pixiv.net/tags/{keyword}/artworks',
        'api_endpoint': f'{API_BASE}/spider_start/single'
    },
    'danbooru': {
        'name': 'Danbooru',
        'enabled': True,
        'base_url': 'https://danbooru.donmai.us',
        'api_url': 'https://danbooru.donmai.us/posts.json',
        'params': {'tags': '{keyword}', 'limit': 100}
    },
    'gelbooru': {
        'name': 'Gelbooru',
        'enabled': True,
        'base_url': 'https://gelbooru.com',
        'api_url': 'https://gelbooru.com/index.php',
        'params': {'page': 'dapi', 's': 'post', 'q': 'index', 'json': 1, 'tags': '{keyword}', 'limit': 100}
    },
    'safebooru': {
        'name': 'Safebooru',
        'enabled': True,
        'base_url': 'https://safebooru.org',
        'api_url': 'https://safebooru.org/index.php',
        'params': {'page': 'dapi', 's': 'post', 'q': 'index', 'json': 1, 'tags': '{keyword}', 'limit': 100}
    },
    'yandere': {
        'name': 'Yande.re',
        'enabled': True,
        'base_url': 'https://yande.re',
        'api_url': 'https://yande.re/post.json',
        'params': {'tags': '{keyword}', 'limit': 100}
    }
}

def get_current_image_count(pinyin):
    """获取角色当前图片数量"""
    role_dir = os.path.join(REORGANIZED_DIR, pinyin)
    if not os.path.exists(role_dir):
        return 0
    count = 0
    for f in os.listdir(role_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            count += 1
    return count

def get_url_count(pinyin):
    """获取角色当前URL数量"""
    url_file = Path(f'{SPIDER_DATA_DIR}/img_url/{pinyin}_img.txt')
    if not url_file.exists():
        return 0
    with open(url_file, 'r', encoding='utf-8') as f:
        return len([line for line in f if line.strip()])

def record_source_info(pinyin, source_name, url_count, keyword_used):
    """记录图片来源信息到数据库/文件"""
    source_db_path = Path(f'{SPIDER_DATA_DIR}/source_info.json')
    
    # 读取现有数据
    data = {}
    if source_db_path.exists():
        try:
            with open(source_db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            data = {}
    
    # 更新数据
    if pinyin not in data:
        data[pinyin] = []
    
    data[pinyin].append({
        'source': source_name,
        'keyword': keyword_used,
        'url_count': url_count,
        'timestamp': datetime.now().isoformat()
    })
    
    # 保存数据
    with open(source_db_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def crawl_from_danbooru(keyword, pinyin):
    """从Danbooru采集图片URL"""
    print(f"    [Danbooru] 尝试: {keyword}")
    source = IMAGE_SOURCES['danbooru']
    
    try:
        params = {k: v.format(keyword=keyword) if isinstance(v, str) else v 
                  for k, v in source['params'].items()}
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        resp = requests.get(source['api_url'], params=params, headers=headers, timeout=30)
        
        if resp.status_code == 200:
            posts = resp.json()
            urls = []
            
            for post in posts:
                if 'file_url' in post:
                    urls.append(post['file_url'])
                elif 'large_file_url' in post:
                    urls.append(post['large_file_url'])
            
            # 保存URL
            if urls:
                url_file = Path(f'{SPIDER_DATA_DIR}/img_url/{pinyin}_img.txt')
                url_file.parent.mkdir(parents=True, exist_ok=True)
                
                existing_urls = set()
                if url_file.exists():
                    with open(url_file, 'r', encoding='utf-8') as f:
                        existing_urls = set(line.strip() for line in f if line.strip())
                
                new_urls = [url for url in urls if url not in existing_urls]
                
                with open(url_file, 'a', encoding='utf-8') as f:
                    for url in new_urls:
                        f.write(url + '\n')
                
                print(f"      ✅ 新增 {len(new_urls)} 个URL")
                record_source_info(pinyin, 'Danbooru', len(new_urls), keyword)
                return len(new_urls)
            else:
                print(f"      ⚠️ 未找到图片")
        else:
            print(f"      ❌ 请求失败: {resp.status_code}")
    except Exception as e:
        print(f"      ❌ 异常: {e}")
    
    return 0

def crawl_from_gelbooru(keyword, pinyin):
    """从Gelbooru采集图片URL"""
    print(f"    [Gelbooru] 尝试: {keyword}")
    source = IMAGE_SOURCES['gelbooru']
    
    try:
        params = {k: v.format(keyword=keyword) if isinstance(v, str) else v 
                  for k, v in source['params'].items()}
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        resp = requests.get(source['api_url'], params=params, headers=headers, timeout=30)
        
        if resp.status_code == 200:
            data = resp.json()
            posts = data.get('post', [])
            urls = []
            
            for post in posts:
                if 'file_url' in post:
                    urls.append(post['file_url'])
            
            if urls:
                url_file = Path(f'{SPIDER_DATA_DIR}/img_url/{pinyin}_img.txt')
                url_file.parent.mkdir(parents=True, exist_ok=True)
                
                existing_urls = set()
                if url_file.exists():
                    with open(url_file, 'r', encoding='utf-8') as f:
                        existing_urls = set(line.strip() for line in f if line.strip())
                
                new_urls = [url for url in urls if url not in existing_urls]
                
                with open(url_file, 'a', encoding='utf-8') as f:
                    for url in new_urls:
                        f.write(url + '\n')
                
                print(f"      ✅ 新增 {len(new_urls)} 个URL")
                record_source_info(pinyin, 'Gelbooru', len(new_urls), keyword)
                return len(new_urls)
            else:
                print(f"      ⚠️ 未找到图片")
        else:
            print(f"      ❌ 请求失败: {resp.status_code}")
    except Exception as e:
        print(f"      ❌ 异常: {e}")
    
    return 0

def crawl_from_safebooru(keyword, pinyin):
    """从Safebooru采集图片URL"""
    print(f"    [Safebooru] 尝试: {keyword}")
    source = IMAGE_SOURCES['safebooru']
    
    try:
        params = {k: v.format(keyword=keyword) if isinstance(v, str) else v 
                  for k, v in source['params'].items()}
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        resp = requests.get(source['api_url'], params=params, headers=headers, timeout=30)
        
        if resp.status_code == 200:
            data = resp.json()
            posts = data.get('post', [])
            urls = []
            
            for post in posts:
                if 'file_url' in post:
                    urls.append(post['file_url'])
            
            if urls:
                url_file = Path(f'{SPIDER_DATA_DIR}/img_url/{pinyin}_img.txt')
                url_file.parent.mkdir(parents=True, exist_ok=True)
                
                existing_urls = set()
                if url_file.exists():
                    with open(url_file, 'r', encoding='utf-8') as f:
                        existing_urls = set(line.strip() for line in f if line.strip())
                
                new_urls = [url for url in urls if url not in existing_urls]
                
                with open(url_file, 'a', encoding='utf-8') as f:
                    for url in new_urls:
                        f.write(url + '\n')
                
                print(f"      ✅ 新增 {len(new_urls)} 个URL")
                record_source_info(pinyin, 'Safebooru', len(new_urls), keyword)
                return len(new_urls)
            else:
                print(f"      ⚠️ 未找到图片")
        else:
            print(f"      ❌ 请求失败: {resp.status_code}")
    except Exception as e:
        print(f"      ❌ 异常: {e}")
    
    return 0

def crawl_from_yandere(keyword, pinyin):
    """从Yande.re采集图片URL"""
    print(f"    [Yande.re] 尝试: {keyword}")
    source = IMAGE_SOURCES['yandere']
    
    try:
        params = {k: v.format(keyword=keyword) if isinstance(v, str) else v 
                  for k, v in source['params'].items()}
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        resp = requests.get(source['api_url'], params=params, headers=headers, timeout=30)
        
        if resp.status_code == 200:
            posts = resp.json()
            urls = []
            
            for post in posts:
                if 'file_url' in post:
                    urls.append(post['file_url'])
                elif 'jpeg_url' in post:
                    urls.append(post['jpeg_url'])
            
            if urls:
                url_file = Path(f'{SPIDER_DATA_DIR}/img_url/{pinyin}_img.txt')
                url_file.parent.mkdir(parents=True, exist_ok=True)
                
                existing_urls = set()
                if url_file.exists():
                    with open(url_file, 'r', encoding='utf-8') as f:
                        existing_urls = set(line.strip() for line in f if line.strip())
                
                new_urls = [url for url in urls if url not in existing_urls]
                
                with open(url_file, 'a', encoding='utf-8') as f:
                    for url in new_urls:
                        f.write(url + '\n')
                
                print(f"      ✅ 新增 {len(new_urls)} 个URL")
                record_source_info(pinyin, 'Yande.re', len(new_urls), keyword)
                return len(new_urls)
            else:
                print(f"      ⚠️ 未找到图片")
        else:
            print(f"      ❌ 请求失败: {resp.status_code}")
    except Exception as e:
        print(f"      ❌ 异常: {e}")
    
    return 0

def spider_role_multi_source(pinyin, names):
    """使用多图片来源采集单个角色"""
    print(f"\n{'='*70}")
    print(f"🚀 开始采集: {names['cn']} ({pinyin})")
    print(f"{'='*70}")
    
    initial_count = get_url_count(pinyin)
    print(f"  当前URL数量: {initial_count}")
    print(f"  当前图片数量: {get_current_image_count(pinyin)}")
    
    total_added = 0
    
    # 尝试不同语言名称从不同来源采集
    keywords_to_try = [
        (names['cn'], '中文'),
        (names['en'], '英文'),
        (names['jp'], '日文')
    ]
    
    for keyword, lang in keywords_to_try:
        if total_added >= TARGET_COUNT:
            break
        
        print(f"\n  [{lang}] 关键词: {keyword}")
        
        # 从Danbooru采集
        if IMAGE_SOURCES['danbooru']['enabled']:
            added = crawl_from_danbooru(keyword, pinyin)
            total_added += added
            if added > 0:
                time.sleep(2)
        
        # 从Gelbooru采集
        if IMAGE_SOURCES['gelbooru']['enabled'] and total_added < TARGET_COUNT:
            added = crawl_from_gelbooru(keyword, pinyin)
            total_added += added
            if added > 0:
                time.sleep(2)
        
        # 从Safebooru采集
        if IMAGE_SOURCES['safebooru']['enabled'] and total_added < TARGET_COUNT:
            added = crawl_from_safebooru(keyword, pinyin)
            total_added += added
            if added > 0:
                time.sleep(2)
        
        # 从Yande.re采集
        if IMAGE_SOURCES['yandere']['enabled'] and total_added < TARGET_COUNT:
            added = crawl_from_yandere(keyword, pinyin)
            total_added += added
            if added > 0:
                time.sleep(2)
    
    final_count = get_url_count(pinyin)
    print(f"\n  采集完成: {names['cn']}")
    print(f"  URL数量: {initial_count} → {final_count} (+{final_count - initial_count})")
    
    return final_count - initial_count

def main():
    print("="*70)
    print("🌍 使用多图片来源采集角色图片")
    print("="*70)
    
    # 显示图片来源配置
    print("\n📋 图片来源配置:")
    for source_id, source_config in IMAGE_SOURCES.items():
        status = "✅ 启用" if source_config['enabled'] else "❌ 禁用"
        print(f"  {source_config['name']}: {status}")
    
    # 获取需要采集的角色
    roles_to_crawl = []
    for pinyin, names in ROLES_NEEDING_MORE.items():
        img_count = get_current_image_count(pinyin)
        if img_count < TARGET_COUNT:
            roles_to_crawl.append({
                'pinyin': pinyin,
                'names': names,
                'img_count': img_count,
                'needed': TARGET_COUNT - img_count
            })
    
    roles_to_crawl.sort(key=lambda x: x['needed'], reverse=True)
    
    print(f"\n找到 {len(roles_to_crawl)} 个角色需要补充图片")
    for role in roles_to_crawl:
        print(f"  {role['names']['cn']}: 当前 {role['img_count']} 张, 需要 {role['needed']} 张")
    
    # 开始采集
    total_added = 0
    for role in roles_to_crawl:
        added = spider_role_multi_source(role['pinyin'], role['names'])
        total_added += added
        time.sleep(3)
    
    print("\n" + "="*70)
    print(f"🎉 多来源采集完成！共新增 {total_added} 个URL")
    print("="*70)
    
    # 显示最终结果
    print("\n📊 最终统计:")
    for role in roles_to_crawl:
        pinyin = role['pinyin']
        names = role['names']
        img_count = get_current_image_count(pinyin)
        url_count = get_url_count(pinyin)
        status = "✅" if img_count >= TARGET_COUNT else "❌"
        print(f"  {status} {names['cn']}: {img_count} 张图片, {url_count} 个URL")

if __name__ == '__main__':
    main()
