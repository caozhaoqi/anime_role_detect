#!/usr/bin/env python3
"""为图片不足50张的角色下载图片"""
import requests
import hashlib
from pathlib import Path

REORGANIZED_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
TARGET_COUNT = 50

PINYIN_MAPPING = {
    'a1luo4na4': '阿洛娜',
    'pu3la1na4': '普拉娜',
    'na4xi1da2': '纳西妲',
    'ti2bao3': '缇宝',
    'ke3li4': '可莉',
    'di2ao4na4': '迪奥娜',
    'yao2yao2': '瑶瑶',
    'xi1ge2wen2': '希格雯',
    'lei3bei4': '蕾贝',
    'hei1ta3': '黑塔',
    'fu2xuan2': '符玄',
    'qi1qi1': '七七',
    'zao3you4': '早柚',
    'duo1li4': '多莉',
    'ka3qi2na4': '卡齐娜',
    'san1yue4qi1': '三月七',
    'hua1huo3': '花火',
    'yin2lang2': '银狼',
    'tian1tong2ai4li4si1': '天童爱丽丝',
    'zao3wu4': '早雾',
    'wei2li3nai4': '维里奈',
    'an1ke3': '安可',
    'you4hu2': '釉瑚',
    'lu4mu4yuan2': '鹿目圆',
    'xiao3mei3yan4': '晓美焰',
    'xue4xiao3ban3': '血小板',
    'lei2mu3': '雷姆',
    'la1mu3': '拉姆',
    'kang1na4': '康娜',
    'si4mi4nai3': '四糸乃',
    'kai3lu4': '凯露',
    'yi1li4ya3': '伊莉雅',
    'ren3ye3ren3': '忍野忍',
    'zhi4nai3': '智乃',
    'xiao3mai2': '小埋',
    'sha1wu4': '纱雾',
    'mao1gong1you4nai4': '猫宫又奈',
    'de2li4sha1': '德丽莎',
    'bu4luo4ni2ya4': '布洛妮娅',
    'ke3lin2': '可琳',
    'shen1yue4': '神乐',
    'bai2shang4chui1xue3': '白上吹雪',
    'yue4qian1ye4': '月千夜',
    'li4ta3la1': '莉塔拉',
    'wei2pu3lei3': '维普蕾',
    'xia4ke4li3': '夏克里',
    'na4gan1': '纳甘',
    'ke1xie4ni2ya4': '科谢尼娅',
    'kou4er3fu2': '寇尔芙',
    'ke4luo2li4ke1': '克罗丽科',
    'pei4li3ti2ya4': '佩里缇亚',
    'a1ni4ya4': '阿尼亚',
    'luo4qian4': '洛茜',
    'mi2dou4zi': '祢豆子',
    'xi1er3': '希儿',
    'xing4': '杏',
    'yi1se4lin2': '伊瑟琳',
    'fu2lan2': '芙兰',
    'fei1mi3li4si1': '菲米莉丝',
    'ke4la1la1': '克拉拉'
}

def get_current_count(pinyin):
    """获取角色当前图片数量"""
    img_dir = Path(REORGANIZED_DIR) / pinyin
    if not img_dir.exists():
        return 0
    return len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png'))) + len(list(img_dir.glob('*.webp')))

def download_role_images(role_name, role_pinyin, target_count=TARGET_COUNT):
    """为单个角色下载图片"""
    img_dir = Path(REORGANIZED_DIR) / role_pinyin
    img_dir.mkdir(parents=True, exist_ok=True)
    
    url_file = Path(f'spider_image_system/data/img_url/{role_pinyin}_img.txt')
    if not url_file.exists():
        print(f"❌ {role_name}: 未找到URL文件")
        return 0
    
    with open(url_file, 'r', encoding='utf-8') as f:
        urls = [l.strip() for l in f if l.strip()]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.pixiv.net/'
    })
    
    existing_hashes = set()
    for img_path in img_dir.glob('*'):
        if img_path.is_file():
            try:
                with open(img_path, 'rb') as f:
                    existing_hashes.add(hashlib.md5(f.read()).hexdigest())
            except:
                pass
    
    current_count = len(existing_hashes)
    print(f"\n📁 {role_name} ({role_pinyin}): {len(urls)} 个URL, 当前 {current_count} 张")
    
    if current_count >= target_count:
        print(f"   ⏭️ 已达标，无需补充")
        return 0
    
    needed = target_count - current_count
    downloaded = 0
    failed = 0
    
    for url in urls:
        if downloaded >= needed:
            break
        
        try:
            resp = session.get(url, timeout=15)
            content = resp.content
            
            if len(content) < 1024:
                failed += 1
                continue
            
            file_hash = hashlib.md5(content).hexdigest()
            
            if file_hash in existing_hashes:
                continue
            
            ext = '.jpg'
            if url.lower().endswith('.png'):
                ext = '.png'
            elif url.lower().endswith('.webp'):
                ext = '.webp'
            
            filepath = img_dir / f'{len(list(img_dir.glob("*")))}{ext}'
            with open(filepath, 'wb') as f:
                f.write(content)
            
            existing_hashes.add(file_hash)
            downloaded += 1
            
            if downloaded % 5 == 0:
                print(f"   已下载 {downloaded}/{needed}...")
        except Exception as e:
            failed += 1
            continue
    
    final_count = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png'))) + len(list(img_dir.glob('*.webp')))
    print(f"   ✅ 新增 {downloaded} 张, 失败 {failed} 个, 总计: {final_count} 张")
    
    if final_count >= target_count:
        print(f"   🎉 {role_name} 已达标!")
    
    return downloaded

def main():
    print("=" * 70)
    print("🚀 下载图片不足50张的角色")
    print("=" * 70)
    
    roles_needing_more = []
    for pinyin, name in PINYIN_MAPPING.items():
        count = get_current_count(pinyin)
        if count < TARGET_COUNT:
            roles_needing_more.append({
                'name': name,
                'pinyin': pinyin,
                'current_count': count,
                'needed': TARGET_COUNT - count
            })
    
    roles_needing_more.sort(key=lambda x: x['needed'], reverse=True)
    
    print(f"\n找到 {len(roles_needing_more)} 个角色需要补充图片")
    print("\n需要下载的角色（按需求排序）:")
    for role in roles_needing_more:
        print(f"  {role['name']}: 当前 {role['current_count']} 张, 需要 {role['needed']} 张")
    
    total_downloaded = 0
    
    for role in roles_needing_more:
        downloaded = download_role_images(role['name'], role['pinyin'])
        total_downloaded += downloaded
    
    print("\n" + "=" * 70)
    print(f"🎉 下载完成！共新增 {total_downloaded} 张图片")
    print("=" * 70)

if __name__ == '__main__':
    main()
