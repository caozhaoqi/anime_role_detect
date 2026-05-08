#!/usr/bin/env python3
import requests
import hashlib
from pathlib import Path

def download_role_images(role_name, role_pinyin, target_count=50):
    """为单个角色下载图片"""
    img_dir = Path('data/organized_images') / role_pinyin
    img_dir.mkdir(parents=True, exist_ok=True)
    
    url_file = Path(f'spider_image_system/data/img_url/{role_pinyin}_img.txt')
    if not url_file.exists():
        print(f"❌ {role_name}: 未找到URL文件")
        return 0
    
    with open(url_file) as f:
        urls = [l.strip() for l in f if l.strip()]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.pixiv.net/'
    })
    
    # 获取已存在的图片hash
    existing_hashes = set()
    for img_path in img_dir.glob('*'):
        if img_path.is_file():
            try:
                with open(img_path, 'rb') as f:
                    existing_hashes.add(hashlib.md5(f.read()).hexdigest())
            except:
                pass
    
    current_count = len(existing_hashes)
    print(f"\n📁 {role_name}: {len(urls)} 个URL, 当前 {current_count} 张")
    
    if current_count >= target_count:
        print(f"   ⏭️ 已达标，无需补充")
        return 0
    
    needed = target_count - current_count
    downloaded = 0
    
    for url in urls:
        if downloaded >= needed:
            break
        
        try:
            resp = session.get(url, timeout=15)
            content = resp.content
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
            
            if downloaded % 10 == 0:
                print(f"   已下载 {downloaded}/{needed}...")
        except Exception as e:
            continue
    
    final_count = len(list(img_dir.glob('*')))
    print(f"   ✅ 新增 {downloaded} 张, 总计: {final_count} 张")
    
    if final_count >= target_count:
        print(f"   🎉 {role_name} 已达标!")
    
    return downloaded

def main():
    print("=" * 70)
    print("🚀 下载剩余角色图片")
    print("=" * 70)
    
    roles = [
        ('月千夜', 'yue4qian1ye4'),  # 47→50 (有51个URL)
        ('爱丽儿', 'ai4li4er3'),     # 41→50 (有41个URL，差9个)
        ('小闪', 'xiao3shan3'),      # 29→50 (有29个URL，差21个)
        ('釉壶', 'you4hu2'),         # 29→50 (有6个URL，差24个)
        ('克萝萝', 'ke4luo2luo2'),   # 20→50 (有20个URL，差30个)
        ('芙丽希娅', 'fu2li4xi1ya4'), # 11→50 (有11个URL，差39个)
    ]
    
    total_downloaded = 0
    
    for role_name, role_pinyin in roles:
        downloaded = download_role_images(role_name, role_pinyin)
        total_downloaded += downloaded
    
    print("\n" + "=" * 70)
    print(f"🎉 下载完成！共新增 {total_downloaded} 张图片")
    print("=" * 70)

if __name__ == '__main__':
    main()
