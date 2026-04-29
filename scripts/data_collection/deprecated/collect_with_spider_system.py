#!/usr/bin/env python3
"""
使用 spider_image_system 进行二次元图片采集
"""
import os
import sys
import time
import json
import requests
import hashlib
import random
import re
from PIL import Image
from io import BytesIO
from pathlib import Path

class SpiderSystemCollector:
    """使用爬虫系统进行二次元图片采集"""
    
    def __init__(self, output_dir='./anime_images'):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        })
        os.makedirs(output_dir, exist_ok=True)
    
    def search_pixiv_images(self, keyword, count=30):
        """搜索Pixiv图片（使用镜像站点）"""
        urls_found = []
        
        # 尝试多个镜像站点
        mirror_sites = [
            "https://pixiv.srpr.cc",
            "https://pixiv.888718.xyz",
        ]
        
        for site in mirror_sites:
            try:
                search_url = f"{site}/search.php?s={requests.utils.quote(keyword)}"
                response = self.session.get(search_url, timeout=15)
                
                if response.status_code == 200:
                    img_pattern = re.compile(r'https?://[^\s"]+\.(jpg|png)', re.IGNORECASE)
                    matches = img_pattern.findall(response.text)
                    
                    for match in matches[:count]:
                        if match not in urls_found:
                            urls_found.append(match)
            except Exception as e:
                print(f"搜索失败 {site}: {e}")
            time.sleep(2)
        
        return urls_found
    
    def try_direct_download(self, keyword, count=30):
        """尝试直接下载图片"""
        downloaded = []
        target_url = "https://pximg.lolicon.run"
        
        for _ in range(count * 3):
            if len(downloaded) >= count:
                break
                
            try:
                img_id = random.randint(1000000, 9999999)
                img_url = f"{target_url}/img-original/img/2024/01/01/{img_id}_p0.jpg"
                
                response = self.session.get(img_url, timeout=10)
                if response.status_code == 200 and len(response.content) > 1000:
                    downloaded.append(response.content)
                    
            except Exception as e:
                pass
            time.sleep(0.3)
        
        return downloaded
    
    def search_and_download(self, keyword, count=30):
        """搜索并下载图片"""
        images = []
        
        print(f"尝试直接下载 {keyword} 的图片...")
        direct_imgs = self.try_direct_download(keyword, count)
        images.extend(direct_imgs)
        print(f"直接下载成功: {len(direct_imgs)} 张")
        
        if len(images) >= count:
            return images[:count]
        
        print(f"尝试搜索 {keyword} 的图片...")
        urls = self.search_pixiv_images(keyword, count)
        
        for url in urls[:count - len(images)]:
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200 and len(response.content) > 1000:
                    images.append(response.content)
            except:
                pass
            time.sleep(0.3)
        
        return images
    
    def is_valid_image(self, content):
        """检查图片是否有效"""
        try:
            img = Image.open(BytesIO(content))
            width, height = img.size
            if width >= 512 and height >= 512:
                return True, width, height
            return False, width, height
        except:
            return False, 0, 0
    
    def collect_for_role(self, role_name, target_count=30):
        """为角色采集图片"""
        role_dir = os.path.join(self.output_dir, role_name)
        os.makedirs(role_dir, exist_ok=True)
        
        existing = [f for f in os.listdir(role_dir) if f.endswith('.jpg')]
        existing_count = len(existing)
        
        if existing_count >= target_count:
            print(f"角色 {role_name} 已有 {existing_count} 张图片，跳过")
            return {'role': role_name, 'success': 0, 'failed': 0, 'existing': existing_count}
        
        need_count = target_count - existing_count
        print(f"角色 {role_name} 需要采集 {need_count} 张图片")
        
        keywords = role_name.split()
        all_images = []
        
        for keyword in keywords[:3]:
            imgs = self.search_and_download(keyword, need_count)
            all_images.extend(imgs)
            time.sleep(1)
        
        success_count = 0
        failed_count = 0
        seen_hashes = set()
        
        for content in all_images:
            if success_count >= need_count:
                break
            
            is_valid, width, height = self.is_valid_image(content)
            if not is_valid:
                failed_count += 1
                continue
            
            file_hash = hashlib.md5(content).hexdigest()
            if file_hash in seen_hashes:
                continue
            seen_hashes.add(file_hash)
            
            save_path = os.path.join(role_dir, f"{file_hash}.jpg")
            
            if os.path.exists(save_path):
                continue
            
            try:
                with open(save_path, 'wb') as f:
                    f.write(content)
                success_count += 1
                print(f"  成功: {width}x{height}")
            except Exception as e:
                failed_count += 1
        
        print(f"角色 {role_name} 采集完成: 成功 {success_count}, 失败 {failed_count}")
        return {
            'role': role_name,
            'success': success_count,
            'failed': failed_count,
            'existing': existing_count
        }
    
    def collect_all(self, role_list, target_count=30):
        """采集所有角色"""
        print(f"开始采集，共 {len(role_list)} 个角色")
        
        results = []
        for role in role_list:
            result = self.collect_for_role(role, target_count)
            results.append(result)
        
        total_success = sum(r['success'] for r in results)
        total_failed = sum(r['failed'] for r in results)
        
        print(f"\n=== 采集完成 ===")
        print(f"成功下载: {total_success}")
        print(f"下载失败: {total_failed}")
        
        return {
            'total_roles': len(results),
            'total_success': total_success,
            'total_failed': total_failed,
            'details': results
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='使用爬虫系统采集二次元图片')
    parser.add_argument('--role_file', type=str, required=True, help='角色列表文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--target_count', type=int, default=30, help='目标数量')
    
    args = parser.parse_args()
    
    with open(args.role_file, 'r', encoding='utf-8') as f:
        role_list = [line.strip() for line in f if line.strip()]
    
    print(f"读取到 {len(role_list)} 个角色")
    
    collector = SpiderSystemCollector(output_dir=args.output_dir)
    result = collector.collect_all(role_list, args.target_count)
    
    with open('spider_system_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("\n采集完成！")

if __name__ == "__main__":
    main()
