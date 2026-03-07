#!/usr/bin/env python3
"""
Pixiv数据采集器
使用Pixiv API采集二次元角色图片
"""
import os
import argparse
import random
from time import sleep
from pixivpy3 import *

def download_image(url, save_path, headers=None):
    """下载图片"""
    try:
        default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.pixiv.net/",
            "Accept": "image/*"
        }
        
        if headers:
            default_headers.update(headers)
        
        response = requests.get(url, headers=default_headers, timeout=20)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"下载图片失败: {e}")
        return False

def collect_from_pixiv_api(tags, limit, output_dir, refresh_token):
    """使用pixivpy API从Pixiv采集图片"""
    print(f"使用Pixiv API搜索标签: {tags}")
    
    if not refresh_token:
        print("未提供refresh_token，跳过Pixiv API采集")
        return 0
        
    try:
        api = AppPixivAPI()
        api.auth(refresh_token=refresh_token)
        print("Pixiv API登录成功")
        
        json_result = api.search_illust(tags, search_target='exact_tag_for_title_and_caption')
        
        downloaded = 0
        illust_count = 0
        
        for illust in json_result.illusts:
            if illust_count >= limit:
                break
            
            if illust.x_restrict > 0:
                print(f"跳过受限内容: {illust.title}")
                continue

            if illust.page_count == 1:
                image_url = illust.meta_single_page.get('original_image_url', illust.image_urls.large)
                print(f"准备下载: {illust.title} - {image_url}")
                
                file_ext = os.path.splitext(image_url)[1]
                save_path = os.path.join(output_dir, f"pixiv_{illust.id}_p0{file_ext}")
                
                if download_image(image_url, save_path):
                    downloaded += 1
                    illust_count += 1
                    sleep(random.uniform(1, 3))
            else:
                for page in illust.meta_pages:
                    if illust_count >= limit:
                        break
                    
                    image_url = page.image_urls.original
                    print(f"准备下载: {illust.title} (p{page.image_urls.original.split('_p')[-1].split('.')[0]}) - {image_url}")
                    
                    file_ext = os.path.splitext(image_url)[1]
                    page_num = page.image_urls.original.split('_p')[-1].split('.')[0]
                    save_path = os.path.join(output_dir, f"pixiv_{illust.id}_p{page_num}{file_ext}")

                    if download_image(image_url, save_path):
                        downloaded += 1
                        illust_count += 1
                        sleep(random.uniform(1, 3))
        
        print(f"Pixiv API下载完成，成功下载 {downloaded} 张图片")
        return downloaded

    except Exception as e:
        print(f"Pixiv API采集失败: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Pixiv数据采集器')
    parser.add_argument('--tag', required=True, help='要采集的标签')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--limit', type=int, default=20, help='采集图片数量')
    parser.add_argument('--refresh_token', help='Pixiv API的refresh_token')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    collect_from_pixiv_api(args.tag, args.limit, args.output_dir, args.refresh_token)

if __name__ == '__main__':
    main()
