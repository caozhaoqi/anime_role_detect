#!/usr/bin/env python3
"""
从Danbooru、Pixiv、Bing等网站采集测试数据脚本
"""
import os
import requests
import argparse
import json
import random
import base64
import re
from time import sleep
from pixivpy3 import *

def download_image(url, save_path, headers=None):
    """下载图片"""
    try:
        # 设置默认请求头
        default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.pixiv.net/",  # Pixiv需要Referer
            "Accept": "image/*"
        }
        
        # 合并传入的headers
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
        # 初始化API
        api = AppPixivAPI()
        
        # 登录
        # 建议将refresh_token存储在环境变量中，而不是直接写在代码里
        # export PIXIV_REFRESH_TOKEN="your_token"
        # refresh_token = os.getenv("PIXIV_REFRESH_TOKEN")
        api.auth(refresh_token=refresh_token)
        print("Pixiv API登录成功")
        
        # 搜索插画
        json_result = api.search_illust(tags, search_target='exact_tag_for_title_and_caption')
        
        downloaded = 0
        illust_count = 0
        
        for illust in json_result.illusts:
            if illust_count >= limit:
                break
            
            # 过滤R-18内容
            if illust.x_restrict > 0:
                print(f"跳过受限内容: {illust.title}")
                continue

            # 下载单张或多张图片
            if illust.page_count == 1:
                image_url = illust.meta_single_page.get('original_image_url', illust.image_urls.large)
                print(f"准备下载: {illust.title} - {image_url}")
                
                file_ext = os.path.splitext(image_url)[1]
                save_path = os.path.join(output_dir, f"pixiv_{illust.id}_p0{file_ext}")
                
                if download_image(image_url, save_path):
                    downloaded += 1
                    illust_count += 1
                    sleep(random.uniform(1, 3)) # 随机延时
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
                        sleep(random.uniform(1, 3)) # 随机延时
        
        print(f"Pixiv API下载完成，成功下载 {downloaded} 张图片")
        return downloaded

    except Exception as e:
        print(f"Pixiv API采集失败: {e}")
        return 0

def collect_from_danbooru(tags, limit, output_dir, api_key=None, user=None):
    """从Danbooru搜索并下载二次元图片"""
    print(f"从Danbooru搜索标签: {tags}")
    
    try:
        import urllib.parse
        encoded_tags = urllib.parse.quote(tags)
        api_url = f"https://danbooru.donmai.us/posts.json?limit={limit*2}&tags={encoded_tags}" # 多获取一些以备过滤
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json"
        }
        if api_key and user:
            headers["Authorization"] = f"Basic {base64.b64encode(f'{user}:{api_key}'.encode()).decode()}"
        
        response = requests.get(api_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        posts = response.json()
        
        image_urls = []
        for post in posts:
            # 过滤不合适的tag
            if 'file_url' in post and 'rating' in post and post['rating'] == 's': # 只选择安全内容
                image_urls.append(post["file_url"])
        
        print(f"找到 {len(image_urls)} 个安全结果")
        
        downloaded = 0
        for i, image_url in enumerate(image_urls[:limit]):
            file_ext = os.path.splitext(image_url)[1]
            if not file_ext:
                file_ext = ".jpg"
            
            clean_tags = tags.replace(" ", "_").replace(":", "")
            save_path = os.path.join(output_dir, f"danbooru_{clean_tags}_{i+1}{file_ext}")
            
            print(f"下载 {image_url} 到 {save_path}")
            if download_image(image_url, save_path):
                downloaded += 1
                sleep(1)
        
        return downloaded
    except Exception as e:
        print(f"Danbooru采集失败: {e}")
        return 0

def collect_from_safebooru(tags, limit, output_dir):
    """从Safebooru搜索并下载二次元图片（无需API Key，内容相对安全）"""
    print(f"从Safebooru搜索标签: {tags}")
    
    try:
        import urllib.parse
        encoded_tags = urllib.parse.quote(tags)
        # Safebooru API URL
        api_url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&limit={limit*2}&tags={encoded_tags}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(api_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        posts = response.json()
        
        image_urls = []
        for post in posts:
            if 'file_url' in post:
                # Safebooru的file_url有时是相对路径
                url = post["file_url"]
                if not url.startswith("http"):
                    url = "https://safebooru.org/images/" + post["directory"] + "/" + post["image"]
                image_urls.append(url)
        
        print(f"找到 {len(image_urls)} 个结果")
        
        downloaded = 0
        for i, image_url in enumerate(image_urls[:limit]):
            file_ext = os.path.splitext(image_url)[1]
            if not file_ext:
                file_ext = ".jpg"
            
            clean_tags = tags.replace(" ", "_").replace(":", "")
            save_path = os.path.join(output_dir, f"safebooru_{clean_tags}_{i+1}{file_ext}")
            
            print(f"下载 {image_url} 到 {save_path}")
            if download_image(image_url, save_path):
                downloaded += 1
                sleep(1)
        
        return downloaded
    except Exception as e:
        print(f"Safebooru采集失败: {e}")
        return 0

def collect_from_bing(query, limit, output_dir):
    """从Bing搜索并下载图片（无需API Key，模拟浏览器）"""
    print(f"从Bing搜索: {query}")
    
    try:
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        # Bing图片搜索URL
        url = f"https://www.bing.com/images/async?q={encoded_query}&first=0&count={limit*2}&adlt=off&qft="
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # 使用正则提取图片URL (murl)
        # Bing返回的HTML中包含类似 murl&quot;:&quot;http...&quot; 的结构
        image_urls = re.findall(r'murl&quot;:&quot;(http[^&]+)&quot;', response.text)
        
        print(f"找到 {len(image_urls)} 个结果")
        
        downloaded = 0
        for i, image_url in enumerate(image_urls[:limit]):
            # 尝试推断文件扩展名
            file_ext = ".jpg"
            if ".png" in image_url.lower():
                file_ext = ".png"
            elif ".jpeg" in image_url.lower():
                file_ext = ".jpeg"
                
            clean_query = query.replace(" ", "_").replace(":", "")
            save_path = os.path.join(output_dir, f"bing_{clean_query}_{i+1}{file_ext}")
            
            print(f"下载 {image_url} 到 {save_path}")
            if download_image(image_url, save_path):
                downloaded += 1
                sleep(1)
        
        return downloaded
    except Exception as e:
        print(f"Bing采集失败: {e}")
        return 0

def collect_from_pixiv(tags, limit, output_dir, refresh_token=None):
    """从Pixiv采集二次元图片，失败时尝试其他来源"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"开始从Pixiv采集，搜索标签: {tags}")
    
    # 优先使用API
    downloaded = collect_from_pixiv_api(tags, limit, output_dir, refresh_token)
    
    # 如果API失败，回退到Danbooru
    if downloaded == 0:
        print("Pixiv API采集失败或未配置，尝试Danbooru...")
        downloaded = collect_from_danbooru(tags, limit, output_dir)

    # 如果Danbooru失败，尝试Safebooru
    if downloaded == 0:
        print("Danbooru采集失败，尝试Safebooru...")
        downloaded = collect_from_safebooru(tags, limit, output_dir)

    # 如果Safebooru失败，尝试Bing
    if downloaded == 0:
        print("Safebooru采集失败，尝试Bing...")
        # Bing搜索时加上 "anime" 或 "genshin" 等关键词以提高相关性
        bing_query = f"{tags} anime character"
        downloaded = collect_from_bing(bing_query, limit, output_dir)

    # 如果仍然没有下载到图片，使用本地样本作为备选
    if downloaded == 0:
        print("所有主要源都采集失败，使用本地样本作为备选")
        downloaded = collect_from_local_sample(output_dir, limit)
    
    print(f"采集完成，共成功下载 {downloaded} 张图片")
    return downloaded

def collect_from_local_sample(output_dir, count=5):
    """从本地样本获取图片"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("正在下载/生成本地样本图片...")
    downloaded = 0
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        for i in range(count):
            img = Image.new('RGB', (800, 600), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            d = ImageDraw.Draw(img)
            text = f"Sample Image {i+1}"
            try:
                font = ImageFont.truetype("Arial.ttf", 36)
            except IOError:
                font = ImageFont.load_default()
            
            text_bbox = d.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (800 - text_width) // 2
            text_y = (600 - text_height) // 2
            
            d.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
            
            save_path = os.path.join(output_dir, f"sample_{i+1}.jpg")
            img.save(save_path)
            print(f"生成本地样本图片到 {save_path}")
            downloaded += 1
    except Exception as e:
        print(f"生成本地样本失败: {e}")
    
    return downloaded

def collect_single_character_data(character_name, limit, output_dir, refresh_token=None, api_key=None, user=None):
    """采集单角色数据"""
    tags = character_name # 默认使用角色名作为tag
    
    # 针对特定游戏优化tag
    if "鸣潮" in character_name or "wuthering_waves" in character_name.lower():
        character_tag = character_name.replace("鸣潮_", "").replace("鸣潮 ", "")
        tags = f"wuthering_waves {character_tag}"
    elif "原神" in character_name or "genshin" in character_name.lower():
        character_tag = character_name.replace("原神_", "").replace("原神 ", "")
        tags = f"genshin_impact {character_tag}"
    
    return collect_from_pixiv(tags, limit, output_dir, refresh_token)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从Pixiv等网站采集测试数据脚本")
    parser.add_argument("--character", required=True, help="角色名称 (例如: 'rem_(re:zero)', 'genshin_impact klee')")
    parser.add_argument("--limit", type=int, default=5, help="采集图片数量")
    parser.add_argument("--output_dir", help="输出目录 (默认为 'tests/test_images/single_character/<character_name>')")
    
    # API认证参数
    parser.add_argument("--refresh_token", help="Pixiv API的refresh_token。强烈建议使用环境变量 PIXIV_REFRESH_TOKEN")
    parser.add_argument("--api_key", help="Danbooru API密钥 (可选)")
    parser.add_argument("--user", help="Danbooru用户名 (可选)")
    
    args = parser.parse_args()
    
    # 优先从环境变量获取refresh_token
    refresh_token = args.refresh_token or os.getenv("PIXIV_REFRESH_TOKEN")
    
    # 确定输出目录
    output_dir = args.output_dir or os.path.join("tests/test_images/single_character", args.character.replace(" ", "_").replace(":", "_"))
    
    # 执行采集
    collect_single_character_data(args.character, args.limit, output_dir, refresh_token, args.api_key, args.user)

if __name__ == "__main__":
    main()
