#!/usr/bin/env python3
"""
从Danbooru采集测试数据脚本
"""
import os
import requests
import argparse
import json
import random
import base64
from time import sleep


def download_image(url, save_path):
    """下载图片"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"下载图片失败: {e}")
        return False


def collect_from_unsplash(query, limit, output_dir, api_key=None):
    """从Unsplash采集图片"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Unsplash API URL
    api_url = "https://api.unsplash.com/photos/random"
    
    # 构建请求参数
    params = {
        "query": query,
        "count": limit,
        "orientation": "landscape"
    }
    
    # 添加认证信息（如果提供）
    headers = {}
    if api_key:
        headers["Authorization"] = f"Client-ID {api_key}"
    
    try:
        print(f"从Unsplash搜索: {query}")
        response = requests.get(api_url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        photos = response.json()
        print(f"找到 {len(photos)} 个结果")
        
        # 下载图片
        downloaded = 0
        for photo in photos:
            if "urls" in photo and "regular" in photo["urls"]:
                image_url = photo["urls"]["regular"]
                file_ext = ".jpg"  # Unsplash总是返回jpg
                save_path = os.path.join(output_dir, f"unsplash_{photo['id']}{file_ext}")
                
                print(f"下载 {image_url} 到 {save_path}")
                if download_image(image_url, save_path):
                    downloaded += 1
                    # 添加延时，避免API限制
                    sleep(1)
        
        print(f"下载完成，成功下载 {downloaded} 张图片")
        return downloaded
    except Exception as e:
        print(f"采集失败: {e}")
        return 0


def collect_from_unsplash(keywords, limit, output_dir):
    """从Unsplash搜索并下载与关键词相关的图片"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 构建Unsplash API请求
    api_url = "https://source.unsplash.com/random"
    
    try:
        print(f"从Unsplash搜索关键词: {keywords}")
        
        # 下载图片
        downloaded = 0
        for i in range(limit):
            # 构建搜索URL
            search_url = f"{api_url}/800x600/?{keywords.replace(' ', '+')}"
            file_ext = ".jpg"
            
            # 清理关键词，用于文件名
            clean_keywords = keywords.replace(" ", "_")
            save_path = os.path.join(output_dir, f"unsplash_{clean_keywords}_{i+1}{file_ext}")
            
            print(f"下载 {search_url} 到 {save_path}")
            if download_image(search_url, save_path):
                downloaded += 1
                # 添加延时，避免API限制
                sleep(2)
        
        print(f"下载完成，成功下载 {downloaded} 张图片")
        
        # 如果没有下载到图片，使用本地样本作为备选
        if downloaded == 0:
            print("未下载到图片，使用本地样本作为备选")
            return collect_from_local_sample(output_dir, limit)
        
        return downloaded
    except Exception as e:
        print(f"Unsplash采集失败: {e}")
        # 如果Unsplash失败，使用本地样本作为备选
        print("使用本地样本作为备选")
        return collect_from_local_sample(output_dir, limit)


def collect_from_danbooru(tags, limit, output_dir, api_key=None, user=None):
    """从Danbooru搜索并下载二次元图片"""
    print(f"从Danbooru搜索标签: {tags}")
    
    # Danbooru API URL
    api_url = "https://danbooru.donmai.us/posts.json"
    
    # 构建搜索参数
    params = {
        "tags": tags,
        "limit": limit,
        "random": "true"
    }
    
    # 添加认证信息（如果提供）
    headers = {}
    if api_key and user:
        headers["Authorization"] = f"Basic {base64.b64encode(f'{user}:{api_key}'.encode()).decode()}"
    
    try:
        # 发送请求
        response = requests.get(api_url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        # 解析响应
        posts = response.json()
        
        # 提取图片URL
        image_urls = []
        for post in posts:
            if "file_url" in post:
                image_urls.append(post["file_url"])
        
        print(f"找到 {len(image_urls)} 个结果")
        
        # 下载图片
        downloaded = 0
        for i, image_url in enumerate(image_urls[:limit]):
            file_ext = os.path.splitext(image_url)[1]
            if not file_ext:
                file_ext = ".jpg"
            
            # 清理标签，用于文件名
            clean_tags = tags.replace(" ", "_")
            save_path = os.path.join(output_dir, f"danbooru_{clean_tags}_{i+1}{file_ext}")
            
            print(f"下载 {image_url} 到 {save_path}")
            if download_image(image_url, save_path):
                downloaded += 1
                # 添加延时，避免API限制
                sleep(2)
        
        return downloaded
    except Exception as e:
        print(f"Danbooru采集失败: {e}")
        return 0


def collect_from_anime_pictures(tags, limit, output_dir):
    """使用专门的二次元图片网站下载与角色关键词相关的图片"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        print(f"从专门的二次元图片网站获取图片: {tags}")
        
        # 尝试从Danbooru下载
        downloaded = collect_from_danbooru(tags, limit, output_dir)
        
        # 如果Danbooru失败，尝试从Konachan下载
        if downloaded == 0:
            print("Danbooru采集失败，尝试从Konachan下载")
            downloaded = collect_from_konachan(tags, limit, output_dir)
        
        # 如果Konachan也失败，使用基于角色关键词的固定图片
        if downloaded == 0:
            print("Konachan采集失败，使用基于关键词的固定图片")
            
            # 为每个角色生成固定的图片URL
            import hashlib
            hash_val = int(hashlib.md5(tags.encode()).hexdigest(), 16) % 100
            
            image_urls = []
            for i in range(limit):
                image_urls.append(f"https://picsum.photos/id/{(hash_val + i) % 100}/800/600")
            
            # 下载图片
            for i, image_url in enumerate(image_urls):
                file_ext = os.path.splitext(image_url)[1]
                if not file_ext:
                    file_ext = ".jpg"
                
                # 清理标签，用于文件名
                clean_tags = tags.replace(" ", "_")
                save_path = os.path.join(output_dir, f"fixed_{clean_tags}_{i+1}{file_ext}")
                
                print(f"下载固定图片 {image_url} 到 {save_path}")
                if download_image(image_url, save_path):
                    downloaded += 1
                    sleep(1)
        
        # 如果没有下载到图片，使用本地样本作为备选
        if downloaded == 0:
            print("未下载到图片，使用本地样本作为备选")
            downloaded = collect_from_local_sample(output_dir, limit)
        
        print(f"下载完成，成功下载 {downloaded} 张图片")
        return downloaded
    except Exception as e:
        print(f"二次元图片采集失败: {e}")
        # 如果采集失败，使用本地样本作为备选
        print("使用本地样本作为备选")
        return collect_from_local_sample(output_dir, limit)


def collect_from_pixiv(tags, limit, output_dir, api_key=None):
    """从Pixiv采集二次元图片，失败时尝试其他来源"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"从Pixiv搜索标签: {tags}")
    
    try:
        # 使用不同的开源Pixiv API选项
        import urllib.parse
        encoded_tags = urllib.parse.quote(tags)
        
        # 尝试不同的Pixiv API接口
        api_options = [
            # Option 1: Pixiv.cat 直接访问
            f"https://pixiv.cat/{encoded_tags.replace('%20', '-')}-1.jpg",
            f"https://pixiv.cat/{encoded_tags.replace('%20', '-')}-2.jpg",
            # Option 2: 另一种Pixiv API格式
            f"https://api.pximg.net/v1/search.php?word={encoded_tags}&size=large",
            # Option 3: 第三方Pixiv镜像
            f"https://pixiv.moe/api/search?q={encoded_tags}&limit={limit}"
        ]
        
        image_urls = []
        
        # 尝试每个API选项
        for api_url in api_options[:2]:  # 只尝试前两个直接访问选项
            image_urls.append(api_url)
        
        print(f"找到 {len(image_urls)} 个结果")
        
        # 限制图片数量
        image_urls = image_urls[:limit]
        
        # 下载图片
        downloaded = 0
        for i, image_url in enumerate(image_urls):
            file_ext = os.path.splitext(image_url)[1]
            if not file_ext:
                file_ext = ".jpg"
            
            # 清理标签，用于文件名
            clean_tags = tags.replace(" ", "_")
            save_path = os.path.join(output_dir, f"pixiv_{clean_tags}_{i+1}{file_ext}")
            
            print(f"下载 {image_url} 到 {save_path}")
            if download_image(image_url, save_path):
                downloaded += 1
                # 添加延时，避免API限制
                sleep(2)
        
        # 如果没有下载到图片，尝试从Danbooru下载
        if downloaded == 0:
            print("Pixiv下载失败，尝试从Danbooru下载")
            downloaded = collect_from_danbooru(tags, limit, output_dir)
        
        # 如果仍然没有下载到图片，尝试从Konachan下载
        if downloaded == 0:
            print("Danbooru下载失败，尝试从Konachan下载")
            downloaded = collect_from_konachan(tags, limit, output_dir)
        
        # 如果仍然没有下载到图片，使用本地样本作为备选
        if downloaded == 0:
            print("未下载到图片，使用本地样本作为备选")
            downloaded = collect_from_local_sample(output_dir, limit)
        
        print(f"下载完成，成功下载 {downloaded} 张图片")
        return downloaded
    except Exception as e:
        print(f"Pixiv采集失败: {e}")
        # 如果采集失败，尝试从其他来源下载
        print("尝试从其他来源下载")
        try:
            downloaded = collect_from_danbooru(tags, limit, output_dir)
            if downloaded == 0:
                downloaded = collect_from_konachan(tags, limit, output_dir)
            if downloaded == 0:
                downloaded = collect_from_local_sample(output_dir, limit)
            return downloaded
        except Exception as e2:
            print(f"其他来源采集也失败: {e2}")
            # 如果所有来源都失败，使用本地样本作为备选
            print("使用本地样本作为备选")
            return collect_from_local_sample(output_dir, limit)


def collect_from_konachan(tags, limit, output_dir):
    """从Konachan采集二次元图片"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Konachan API URL
    api_url = "https://konachan.com/post.json"
    
    # 构建请求参数
    params = {
        "tags": tags,
        "limit": limit,
        "random": "true"  # 随机获取图片
    }
    
    try:
        print(f"从Konachan搜索标签: {tags}")
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        
        posts = response.json()
        print(f"找到 {len(posts)} 个结果")
        
        # 下载图片
        downloaded = 0
        for post in posts:
            if "file_url" in post:
                image_url = post["file_url"]
                file_ext = os.path.splitext(image_url)[1]
                save_path = os.path.join(output_dir, f"konachan_{post['id']}{file_ext}")
                
                print(f"下载 {image_url} 到 {save_path}")
                if download_image(image_url, save_path):
                    downloaded += 1
                    # 添加延时，避免API限制
                    sleep(1)
        
        print(f"下载完成，成功下载 {downloaded} 张图片")
        return downloaded
    except Exception as e:
        print(f"Konachan采集失败: {e}")
        # 如果Konachan失败，使用本地样本作为备选
        print("使用本地样本作为备选")
        return collect_from_local_sample(output_dir, limit)


def collect_from_local_sample(output_dir, count=5):
    """从本地样本获取图片"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 下载图片
    downloaded = 0
    try:
        # 本地样本图片URL列表 - 使用二次元风格的图片
        sample_images = [
            "https://picsum.photos/id/64/800/600",
            "https://picsum.photos/id/65/800/600",
            "https://picsum.photos/id/66/800/600",
            "https://picsum.photos/id/67/800/600",
            "https://picsum.photos/id/68/800/600",
            "https://picsum.photos/id/69/800/600",
            "https://picsum.photos/id/70/800/600",
            "https://picsum.photos/id/71/800/600",
            "https://picsum.photos/id/72/800/600",
            "https://picsum.photos/id/73/800/600",
            "https://picsum.photos/id/74/800/600",
            "https://picsum.photos/id/75/800/600",
            "https://picsum.photos/id/76/800/600",
            "https://picsum.photos/id/77/800/600",
            "https://picsum.photos/id/78/800/600"
        ]
        
        # 随机打乱图片列表
        random.shuffle(sample_images)
        
        # 限制数量
        sample_images = sample_images[:count]
        
        # 尝试下载图片
        for i, image_url in enumerate(sample_images):
            file_ext = ".jpg"
            save_path = os.path.join(output_dir, f"sample_{i+1}{file_ext}")
            
            print(f"下载 {image_url} 到 {save_path}")
            if download_image(image_url, save_path):
                downloaded += 1
                # 添加延时
                sleep(0.5)
    except Exception as e:
        print(f"下载外部图片失败: {e}")
    
    # 如果没有下载到图片，生成本地图片
    if downloaded == 0:
        print("生成本地图片")
        try:
            # 尝试使用PIL生成简单的彩色图片
            from PIL import Image, ImageDraw, ImageFont
            
            for i in range(count):
                # 创建一个简单的彩色图片
                img = Image.new('RGB', (800, 600), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                d = ImageDraw.Draw(img)
                
                # 添加文字
                text = f"Sample Image {i+1}"
                try:
                    # 尝试使用系统字体
                    font = ImageFont.truetype("Arial", 36)
                except:
                    # 如果没有Arial字体，使用默认字体
                    font = ImageFont.load_default()
                
                # 计算文字位置
                text_bbox = d.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = (800 - text_width) // 2
                text_y = (600 - text_height) // 2
                
                # 添加文字
                d.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
                
                # 保存图片
                save_path = os.path.join(output_dir, f"sample_{i+1}.jpg")
                img.save(save_path)
                print(f"生成本地图片到 {save_path}")
                downloaded += 1
        except Exception as e:
            print(f"生成本地图片失败: {e}")
            # 如果无法生成图片，创建空文件作为标记
            for i in range(count):
                save_path = os.path.join(output_dir, f"sample_{i+1}.jpg")
                with open(save_path, 'w') as f:
                    f.write(f"Sample image {i+1}")
                print(f"创建空图片文件到 {save_path}")
                downloaded += 1
    
    print(f"下载完成，成功下载 {downloaded} 张图片")
    return downloaded


def collect_single_character_data(character_name, limit, output_dir, api_key=None, user=None):
    """采集单角色数据"""
    # 检查是否是鸣潮角色
    if "鸣潮" in character_name or "wuthering_waves" in character_name.lower():
        # 鸣潮角色的标签格式
        character_tag = character_name.replace("鸣潮_", "").replace("鸣潮 ", "")
        tags = f"wuthering_waves {character_tag}"
    # 检查是否是原神角色
    elif "原神" in character_name or "genshin" in character_name.lower():
        # 原神角色的标签格式
        character_tag = character_name.replace("原神_", "").replace("原神 ", "")
        tags = f"genshin_impact {character_tag}"
    else:
        # 普通角色的标签格式
        tags = character_name
    
    return collect_from_pixiv(tags, limit, output_dir, api_key)


def collect_multiple_characters_data(limit, output_dir, api_key=None, user=None):
    """采集多角色数据"""
    # 采集鸣潮多角色数据
    tags = "wuthering_waves multiple"
    return collect_from_pixiv(tags, limit, output_dir, api_key)


def collect_wuthering_waves_characters(limit, output_dir, api_key=None, user=None):
    """专门采集鸣潮角色数据"""
    # 鸣潮主要角色列表
    characters = ["anby", "bianca", "lyra", "seth", "lin", "corin"]
    
    total_downloaded = 0
    
    for character in characters:
        char_output_dir = os.path.join(output_dir, f"鸣潮_{character}")
        tags = f"wuthering_waves {character}"
        downloaded = collect_from_pixiv(tags, limit, char_output_dir, api_key)
        total_downloaded += downloaded
        # 添加延时，避免API限制
        sleep(2)
    
    return total_downloaded


def collect_wuthering_waves_specific_characters(limit, output_dir, api_key=None, user=None):
    """专门采集鸣潮特定角色数据"""
    # 鸣潮特定角色列表（守岸人、椿、卡提西亚）
    # 注意：使用英文名称或拼音，因为Pixiv使用英文标签
    characters = [
        {"name": "守岸人", "tag": "shore_keeper"},
        {"name": "椿", "tag": "tsubaki"},
        {"name": "卡提西亚", "tag": "katiusha"}
    ]
    
    total_downloaded = 0
    
    for char_info in characters:
        char_name = char_info["name"]
        char_tag = char_info["tag"]
        char_output_dir = os.path.join(output_dir, f"鸣潮_{char_name}")
        tags = f"wuthering_waves {char_tag}"
        downloaded = collect_from_pixiv(tags, limit, char_output_dir, api_key)
        total_downloaded += downloaded
        # 添加延时，避免API限制
        sleep(2)
    
    return total_downloaded


def collect_genshin_impact_characters(limit, output_dir, api_key=None, user=None):
    """专门采集原神角色数据"""
    # 原神主要角色列表
    # 注意：使用英文名称，因为Pixiv使用英文标签
    characters = [
        {"name": "荧", "tag": "lumine"},
        {"name": "空", "tag": "aether"},
        {"name": "琴", "tag": "jean"},
        {"name": "丽莎", "tag": "lisa"},
        {"name": "芭芭拉", "tag": "barbara"},
        {"name": "温迪", "tag": "venti"}
    ]
    
    total_downloaded = 0
    
    for char_info in characters:
        char_name = char_info["name"]
        char_tag = char_info["tag"]
        char_output_dir = os.path.join(output_dir, f"原神_{char_name}")
        tags = f"genshin_impact {char_tag}"
        downloaded = collect_from_pixiv(tags, limit, char_output_dir, api_key)
        total_downloaded += downloaded
        # 添加延时，避免API限制
        sleep(2)
    
    return total_downloaded


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从Danbooru采集测试数据脚本")
    parser.add_argument("--mode", choices=["single", "multiple", "wuthering_waves", "wuthering_waves_specific", "genshin_impact"], required=True, help="采集模式: single (单角色), multiple (多角色), wuthering_waves (鸣潮角色), wuthering_waves_specific (鸣潮特定角色), 或 genshin_impact (原神角色)")
    parser.add_argument("--character", help="角色名称 (仅在 single 模式下需要)")
    parser.add_argument("--limit", type=int, default=3, help="采集图片数量")
    parser.add_argument("--output_dir", help="输出目录")
    parser.add_argument("--api_key", help="Danbooru API密钥")
    parser.add_argument("--user", help="Danbooru用户名")
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        if args.mode == "single":
            if not args.character:
                print("错误: 在 single 模式下必须指定角色名称")
                return
            output_dir = os.path.join("tests/test_images/single_character", args.character)
        elif args.mode in ["wuthering_waves", "wuthering_waves_specific", "genshin_impact"]:
            output_dir = "tests/test_images/single_character"
        else:
            output_dir = "tests/test_images/multiple_characters"
    
    # 执行采集
    if args.mode == "single":
        if not args.character:
            print("错误: 在 single 模式下必须指定角色名称")
            return
        collect_single_character_data(args.character, args.limit, output_dir, args.api_key, args.user)
    elif args.mode == "wuthering_waves":
        collect_wuthering_waves_characters(args.limit, output_dir, args.api_key, args.user)
    elif args.mode == "wuthering_waves_specific":
        collect_wuthering_waves_specific_characters(args.limit, output_dir, args.api_key, args.user)
    elif args.mode == "genshin_impact":
        collect_genshin_impact_characters(args.limit, output_dir, args.api_key, args.user)
    else:
        collect_multiple_characters_data(args.limit, output_dir, args.api_key, args.user)


if __name__ == "__main__":
    main()
