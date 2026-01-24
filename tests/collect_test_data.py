#!/usr/bin/env python3
"""
从Danbooru/Safebooru采集测试数据脚本 - 修复版
确保下载图片与角色匹配
"""
import os
import requests
import argparse
import json
import random
import base64
from time import sleep
import urllib.parse

def download_image(url, save_path, headers=None):
    """下载图片"""
    try:
        # 设置默认请求头
        default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": url,
            "Accept": "image/*"
        }
        
        # 合并传入的headers
        if headers:
            default_headers.update(headers)
        
        print(f"正在下载: {url}")
        response = requests.get(url, headers=default_headers, timeout=20)
        response.raise_for_status()
        
        # 检查是否是图片内容
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type and 'octet-stream' not in content_type:
            print(f"跳过非图片内容: {content_type}")
            return False

        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"下载图片失败: {e}")
        return False

def collect_from_safebooru(tags, limit, output_dir):
    """从Safebooru搜索并下载（无需API Key，比较稳定）"""
    print(f"正在从Safebooru搜索标签: {tags}")
    
    # 简单的标签清理
    clean_tags = tags.replace(" ", "_")
    
    # 构建API URL
    # Safebooru API: index.php?page=dapi&s=post&q=index&json=1&tags=...
    base_url = "https://safebooru.org/index.php"
    params = {
        "page": "dapi",
        "s": "post",
        "q": "index",
        "json": "1",
        "limit": limit,
        "tags": tags
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        
        # Safebooru 有时返回空或非JSON
        try:
            posts = response.json()
        except json.JSONDecodeError:
            print("Safebooru返回了非JSON数据，可能是标签搜索无结果。")
            return 0
            
        if not posts:
            print("Safebooru未找到相关图片。")
            return 0
            
        print(f"Safebooru找到 {len(posts)} 个结果")
        
        downloaded = 0
        for i, post in enumerate(posts):
            if downloaded >= limit:
                break
                
            # Safebooru 图片字段通常是 'file_url'，有时只有 'image' 和 'directory'
            if "file_url" in post:
                image_url = post["file_url"]
            elif "image" in post and "directory" in post:
                image_url = f"https://safebooru.org/images/{post['directory']}/{post['image']}"
            else:
                continue

            # 处理扩展名
            file_ext = os.path.splitext(image_url)[1]
            if not file_ext:
                file_ext = ".jpg"
            
            save_path = os.path.join(output_dir, f"safebooru_{clean_tags}_{post['id']}{file_ext}")
            
            if download_image(image_url, save_path):
                downloaded += 1
                sleep(1) # 礼貌延时
        
        return downloaded

    except Exception as e:
        print(f"Safebooru采集异常: {e}")
        return 0

def collect_from_danbooru(tags, limit, output_dir, api_key=None, user=None):
    """从Danbooru搜索并下载"""
    print(f"从Danbooru搜索标签: {tags}")
    
    try:
        encoded_tags = urllib.parse.quote(tags)
        # Danbooru限制非会员只能搜2个标签，这里尽量保证只用关键标签
        api_url = f"https://danbooru.donmai.us/posts.json?limit={limit}&tags={encoded_tags}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json"
        }
        if api_key and user:
            headers["Authorization"] = f"Basic {base64.b64encode(f'{user}:{api_key}'.encode()).decode()}"
        
        response = requests.get(api_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        posts = response.json()
        print(f"Danbooru找到 {len(posts)} 个结果")
        
        downloaded = 0
        for i, post in enumerate(posts):
            if downloaded >= limit:
                break
                
            if "file_url" in post:
                image_url = post["file_url"]
                file_ext = os.path.splitext(image_url)[1]
                if not file_ext: file_ext = ".jpg"
                
                clean_tags = tags.replace(" ", "_")
                # 截断过长的文件名
                clean_tags = clean_tags[:50]
                save_path = os.path.join(output_dir, f"danbooru_{clean_tags}_{post['id']}{file_ext}")
                
                if download_image(image_url, save_path):
                    downloaded += 1
                    sleep(1)
        
        return downloaded
    except Exception as e:
        print(f"Danbooru采集失败: {e}")
        return 0

def collect_from_konachan(tags, limit, output_dir):
    """从Konachan采集"""
    print(f"从Konachan搜索标签: {tags}")
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        params = {
            "tags": tags,
            "limit": limit
        }
        # Konachan API
        api_url = "https://konachan.net/post.json"
        
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        
        posts = response.json()
        print(f"Konachan找到 {len(posts)} 个结果")
        
        downloaded = 0
        for post in posts:
            if downloaded >= limit:
                break
            if "file_url" in post:
                image_url = post["file_url"]
                # 处理以 // 开头的URL
                if image_url.startswith("//"):
                    image_url = "https:" + image_url
                
                file_ext = os.path.splitext(image_url)[1]
                save_path = os.path.join(output_dir, f"konachan_{post['id']}{file_ext}")
                
                if download_image(image_url, save_path):
                    downloaded += 1
                    sleep(1)
                    
        return downloaded
    except Exception as e:
        print(f"Konachan采集失败: {e}")
        return 0

def collect_smart_images(tags, limit, output_dir, api_key=None, user=None):
    """
    智能采集：依次尝试 Safebooru -> Danbooru -> Konachan
    这取代了原有的 Pixiv 采集逻辑，因为通过脚本直接采集 Pixiv 极其困难且容易封号。
    这些图站实际上就是 Pixiv 的镜像，使用相同的标签搜索可以得到精准的角色图片。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"=== 开始为 [{tags}] 采集图片，目标: {limit} 张 ===")
    
    total_downloaded = 0
    
    # 1. 尝试 Safebooru (最容易成功，无需认证)
    if total_downloaded < limit:
        print("--- 尝试源: Safebooru ---")
        count = collect_from_safebooru(tags, limit - total_downloaded, output_dir)
        total_downloaded += count
    
    # 2. 尝试 Danbooru
    if total_downloaded < limit:
        print("--- 尝试源: Danbooru ---")
        count = collect_from_danbooru(tags, limit - total_downloaded, output_dir, api_key, user)
        total_downloaded += count

    # 3. 尝试 Konachan
    if total_downloaded < limit:
        print("--- 尝试源: Konachan ---")
        count = collect_from_konachan(tags, limit - total_downloaded, output_dir)
        total_downloaded += count

    # 4. 只有在完全失败时才使用本地样本，避免生成不匹配的图
    if total_downloaded == 0:
        print("警告: 所有在线来源均未找到匹配图片，尝试放宽搜索条件...")
        # 尝试只搜索第一个标签（通常是角色名）
        simple_tag = tags.split()[0]
        if simple_tag != tags:
            print(f"尝试简化标签搜索: {simple_tag}")
            return collect_smart_images(simple_tag, limit, output_dir, api_key, user)
        else:
            print("无法下载匹配图片。")
    
    print(f"采集完成，共下载 {total_downloaded} 张图片")
    return total_downloaded

def collect_wuthering_waves_specific_characters(limit, output_dir, api_key=None, user=None):
    """专门采集鸣潮特定角色数据"""
    # 修正了标签格式，使其符合 Booru 风格 (Character_Name_(Series))
    # 这样才能搜索到准确的图片
    characters = [
        {"name": "守岸人", "tag": "shorekeeper_(wuthering_waves)"},
        {"name": "椿", "tag": "camellya_(wuthering_waves)"}, # 国际服名为 Camellya
        {"name": "卡提西亚", "tag": "katya_(wuthering_waves)"}, # 如果是尘白禁区的Katya需改为 katya_(snowbreak)
        # 备用：如果是指 炽霞 (Chixia)
        # {"name": "炽霞", "tag": "chixia_(wuthering_waves)"},
    ]
    
    total_downloaded = 0
    
    for char_info in characters:
        char_name = char_info["name"]
        char_tag = char_info["tag"]
        # 为每个角色创建独立文件夹
        char_output_dir = os.path.join(output_dir, f"鸣潮_{char_name}")
        
        print(f"\n>>> 处理角色: {char_name} (标签: {char_tag})")
        downloaded = collect_smart_images(char_tag, limit, char_output_dir, api_key, user)
        total_downloaded += downloaded
        sleep(2)
    
    return total_downloaded

def collect_genshin_impact_characters(limit, output_dir, api_key=None, user=None):
    """专门采集原神角色数据"""
    characters = [
        {"name": "荧", "tag": "lumine_(genshin_impact)"},
        {"name": "空", "tag": "aether_(genshin_impact)"},
        {"name": "琴", "tag": "jean_(genshin_impact)"},
        {"name": "丽莎", "tag": "lisa_(genshin_impact)"},
        {"name": "芭芭拉", "tag": "barbara_(genshin_impact)"},
        {"name": "温迪", "tag": "venti_(genshin_impact)"}
    ]
    
    total_downloaded = 0
    
    for char_info in characters:
        char_name = char_info["name"]
        char_tag = char_info["tag"]
        char_output_dir = os.path.join(output_dir, f"原神_{char_name}")
        
        print(f"\n>>> 处理角色: {char_name} (标签: {char_tag})")
        downloaded = collect_smart_images(char_tag, limit, char_output_dir, api_key, user)
        total_downloaded += downloaded
        sleep(2)
    
    return total_downloaded

def collect_single_character_data(character_name, limit, output_dir, api_key=None, user=None):
    """采集单角色数据"""
    # 智能推断标签格式
    tags = character_name
    
    # 游戏标签映射字典
    game_tag_mappings = {
        "鸣潮": "wuthering_waves",
        "原神": "genshin_impact",
        "蔚蓝档案": "blue_archive",
        "blue archive": "blue_archive"
    }
    
    # 蔚蓝档案角色标准标签映射
    blue_archive_tag_mappings = {
        "星野": "hoshino_(blue_archive)",
        "白子": "shiroko_(blue_archive)",
        "一之濑明日奈": "ichinose_asuna_(blue_archive)",
        "黑子": "kuroko_(blue_archive)",
        "阿罗娜": "arona_(blue_archive)",
        "宫子": "miyako_(blue_archive)",
        "日奈": "hina_(blue_archive)",
        "优花梨": "yuuka_(blue_archive)",
        "hoshino": "hoshino_(blue_archive)",
        "shiroko": "shiroko_(blue_archive)",
        "ichinose_asuna": "ichinose_asuna_(blue_archive)",
        "kuroko": "kuroko_(blue_archive)",
        "arona": "arona_(blue_archive)",
        "miyako": "miyako_(blue_archive)",
        "hina": "hina_(blue_archive)",
        "yuuka": "yuuka_(blue_archive)"
    }
    
    # 鸣潮角色标准标签映射
    wuthering_waves_tag_mappings = {
        "守岸人": "shorekeeper_(wuthering_waves)",
        "椿": "camellya_(wuthering_waves)",
        "卡提西亚": "katya_(wuthering_waves)",
        "anby": "anby_(wuthering_waves)",
        "bianca": "bianca_(wuthering_waves)",
        "corin": "corin_(wuthering_waves)",
        "lin": "lin_(wuthering_waves)",
        "lyra": "lyra_(wuthering_waves)",
        "seth": "seth_(wuthering_waves)"
    }
    
    # 原神角色标准标签映射
    genshin_impact_tag_mappings = {
        "荧": "lumine_(genshin_impact)",
        "空": "aether_(genshin_impact)",
        "琴": "jean_(genshin_impact)",
        "丽莎": "lisa_(genshin_impact)",
        "芭芭拉": "barbara_(genshin_impact)",
        "温迪": "venti_(genshin_impact)",
        "lumine": "lumine_(genshin_impact)",
        "aether": "aether_(genshin_impact)",
        "jean": "jean_(genshin_impact)",
        "lisa": "lisa_(genshin_impact)",
        "barbara": "barbara_(genshin_impact)",
        "venti": "venti_(genshin_impact)"
    }
    
    # 检测游戏类型并生成标准标签
    game_detected = None
    character_name_clean = character_name
    
    # 检测游戏前缀
    for game_cn, game_en in game_tag_mappings.items():
        if game_cn in character_name:
            game_detected = game_en
            character_name_clean = character_name.replace(game_cn + "_", "").replace(game_cn + " ", "")
            break
    
    # 处理不同游戏的标签映射
    if game_detected == "blue_archive":
        # 蔚蓝档案角色标签映射
        if character_name_clean in blue_archive_tag_mappings:
            tags = blue_archive_tag_mappings[character_name_clean]
        else:
            # 如果没有精确映射，使用通用格式
            tags = f"{character_name_clean} blue_archive"
            print(f"提示: 未找到 {character_name_clean} 的标准标签映射，使用通用标签格式")
            
    elif game_detected == "wuthering_waves":
        # 鸣潮角色标签映射
        if character_name_clean in wuthering_waves_tag_mappings:
            tags = wuthering_waves_tag_mappings[character_name_clean]
        else:
            tags = f"{character_name_clean} wuthering_waves"
            print(f"提示: 未找到 {character_name_clean} 的标准标签映射，使用通用标签格式")
            
    elif game_detected == "genshin_impact":
        # 原神角色标签映射
        if character_name_clean in genshin_impact_tag_mappings:
            tags = genshin_impact_tag_mappings[character_name_clean]
        else:
            tags = f"{character_name_clean} genshin_impact"
            print(f"提示: 未找到 {character_name_clean} 的标准标签映射，使用通用标签格式")
            
    else:
        # 自动检测游戏类型
        for game_keyword in ["blue_archive", "wuthering_waves", "genshin_impact"]:
            if game_keyword in character_name.lower():
                tags = character_name
                break
        else:
            # 默认处理
            print("提示: 建议使用标准格式，如 '游戏名_角色名' 或直接使用标准Booru标签")
            tags = character_name
    
    print(f"使用标签: {tags}")
    return collect_smart_images(tags, limit, output_dir, api_key, user)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="二次元角色图片采集脚本 (Fix)")
    parser.add_argument("--mode", choices=["single", "honkai star rail", "genshin_impact"], required=True, help="采集模式")
    parser.add_argument("--character", help="角色名称/标签 (仅在 single 模式下需要，推荐英文，如 'shorekeeper')")
    parser.add_argument("--limit", type=int, default=50, help="每个角色采集图片数量")
    parser.add_argument("--output_dir", help="输出目录")
    parser.add_argument("--api_key", help="Danbooru API密钥 (可选)")
    parser.add_argument("--user", help="Danbooru用户名 (可选)")
    
    args = parser.parse_args()
    
    # 确定输出目录根路径
    root_output_dir = args.output_dir if args.output_dir else "tests/test_images"

    # 执行采集
    if args.mode == "single":
        if not args.character:
            print("错误: 在 single 模式下必须指定角色名称/标签")
            return
        # single模式直接输出到指定文件夹
        final_dir = os.path.join(root_output_dir, "single_character", args.character)
        collect_single_character_data(args.character, args.limit, final_dir, args.api_key, args.user)
        
    elif args.mode == "wuthering_waves_specific":
        final_dir = os.path.join(root_output_dir, "wuthering_waves")
        collect_wuthering_waves_specific_characters(args.limit, final_dir, args.api_key, args.user)
        
    elif args.mode == "genshin_impact":
        final_dir = os.path.join(root_output_dir, "genshin_impact")
        collect_genshin_impact_characters(args.limit, final_dir, args.api_key, args.user)

if __name__ == "__main__":
    main()