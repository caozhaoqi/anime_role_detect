#!/usr/bin/env python3
"""为图片不足50张的角色使用多语言名称采集图片"""
import requests
import time
import os
from pathlib import Path

API_BASE = "http://localhost:33333/api/v1.2.5.260305/sis"
REORGANIZED_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
TARGET_COUNT = 50

# 支持多语言名称的角色映射
ROLE_NAMES = {
    'fu2li4xi1ya4': {
        'cn': '芙丽希娅',
        'en': 'Furishia',
        'jp': 'フリシア'
    },
    'you4hu2': {
        'cn': '釉壶',
        'en': 'Yuh壶',
        'jp': 'ユーフ'
    },
    'ai4li4er3': {
        'cn': '爱丽儿',
        'en': 'Ariel',
        'jp': 'アリエル'
    },
    'yue4qian1ye4': {
        'cn': '月千夜',
        'en': 'Tsukichiya',
        'jp': '月千夜'
    },
    'ke4luo2luo2': {
        'cn': '克萝萝',
        'en': 'Kurolo',
        'jp': 'クロロ'
    },
    'luo4ke4ke4': {
        'cn': '洛可可',
        'en': 'Rokoko',
        'jp': 'ロココ'
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

def reset_spider():
    """重置爬虫状态"""
    try:
        resp = requests.post(f"{API_BASE}/spider/reset", timeout=30)
        print(f"  重置状态: {resp.json().get('msg', '未知')}")
        time.sleep(2)
    except Exception as e:
        print(f"  重置失败: {e}")

def wait_for_completion(timeout=600):
    """等待爬虫完成"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            status = requests.get(f"{API_BASE}/spider/status", timeout=10)
            data = status.json()
            if data.get('code') == 0:
                is_running = data['data']['is_running']
                keyword = data['data'].get('current_keyword', '')
                count = data['data'].get('current_count', 0)
                print(f"    状态: {'运行中' if is_running else '空闲'}, 关键词: {keyword}, 计数: {count}")
                if not is_running:
                    return True
        except Exception as e:
            print(f"    检查状态异常: {e}")
        time.sleep(8)
    return False

def spider_with_name(role_pinyin, name, lang):
    """使用指定语言名称爬取"""
    print(f"    [{lang}] 尝试: {name}")
    try:
        resp = requests.post(f"{API_BASE}/spider_start/single?key_word={name}", timeout=30)
        result = resp.json()
        print(f"      启动结果: {result.get('msg', '未知')}")

        if result.get('code') == 0:
            print("      等待爬取完成...")
            success = wait_for_completion()
            if success:
                print(f"      ✅ [{lang}]爬取完成")
                return True
            else:
                print(f"      ⚠️ [{lang}]等待超时")
        else:
            print(f"      ❌ [{lang}]启动失败: {result.get('msg', '未知错误')}")
    except Exception as e:
        print(f"      ❌ [{lang}]爬取异常: {e}")
    return False

def spider_role_multi_lang(role_pinyin, names):
    """使用多语言名称爬取单个角色"""
    print(f"\n{'='*60}")
    print(f"🚀 开始爬取: {names['cn']} ({role_pinyin})")
    print(f"{'='*60}")

    # 获取当前URL数量
    url_file = Path(f"spider_image_system/data/img_url/{role_pinyin}_img.txt")
    initial_count = 0
    if url_file.exists():
        with open(url_file, 'r', encoding='utf-8') as f:
            initial_count = len([line for line in f if line.strip()])
    print(f"  当前URL数量: {initial_count}")

    # 按优先级尝试不同语言名称
    languages = ['cn', 'en', 'jp']
    
    for lang in languages:
        reset_spider()
        name = names[lang]
        if spider_with_name(role_pinyin, name, lang):
            # 检查URL是否增加
            if url_file.exists():
                with open(url_file, 'r', encoding='utf-8') as f:
                    new_count = len([line for line in f if line.strip()])
                added = new_count - initial_count
                print(f"      新增URL: {added} 个")
                initial_count = new_count
        
        # 如果已经有足够的URL，停止尝试
        if initial_count >= TARGET_COUNT * 2:
            print(f"      ✅ URL已足够，停止爬取")
            break
        
        time.sleep(2)

def main():
    print("="*60)
    print("🌍 使用多语言名称采集图片不足50张的角色")
    print("="*60)

    # 获取需要采集的角色列表
    roles_needing_more = []
    for pinyin, names in ROLE_NAMES.items():
        count = get_current_image_count(pinyin)
        if count < TARGET_COUNT:
            roles_needing_more.append({
                'pinyin': pinyin,
                'names': names,
                'current_count': count,
                'needed': TARGET_COUNT - count
            })
    
    roles_needing_more.sort(key=lambda x: x['needed'], reverse=True)
    
    print(f"\n找到 {len(roles_needing_more)} 个角色需要补充图片")
    for role in roles_needing_more:
        print(f"  {role['names']['cn']}: 当前 {role['current_count']} 张, 需要 {role['needed']} 张")

    # 开始逐个采集
    for role in roles_needing_more:
        print(f"\n📋 处理: {role['names']['cn']}")
        print(f"   当前图片: {role['current_count']} 张")
        
        spider_role_multi_lang(role['pinyin'], role['names'])

    print("\n" + "="*60)
    print("🎉 多语言采集任务已完成！")
    print("="*60)

if __name__ == '__main__':
    main()