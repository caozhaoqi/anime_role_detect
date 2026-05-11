#!/usr/bin/env python3
"""为图片不足50张的角色采集图片"""
import requests
import time
import os
from pathlib import Path

API_BASE = "http://localhost:33333/api/v1.2.5.260305/sis"
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

def get_roles_needing_more():
    """获取图片不足50张的角色列表"""
    roles = []
    for pinyin, name in PINYIN_MAPPING.items():
        count = get_current_image_count(pinyin)
        if count < TARGET_COUNT:
            roles.append({
                'name': name,
                'pinyin': pinyin,
                'current_count': count,
                'needed': TARGET_COUNT - count
            })
    # 按需要数量排序
    roles.sort(key=lambda x: x['needed'], reverse=True)
    return roles

def reset_spider():
    """重置爬虫状态"""
    try:
        resp = requests.post(f"{API_BASE}/spider/reset", timeout=30)
        print(f"重置状态: {resp.json().get('msg', '未知')}")
        time.sleep(3)
    except Exception as e:
        print(f"重置失败: {e}")

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
                print(f"  状态: {'运行中' if is_running else '空闲'}, 关键词: {keyword}, 计数: {count}")
                if not is_running:
                    return True
        except Exception as e:
            print(f"检查状态异常: {e}")
        time.sleep(10)
    return False

def spider_role(role_name):
    """爬取单个角色"""
    print(f"\n{'='*60}")
    print(f"🚀 开始爬取: {role_name}")
    print(f"{'='*60}")

    reset_spider()

    try:
        resp = requests.post(f"{API_BASE}/spider_start/single?key_word={role_name}", timeout=30)
        result = resp.json()
        print(f"启动结果: {result.get('msg', '未知')}")

        if result.get('code') == 0:
            print("等待爬取完成...")
            success = wait_for_completion()
            if success:
                print("✅ 爬取完成")
            else:
                print("⚠️ 等待超时")
        else:
            print(f"❌ 启动失败: {result.get('msg', '未知错误')}")
    except Exception as e:
        print(f"❌ 爬取异常: {e}")

    time.sleep(2)

def main():
    print("="*60)
    print("📷 开始采集图片不足50张的角色")
    print("="*60)

    # 获取需要采集的角色列表
    roles_needing_more = get_roles_needing_more()
    print(f"\n找到 {len(roles_needing_more)} 个角色需要补充图片")

    # 显示需要采集的角色
    print("\n需要采集的角色（按需求排序）:")
    for role in roles_needing_more:
        print(f"  {role['name']} ({role['pinyin']}): 当前 {role['current_count']} 张, 需要补充 {role['needed']} 张")

    # 开始逐个采集
    for role in roles_needing_more:
        print(f"\n📋 处理: {role['name']}")
        print(f"   当前图片: {role['current_count']} 张")
        print(f"   需要补充: {role['needed']} 张")

        # 检查URL文件是否已经有足够的URL
        url_file = Path(f"spider_image_system/data/img_url/{role['pinyin']}_img.txt")
        url_count = 0
        if url_file.exists():
            with open(url_file, 'r', encoding='utf-8') as f:
                url_count = len([line for line in f if line.strip()])
        print(f"   当前URL数量: {url_count}")

        if url_count < TARGET_COUNT * 2:  # 需要两倍的URL（考虑下载失败）
            spider_role(role['name'])
        else:
            print(f"   ✅ URL已足够，跳过爬取")

    print("\n" + "="*60)
    print("🎉 采集任务已全部提交！")
    print("="*60)
    print("\n注意：爬取完成后需要执行下载脚本将图片保存到本地")

if __name__ == '__main__':
    main()