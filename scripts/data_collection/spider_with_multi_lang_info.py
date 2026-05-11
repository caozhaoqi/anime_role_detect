#!/usr/bin/env python3
"""利用loli-role.txt中的多语言名称采集图片不足50张的角色"""
import requests
import time
import os
from pathlib import Path

API_BASE = "http://localhost:33333/api/v1.2.5.260305/sis"
REORGANIZED_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
ROLE_LIST_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
SPIDER_DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data'
TARGET_COUNT = 50

# 拼音映射（用于匹配目录名）
PINYIN_MAPPING = {
    '阿洛娜': 'a1luo4na4',
    '普拉娜': 'pu3la1na4',
    '纳西妲': 'na4xi1da2',
    '缇宝': 'ti2bao3',
    '可莉': 'ke3li4',
    '迪奥娜': 'di2ao4na4',
    '瑶瑶': 'yao2yao2',
    '希格雯': 'xi1ge2wen2',
    '蕾贝': 'lei3bei4',
    '黑塔': 'hei1ta3',
    '符玄': 'fu2xuan2',
    '七七': 'qi1qi1',
    '早柚': 'zao3you4',
    '多莉': 'duo1li4',
    '卡齐娜': 'ka3qi2na4',
    '三月七': 'san1yue4qi1',
    '花火': 'hua1huo3',
    '银狼': 'yin2lang2',
    '天童爱丽丝': 'tian1tong2ai4li4si1',
    '早雾': 'zao3wu4',
    '维里奈': 'wei2li3nai4',
    '安可': 'an1ke3',
    '釉瑚': 'you4hu2',
    '洛可可': 'luo4ke4ke4',
    '鹿目圆': 'lu4mu4yuan2',
    '晓美焰': 'xiao3mei3yan4',
    '血小板': 'xue4xiao3ban3',
    '雷姆': 'lei2mu3',
    '拉姆': 'la1mu3',
    '康娜': 'kang1na4',
    '四糸乃': 'si4mi4nai3',
    '凯露': 'kai3lu4',
    '克萝萝': 'ke4luo2luo2',
    '小闪': 'xiao3shan3',
    '伊莉雅': 'yi1li4ya3',
    '忍野忍': 'ren3ye3ren3',
    '智乃': 'zhi4nai3',
    '小埋': 'xiao3mai2',
    '纱雾': 'sha1wu4',
    '猫宫又奈': 'mao1gong1you4nai4',
    '德丽莎': 'de2li4sha1',
    '布洛妮娅': 'bu4luo4ni2ya4',
    '可琳': 'ke3lin2',
    '爱丽儿': 'ai4li4er3',
    '神乐': 'shen1yue4',
    '白上吹雪': 'bai2shang4chui1xue3',
    '月千夜': 'yue4qian1ye4',
    '芙丽希娅': 'fu2li4xi1ya4',
    '莉塔拉': 'li4ta3la1',
    '维普蕾': 'wei2pu3lei3',
    '夏克里': 'xia4ke4li3',
    '纳甘': 'na4gan1',
    '科谢尼娅': 'ke1xie4ni2ya4',
    '奇塔': 'qi2ta3',
    '寇尔芙': 'kou4er3fu2',
    '克罗丽科': 'ke4luo2li4ke1',
    '佩里缇亚': 'pei4li3ti2ya4',
    '阿尼亚': 'a1ni4ya4',
    '洛茜': 'luo4qian4',
    '祢豆子': 'mi2dou4zi',
    '希儿': 'xi1er3',
    '杏': 'xing4',
    '伊瑟琳': 'yi1se4lin2',
    '芙兰': 'fu2lan2',
    '菲米莉丝': 'fei1mi3li4si1',
    '克拉拉': 'ke4la1la1'
}

def parse_role_list(file_path):
    """解析角色名单文件，提取多语言名称"""
    roles = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            if len(parts) >= 4:
                cn_name = parts[0]
                game = parts[1]
                en_name = parts[2]
                jp_name = ' '.join(parts[3:])
                roles[cn_name] = {
                    'game': game,
                    'cn': cn_name,
                    'en': en_name,
                    'jp': jp_name
                }
    return roles

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

def reset_spider():
    """重置爬虫状态"""
    try:
        resp = requests.post(f"{API_BASE}/spider/reset", timeout=30)
        print(f"    重置状态: {resp.json().get('msg', '未知')}")
        time.sleep(2)
    except Exception as e:
        print(f"    重置失败: {e}")

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
                print(f"      状态: {'运行中' if is_running else '空闲'}, 关键词: {keyword}, 计数: {count}")
                if not is_running:
                    return True
        except Exception as e:
            print(f"      检查状态异常: {e}")
        time.sleep(8)
    return False

def spider_with_keyword(keyword, lang):
    """使用指定关键词爬取"""
    print(f"      [{lang}] 尝试: {keyword}")
    try:
        resp = requests.post(f"{API_BASE}/spider_start/single?key_word={keyword}", timeout=30)
        result = resp.json()
        print(f"        启动结果: {result.get('msg', '未知')}")

        if result.get('code') == 0:
            print("        等待爬取完成...")
            success = wait_for_completion()
            if success:
                print(f"        ✅ [{lang}]爬取完成")
                return True
            else:
                print(f"        ⚠️ [{lang}]等待超时")
        else:
            print(f"        ❌ [{lang}]启动失败: {result.get('msg', '未知错误')}")
    except Exception as e:
        print(f"        ❌ [{lang}]爬取异常: {e}")
    return False

def spider_role_with_multi_lang(cn_name, names):
    """使用多语言名称爬取单个角色"""
    pinyin = PINYIN_MAPPING.get(cn_name)
    if not pinyin:
        print(f"      ❌ 未找到拼音映射: {cn_name}")
        return
    
    print(f"\n{'='*70}")
    print(f"🚀 开始爬取: {cn_name} ({names['game']})")
    print(f"   中文: {names['cn']}")
    print(f"   英文: {names['en']}")
    print(f"   日文: {names['jp']}")
    print(f"{'='*70}")

    initial_url_count = get_url_count(pinyin)
    current_img_count = get_current_image_count(pinyin)
    print(f"   当前图片: {current_img_count} 张")
    print(f"   当前URL: {initial_url_count} 个")

    # 需要补充的数量
    needed = TARGET_COUNT - current_img_count
    if needed <= 0:
        print(f"   ✅ 已达标，跳过")
        return

    # 需要的URL数量（考虑下载失败率）
    needed_urls = needed * 2

    # 按优先级尝试不同语言名称
    keywords = [
        (names['cn'], '中文'),
        (names['en'], '英文'),
        (names['jp'], '日文')
    ]

    for keyword, lang in keywords:
        reset_spider()
        if spider_with_keyword(keyword, lang):
            # 检查URL是否增加
            new_url_count = get_url_count(pinyin)
            added = new_url_count - initial_url_count
            print(f"        新增URL: {added} 个")
            initial_url_count = new_url_count

            # 如果已经有足够的URL，停止尝试
            if initial_url_count >= needed_urls:
                print(f"        ✅ URL已足够，停止爬取")
                break

        time.sleep(2)

    final_url_count = get_url_count(pinyin)
    print(f"\n   爬取完成: {cn_name}")
    print(f"   URL数量: {initial_url_count} → {final_url_count} (+{final_url_count - initial_url_count})")

def main():
    print("="*70)
    print("🌍 使用多语言名称采集图片不足50张的角色")
    print("   数据源: auto_spider_img/loli-role.txt")
    print("="*70)

    # 解析角色名单
    roles = parse_role_list(ROLE_LIST_PATH)
    print(f"\n📋 已加载 {len(roles)} 个角色信息")

    # 获取需要采集的角色列表
    roles_needing_more = []
    for cn_name, names in roles.items():
        pinyin = PINYIN_MAPPING.get(cn_name)
        if pinyin:
            count = get_current_image_count(pinyin)
            if count < TARGET_COUNT:
                roles_needing_more.append({
                    'cn_name': cn_name,
                    'names': names,
                    'current_count': count,
                    'needed': TARGET_COUNT - count
                })

    roles_needing_more.sort(key=lambda x: x['needed'], reverse=True)

    print(f"\n找到 {len(roles_needing_more)} 个角色需要补充图片")
    for role in roles_needing_more:
        print(f"  {role['cn_name']}: 当前 {role['current_count']} 张, 需要 {role['needed']} 张")

    # 开始逐个采集
    for role in roles_needing_more:
        print(f"\n📋 处理: {role['cn_name']}")
        print(f"   所属游戏: {role['names']['game']}")
        spider_role_with_multi_lang(role['cn_name'], role['names'])

    print("\n" + "="*70)
    print("🎉 多语言采集任务已完成！")
    print("="*70)
    print("\n注意：爬取完成后需要执行下载脚本将图片保存到本地")

if __name__ == '__main__':
    main()
