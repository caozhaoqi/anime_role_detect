import os
import shutil
import json
from pathlib import Path

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
    'you4hu2': '釉壶',
    'luo4ke4ke4': '洛可可',
    'lu4mu4yuan2': '鹿目圆',
    'xiao3mei3yan4': '晓美焰',
    'xue4xiao3ban3': '血小板',
    'lei2mu3': '雷姆',
    'la1mu3': '拉姆',
    'kang1na4': '康娜',
    'si4mi4nai3': '四糸乃',
    'kai3lu4': '凯露',
    'ke4luo2luo2': '克萝萝',
    'xiao3shan3': '小闪',
    'yi1li4ya3': '伊莉雅',
    'ren3ye3ren3': '忍野忍',
    'zhi4nai3': '智乃',
    'xiao3mai2': '小埋',
    'sha1wu4': '纱雾',
    'mao1gong1you4nai4': '猫宫又奈',
    'de2li4sha1': '德丽莎',
    'bu4luo4ni2ya4': '布洛妮娅',
    'ke3lin2': '可琳',
    'ai4li4er3': '爱丽儿',
    'shen1yue4': '神乐',
    'bai2shang4chui1xue3': '白上吹雪',
    'yue4qian1ye4': '月千夜',
    'fu2li4xi1ya4': '芙丽希娅',
    'li4ta3la1': '莉塔拉',
    'wei2pu3lei3': '维普蕾',
    'xia4ke4li3': '夏克里',
    'na4gan1': '纳甘',
    'ke1xie4ni2ya4': '科谢尼娅',
    'qi2ta3': '奇塔',
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

def read_role_list(file_path):
    """读取角色名单"""
    roles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if parts:
                    roles.append(parts[0])
    return roles

def get_pinyin_by_name(name):
    """根据中文名获取拼音"""
    for pinyin, chinese_name in PINYIN_MAPPING.items():
        if chinese_name == name:
            return pinyin
    return None

def reorganize_dataset(role_list_path, source_dir, target_dir, max_images_per_role=50):
    """重新整理数据集"""
    roles = read_role_list(role_list_path)
    print(f"读取到 {len(roles)} 个角色")
    
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    stats = []
    total_moved = 0
    
    for role_name in roles:
        pinyin = get_pinyin_by_name(role_name)
        if not pinyin:
            print(f"  未找到角色 {role_name} 的拼音映射，跳过")
            continue
        
        source_role_dir = os.path.join(source_dir, pinyin)
        target_role_dir = os.path.join(target_dir, pinyin)
        
        if not os.path.exists(source_role_dir):
            print(f"  源目录不存在: {source_role_dir}")
            stats.append({'name': role_name, 'pinyin': pinyin, 'count': 0})
            continue
        
        Path(target_role_dir).mkdir(parents=True, exist_ok=True)
        
        files = []
        for f in os.listdir(source_role_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                files.append(f)
        
        files.sort()
        selected_files = files[:max_images_per_role]
        
        for filename in selected_files:
            src = os.path.join(source_role_dir, filename)
            dst = os.path.join(target_role_dir, filename)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
        
        count = len(selected_files)
        total_moved += count
        stats.append({'name': role_name, 'pinyin': pinyin, 'count': count})
        print(f"  {role_name} ({pinyin}): {count} 张图片")
    
    print(f"\n整理完成！")
    print(f"总计: {len(stats)} 个角色, {total_moved} 张图片")
    
    with open(os.path.join(target_dir, 'dataset_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return stats

def main():
    role_list_path = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
    source_dir = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
    target_dir = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
    
    print("="*60)
    print("重新整理数据集")
    print("="*60)
    print(f"角色名单: {role_list_path}")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"每角色最大图片数: 50")
    print()
    
    reorganize_dataset(role_list_path, source_dir, target_dir, max_images_per_role=50)

if __name__ == '__main__':
    main()