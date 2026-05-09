import os
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

ORGANIZED_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
REORGANIZED_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
OUTPUT_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/角色分布对比报告.md'

def count_images(dir_path):
    """统计目录中图片数量"""
    if not os.path.exists(dir_path):
        return 0
    count = 0
    for f in os.listdir(dir_path):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            count += 1
    return count

def generate_report():
    print("="*60)
    print("生成角色分布对比报告")
    print("="*60)
    
    org_stats = {}
    reorg_stats = {}
    
    # 统计 organized_images
    for item in os.listdir(ORGANIZED_DIR):
        item_path = os.path.join(ORGANIZED_DIR, item)
        if os.path.isdir(item_path) and item in PINYIN_MAPPING:
            count = count_images(item_path)
            org_stats[item] = count
    
    # 统计 reorganized_dataset
    for item in os.listdir(REORGANIZED_DIR):
        item_path = os.path.join(REORGANIZED_DIR, item)
        if os.path.isdir(item_path) and item in PINYIN_MAPPING:
            count = count_images(item_path)
            reorg_stats[item] = count
    
    # 按拼音排序
    sorted_pinyins = sorted(org_stats.keys())
    
    # 生成报告内容
    content = f"""# 角色分布对比报告

生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 概览

| 目录 | 角色数 | 图片总数 |
|------|--------|----------|
| organized_images | {len(org_stats)} | {sum(org_stats.values())} |
| reorganized_dataset | {len(reorg_stats)} | {sum(reorg_stats.values())} |

---

## 详细分布

| 中文名 | 拼音 | organized_images | reorganized_dataset | 差异 |
|--------|------|------------------|---------------------|------|
"""
    
    for pinyin in sorted_pinyins:
        org_count = org_stats.get(pinyin, 0)
        reorg_count = reorg_stats.get(pinyin, 0)
        diff = reorg_count - org_count
        diff_str = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "-"
        content += f"| {PINYIN_MAPPING[pinyin]} | {pinyin} | {org_count} | {reorg_count} | {diff_str} |\n"
    
    content += """

---

## 统计分析

### organized_images 分布
"""
    
    org_sorted = sorted(org_stats.items(), key=lambda x: x[1], reverse=True)
    content += f"- 最多图片: {PINYIN_MAPPING[org_sorted[0][0]]} ({org_sorted[0][1]} 张)\n"
    content += f"- 最少图片: {PINYIN_MAPPING[org_sorted[-1][0]]} ({org_sorted[-1][1]} 张)\n"
    content += f"- 平均: {sum(org_stats.values()) / len(org_stats):.1f} 张/角色\n"
    
    content += """

### reorganized_dataset 分布
"""
    
    reorg_sorted = sorted(reorg_stats.items(), key=lambda x: x[1], reverse=True)
    content += f"- 最多图片: {PINYIN_MAPPING[reorg_sorted[0][0]]} ({reorg_sorted[0][1]} 张)\n"
    content += f"- 最少图片: {PINYIN_MAPPING[reorg_sorted[-1][0]]} ({reorg_sorted[-1][1]} 张)\n"
    content += f"- 平均: {sum(reorg_stats.values()) / len(reorg_stats):.1f} 张/角色\n"
    
    content += """

---

## 备注

1. `organized_images`: 原始数据集，包含所有爬取的图片（含NSFW内容）
2. `reorganized_dataset`: 重新整理的精选数据集，每角色最多50张
3. 差异列显示 reorganized_dataset 相对于 organized_images 的变化
4. 部分角色在 reorganized_dataset 中数量超过 organized_images，是因为从 trash_nsfw 目录补充了数据

"""
    
    # 写入文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"报告已生成: {OUTPUT_FILE}")
    print(f"organized_images: {len(org_stats)} 角色, {sum(org_stats.values())} 图片")
    print(f"reorganized_dataset: {len(reorg_stats)} 角色, {sum(reorg_stats.values())} 图片")

if __name__ == '__main__':
    generate_report()