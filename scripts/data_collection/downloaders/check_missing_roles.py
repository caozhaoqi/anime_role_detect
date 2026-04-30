#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查缺少URL的角色
"""

import os

# 配置
ROLE_LIST_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'

# 拼音映射表（用于匹配）
PINYIN_MAPPING = {
    '阿洛娜': ['a1lu4', 'a1luo2na4', 'a1luo4na4'],
    '普拉娜': ['pu3la1na4'],
    '纳西妲': ['na4xi1da2', 'na4xi1da4'],
    '缇宝': ['ti2bao3'],
    '可莉': ['ke3li4', 'ke3li2'],
    '迪奥娜': ['di2ao4na4'],
    '瑶瑶': ['yao2yao2'],
    '希格雯': ['xi1ge2wen2'],
    '蕾贝': ['lei3bei4'],
    '黑塔': ['hei1ta3'],
    '符玄': ['fu2xuan2'],
    '七七': ['qi1qi1'],
    '早柚': ['zao3you4'],
    '多莉': ['duo1li4'],
    '卡齐娜': ['ka3qi2na4'],
    '三月七': ['san1yue4qi1'],
    '花火': ['hua1huo3'],
    '银狼': ['yin2lang2'],
    '天童爱丽丝': ['tian1tong2ai4li4si1', 'aris'],
    '早雾': ['zao3wu4', 'hayiri'],
    '维里奈': ['wei2li3nai4', 'verina'],
    '安可': ['an1ke3', 'encore'],
    '釉壶': ['you4hu2'],
    '洛可可': ['luo4ke4ke4', 'roccia'],
    '鹿目圆': ['lu4mu4yuan2', 'madoka'],
    '晓美焰': ['xiao3mei3yan4', 'homura'],
    '血小板': ['xue3xiao3ban3'],
    '雷姆': ['lei2mu3', 'rem'],
    '拉姆': ['la1mu3', 'ram'],
    '康娜': ['kang1na4', 'kanna'],
    '四糸乃': ['si4mi4nai3', 'yoshino'],
    '凯露': ['kai3lu4', 'kyaru'],
    '克萝萝': ['ke4luo2luo2', 'klor'],
    '小闪': ['xiao3shan3', 'flash'],
    '伊莉雅': ['yi1li4ya3', 'illya'],
    '忍野忍': ['ren3ye3ren3', 'oshino'],
    '智乃': ['zhi4nai3', 'chino'],
    '小埋': ['xiao3mai2', 'tsumugi'],
    '纱雾': ['sha1wu4', 'sagiri'],
    '猫宫又奈': ['mao1gong1you4nai4', 'yanagi'],
    '德丽莎': ['de2li4sha1', 'theresa'],
    '布洛妮娅': ['bu4luo4ni2ya4', 'bronya'],
    '可琳': ['ke3lin2', 'kira'],
    '爱丽儿': ['ai4li4er3', 'ariel'],
    '神乐': ['shen1yue4', 'kagura'],
    '白上吹雪': ['bai2shang4chui1xue3', 'shirogane'],
    '月千夜': ['yue4qian1ye4', 'tsukiyo'],
    '芙丽希娅': ['fu2li4xi1ya4', 'furisia'],
    '莉塔拉': ['li4ta3la1', 'lita'],
    '维普蕾': ['wei2pu3lei3', 'viprey'],
    '夏克里': ['xia4ke4li3', 'shakri'],
    '纳甘': ['na4gan1', 'nagan'],
    '科谢尼娅': ['ke1xie4ni2ya4', 'koshenia'],
    '奇塔': ['qi2ta3', 'kita'],
    '寇尔芙': ['kou4er3fu2', 'korvu'],
    '克罗丽科': ['ke4luo2li4ke1', 'krokri'],
    '佩里缇亚': ['pei4li3ti2ya4', 'peritia'],
    '阿尼亚': ['a1ni4ya4', 'anya'],
    '洛茜': ['luo4qian4', 'rosci'],
    '祢豆子': ['ni2dou4zi5', 'nezuko'],
    '希儿': ['xi1er3', 'seele'],
    '杏': ['xing4', 'an'],
    '伊瑟琳': ['yi1se4lin2', 'iselin'],
    '芙兰': ['fu2lan2', 'fran'],
    '菲米莉丝': ['fei1mi3li4si1', 'fimilis'],
}


def get_existing_roles():
    """获取已有的角色URL文件名"""
    existing = set()
    if os.path.exists(URL_DIR):
        for filename in os.listdir(URL_DIR):
            if filename.endswith('_img.txt'):
                role_pinyin = filename.replace('_img.txt', '')
                existing.add(role_pinyin)
    return existing


def get_all_roles():
    """获取完整角色列表"""
    roles = []
    with open(ROLE_LIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # 获取中文角色名（第一个字段）
                parts = line.split(' ')
                chinese_name = parts[0]
                roles.append(chinese_name)
    return roles


def check_missing():
    """检查缺少的角色"""
    all_roles = get_all_roles()
    existing_roles = get_existing_roles()
    
    missing = []
    for role in all_roles:
        # 检查是否有对应的拼音文件
        pinyins = PINYIN_MAPPING.get(role, [])
        
        # 如果有拼音映射，检查是否有任何一个拼音文件存在
        if pinyins:
            found = False
            for pinyin in pinyins:
                if pinyin in existing_roles:
                    found = True
                    break
            if not found:
                missing.append(role)
        else:
            # 如果没有拼音映射，标记为需要确认
            missing.append(f"{role} (无拼音映射)")
    
    return missing, len(all_roles)


def main():
    """主函数"""
    missing, total = check_missing()
    
    print(f"=== 角色URL检查结果 ===")
    print(f"总角色数: {total}")
    print(f"已采集URL: {total - len(missing)}")
    print(f"缺少URL: {len(missing)}")
    print()
    
    if missing:
        print("缺少URL的角色:")
        for i, role in enumerate(missing, 1):
            print(f"  {i}. {role}")
    else:
        print("✓ 所有角色的URL都已采集完成！")


if __name__ == '__main__':
    main()
