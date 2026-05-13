import os

def analyze_url_by_role():
    # 读取角色名单
    roles = []
    with open('auto_spider_img/loli-role.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                if '→' in line:
                    content = line.split('→')[1].strip()
                else:
                    content = line
                parts = content.split(' ')
                roles.append({
                    'chinese': parts[0],
                    'source': parts[1],
                    'english': parts[2] if len(parts) > 2 else '',
                    'japanese': parts[3] if len(parts) > 3 else ''
                })
    
    # 拼音映射表
    pinyin_map = {
        '阿洛娜': 'a1luo4na4',
        '普拉娜': 'pu3la1na4',
        '砂狼白子': 'sha1lang2bai2zi3',
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
        '派蒙': 'pai4meng2',
        '卡齐娜': 'ka3qi2na4',
        '三月七': 'san1yue4qi1',
        '花火': 'hua1huo3',
        '火花': 'hua1huo3',
        '银狼': 'yin2lang2',
        '天童爱丽丝': 'tian1tong2ai4li4si1',
        '早雾': 'zao3wu4',
        '维里奈': 'wei2li3nai4',
        '安可': 'an1ke3',
        '釉瑚': 'you4hu2',
        '鹿目圆': 'lu4mu4yuan2',
        '晓美焰': 'xiao3mei3yan4',
        '血小板': 'xue4xiao3ban3',
        '雷姆': 'lei2mu3',
        '拉姆': 'la1mu3',
        '康娜': 'kang1na4',
        '四糸乃': 'si4mi4nai3',
        '凯露': 'kai3lu4',
        '伊莉雅': 'yi1li4ya3',
        '忍野忍': 'ren3ye3ren3',
        '香风智乃': 'xiang1feng1zhi4nai3',
        '小埋': 'xiao3mai2',
        '纱雾': 'sha1wu4',
        '猫宫又奈': 'mao1gong1you4nai4',
        '德丽莎': 'de2li4sha1',
        '布洛妮娅': 'bu4luo4ni2ya4',
        '可琳': 'ke3lin2',
        '神乐': 'shen1yue4',
        '白上吹雪': 'bai2shang4chui1xue3',
        '月千夜': 'yue4qian1ye4',
        '莉塔拉': 'li4ta3la1',
        '维普蕾': 'wei2pu3lei3',
        '夏克里': 'sha1wu4',
        '纳甘': 'na4gan1',
        '科谢尼娅': 'ke1xie4ni2ya4',
        '寇尔芙': 'kou4er3fu2',
        '克罗丽科': 'ke4luo2li4ke1',
        '佩里缇亚': 'pei4li3ti2ya4',
        '阿尼亚': 'a1ni4ya4',
        '洛茜': 'luo4qian4',
        '灶门祢豆子': 'ni2dou4zi5',
        '希儿': 'xi1er3',
        '杏': 'kan1',
        '伊瑟琳': 'yi1se4lin2',
        '芙兰': 'fu2lan2',
        '菲米莉丝': 'fei1mi3li4si1',
        '克拉拉': 'ke1la1la1',
        '铃兰': 'ling2lan2',
        '白咲花': 'bai2xiao4hua1',
        '星野日向': 'xing1ye3ri4xiang4',
        '姬坂乃爱': 'ji1ban3nai4ai4',
        '种村小依': 'zhong3cun1xiao3yi1',
        '小之森夏音': 'xiao3zhi1sen1xia4yin1',
        '雏鹤爱': 'chu2he4ai4',
        '夜叉神天衣': 'ye4cha1shen2tian1yi1',
        '空银子': 'kong1yin2zi3',
        '早濑优香': 'zao3lai4you1xiang1',
        '一之濑明日奈': 'yi1zhi1lai4ming2ri4nai4',
        '空崎日奈': 'kong1qi2ri4nai4',
        '圣园未花': 'sheng4yuan2wei4hua1',
        '小鸟游星野': 'xiao3niao3you2xing1ye3'
    }
    
    # 获取所有URL文件
    url_dir = 'spider_image_system/data/img_url'
    url_files = {f.replace('_img.txt', ''): f for f in os.listdir(url_dir) if f.endswith('_img.txt')}
    
    print('=' * 80)
    print('          按角色统计URL分布')
    print('=' * 80)
    
    # 统计每个角色的URL数量
    role_stats = []
    total_urls = 0
    matched_roles = 0
    unmatched_roles = 0
    
    for role in roles:
        chinese = role['chinese']
        english = role['english']
        japanese = role['japanese']
        
        # 查找匹配的URL文件
        url_count = 0
        matched_file = None
        
        # 尝试拼音匹配
        if chinese in pinyin_map:
            pinyin = pinyin_map[chinese]
            if pinyin in url_files:
                with open(os.path.join(url_dir, url_files[pinyin]), 'r') as f:
                    url_count = len(f.readlines())
                matched_file = url_files[pinyin]
        
        # 尝试英文名匹配
        if url_count == 0 and english:
            english_lower = english.lower().replace(' ', '_')
            if english_lower in url_files:
                with open(os.path.join(url_dir, url_files[english_lower]), 'r') as f:
                    url_count = len(f.readlines())
                matched_file = url_files[english_lower]
        
        # 尝试日文名匹配
        if url_count == 0 and japanese:
            if japanese in url_files:
                with open(os.path.join(url_dir, url_files[japanese]), 'r') as f:
                    url_count = len(f.readlines())
                matched_file = url_files[japanese]
        
        total_urls += url_count
        
        if url_count > 0:
            matched_roles += 1
            role_stats.append({
                'chinese': chinese,
                'english': english,
                'url_count': url_count,
                'file': matched_file
            })
        else:
            unmatched_roles += 1
    
    # 按URL数量排序
    role_stats.sort(key=lambda x: x['url_count'], reverse=True)
    
    print(f'\n【一、总体统计】')
    print(f'  角色总数: {len(roles)} 个')
    print(f'  已匹配角色: {matched_roles} 个')
    print(f'  未匹配角色: {unmatched_roles} 个')
    print(f'  总URL数量: {total_urls:,} 个')
    print(f'  平均每角色: {total_urls // matched_roles} 个')
    
    print(f'\n【二、角色URL数量TOP20】')
    print('-' * 75)
    print(f'{"排名":<4} {"角色中文名":<12} {"英文名":<12} {"URL数量":>8} {"URL文件":<20}')
    print('-' * 75)
    for i, stat in enumerate(role_stats[:20], 1):
        print(f'{i:<4} {stat["chinese"][:11]:<12} {stat["english"][:11]:<12} {stat["url_count"]:>8} {stat["file"][:19]:<20}')
    
    print(f'\n【三、未匹配角色列表】')
    print('-' * 50)
    print(f'{"序号":<4} {"角色中文名":<12} {"英文名":<15}')
    print('-' * 50)
    idx = 1
    for role in roles:
        chinese = role['chinese']
        found = False
        
        # 检查是否匹配
        if chinese in pinyin_map:
            if pinyin_map[chinese] in url_files:
                found = True
        if not found and role['english']:
            if role['english'].lower().replace(' ', '_') in url_files:
                found = True
        if not found and role['japanese']:
            if role['japanese'] in url_files:
                found = True
        
        if not found:
            print(f'{idx:<4} {chinese[:11]:<12} {role["english"][:14]:<15}')
            idx += 1
    
    print(f'\n【四、URL数量分布区间】')
    print('-' * 40)
    ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 1000)]
    for r in ranges:
        cnt = sum(1 for s in role_stats if r[0] <= s['url_count'] < r[1])
        print(f'  {r[0]}-{r[1]}: {cnt} 个角色')

if __name__ == '__main__':
    analyze_url_by_role()
