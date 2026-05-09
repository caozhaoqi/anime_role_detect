import os

LOW_DATA_ROLES = {
    'fu2li4xi1ya4': '芙丽希娅',    # 目标30张
    'ai4li4er3': '爱丽儿',          # 目标30张
    'yue4qian1ye4': '月千夜',      # 目标30张
    'zao3wu4': '早雾',              # 目标30张
    'xiao3shan3': '小闪',          # 目标30张
    'fu2lan2': '芙兰',              # 目标30张
    'lei2mu3': '雷姆',              # 目标30张
    'a1ni4ya4': '阿尼亚',           # 目标30张
}

SOURCE_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
TARGET_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
MIN_IMAGES = 30

def count_images(dir_path):
    """统计目录中图片数量"""
    if not os.path.exists(dir_path):
        return 0
    count = 0
    for f in os.listdir(dir_path):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            count += 1
    return count

def check_available_supplement():
    print("="*60)
    print("检查 organized_images 目录是否可补充数据")
    print("="*60)
    print()

    can_supplement = False
    results = []

    for pinyin, role_name in LOW_DATA_ROLES.items():
        source_count = count_images(os.path.join(SOURCE_DIR, pinyin))
        target_count = count_images(os.path.join(TARGET_DIR, pinyin))
        
        available = source_count - target_count
        needed = MIN_IMAGES - target_count
        can_add = min(available, needed)
        
        results.append({
            'name': role_name,
            'pinyin': pinyin,
            'source_total': source_count,
            'target_current': target_count,
            'available_in_source': available,
            'still_needed': needed,
            'can_add': can_add
        })
        
        if can_add > 0:
            can_supplement = True

    print(f"{'角色':<10} {'拼音':<15} {'源总量':<8} {'目标当前':<8} {'源剩余':<8} {'仍需':<6} {'可补充':<6}")
    print("-" * 80)
    
    for r in results:
        status = "✓" if r['can_add'] > 0 else "-"
        print(f"{r['name']:<10} {r['pinyin']:<15} {r['source_total']:<8} {r['target_current']:<8} {r['available_in_source']:<8} {r['still_needed']:<6} {r['can_add']:<6} {status}")

    print()
    if can_supplement:
        total_can_add = sum(r['can_add'] for r in results)
        print(f"✓ 可以从 organized_images 补充 {total_can_add} 张图片")
    else:
        print("✗ organized_images 目录中已没有可补充的数据")
        
    print("\n提示: 已从 trash_nsfw 目录解除R18限制并补充过数据")
    print("如需更多数据，可能需要重新爬取或从其他来源获取")

    return can_supplement

if __name__ == '__main__':
    check_available_supplement()