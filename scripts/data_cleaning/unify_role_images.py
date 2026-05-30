import os
import shutil


def unify_role_images():
    # 读取角色名单
    roles = []
    with open("auto_spider_img/loli-role.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                if "→" in line:
                    content = line.split("→")[1].strip()
                else:
                    content = line
                parts = content.split(" ")
                roles.append(
                    {
                        "chinese": parts[0],
                        "source": parts[1],
                        "english": parts[2] if len(parts) > 2 else "",
                        "japanese": parts[3] if len(parts) > 3 else "",
                    }
                )

    # 拼音映射表
    pinyin_map = {
        "阿洛娜": "a1luo4na4",
        "普拉娜": "pu3la1na4",
        "砂狼白子": "sha1lang2bai2zi3",
        "纳西妲": "na4xi1da2",
        "缇宝": "ti2bao3",
        "可莉": "ke3li4",
        "迪奥娜": "di2ao4na4",
        "瑶瑶": "yao2yao2",
        "希格雯": "xi1ge2wen2",
        "蕾贝": "lei3bei4",
        "黑塔": "hei1ta3",
        "符玄": "fu2xuan2",
        "七七": "qi1qi1",
        "早柚": "zao3you4",
        "多莉": "duo1li4",
        "派蒙": "pai4meng2",
        "卡齐娜": "ka3qi2na4",
        "三月七": "san1yue4qi1",
        "花火": "hua1huo3",
        "火花": "hua1huo3",
        "银狼": "yin2lang2",
        "天童爱丽丝": "tian1tong2ai4li4si1",
        "早雾": "zao3wu4",
        "维里奈": "wei2li3nai4",
        "安可": "an1ke3",
        "釉瑚": "you4hu2",
        "鹿目圆": "lu4mu4yuan2",
        "晓美焰": "xiao3mei3yan4",
        "血小板": "xue4xiao3ban3",
        "雷姆": "lei2mu3",
        "拉姆": "la1mu3",
        "康娜": "kang1na4",
        "四糸乃": "si4mi4nai3",
        "凯露": "kai3lu4",
        "伊莉雅": "yi1li4ya3",
        "忍野忍": "ren3ye3ren3",
        "香风智乃": "xiang1feng1zhi4nai3",
        "小埋": "xiao3mai2",
        "纱雾": "sha1wu4",
        "猫宫又奈": "mao1gong1you4nai4",
        "德丽莎": "de2li4sha1",
        "布洛妮娅": "bu4luo4ni2ya4",
        "可琳": "ke3lin2",
        "神乐": "shen1yue4",
        "白上吹雪": "bai2shang4chui1xue3",
        "月千夜": "yue4qian1ye4",
        "莉塔拉": "li4ta3la1",
        "维普蕾": "wei2pu3lei3",
        "夏克里": "sha1wu4",
        "纳甘": "na4gan1",
        "科谢尼娅": "ke1xie4ni2ya4",
        "寇尔芙": "kou4er3fu2",
        "克罗丽科": "ke4luo2li4ke1",
        "佩里缇亚": "pei4li3ti2ya4",
        "阿尼亚": "a1ni4ya4",
        "洛茜": "luo4qian4",
        "灶门祢豆子": "ni2dou4zi5",
        "希儿": "xi1er3",
        "杏": "kan1",
        "伊瑟琳": "yi1se4lin2",
        "芙兰": "fu2lan2",
        "菲米莉丝": "fei1mi3li4si1",
        "克拉拉": "ke1la1la1",
        "铃兰": "ling2lan2",
        "白咲花": "bai2xiao4hua1",
        "星野日向": "xing1ye3ri4xiang4",
        "姬坂乃爱": "ji1ban3nai4ai4",
        "种村小依": "zhong3cun1xiao3yi1",
        "小之森夏音": "xiao3zhi1sen1xia4yin1",
        "雏鹤爱": "chu2he4ai4",
        "夜叉神天衣": "ye4cha1shen2tian1yi1",
        "空银子": "kong1yin2zi3",
        "早濑优香": "zao3lai4you1xiang1",
        "一之濑明日奈": "yi1zhi1lai4ming2ri4nai4",
        "空崎日奈": "kong1qi2ri4nai4",
        "圣园未花": "sheng4yuan2wei4hua1",
        "小鸟游星野": "xiao3niao3you2xing1ye3",
    }

    # 源目录和目标目录
    source_dir = "data/reorganized_dataset"
    target_dir = "data/unified_dataset"

    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 70)
    print("          角色图片统一脚本")
    print("=" * 70)

    total_roles = len(roles)
    processed_count = 0
    skipped_count = 0

    for role in roles:
        chinese_name = role["chinese"]
        english_name = role["english"]
        japanese_name = role["japanese"]

        # 获取可能的目录名
        possible_dirs = []

        # 拼音目录
        if chinese_name in pinyin_map:
            possible_dirs.append(pinyin_map[chinese_name])

        # 英文名目录（可能有空格或下划线）
        if english_name:
            possible_dirs.append(english_name.replace(" ", "_"))
            possible_dirs.append(english_name.replace(" ", ""))

        # 日文名目录
        if japanese_name:
            possible_dirs.append(japanese_name)

        # 去重
        possible_dirs = list(set(possible_dirs))

        # 查找实际存在的目录
        existing_dirs = []
        for dir_name in possible_dirs:
            dir_path = os.path.join(source_dir, dir_name)
            if os.path.isdir(dir_path):
                existing_dirs.append(dir_path)

        if not existing_dirs:
            skipped_count += 1
            print(f"⏭️ [{processed_count+1}/{total_roles}] {chinese_name}: 无图片目录，跳过")
            continue

        # 创建统一目录（使用中文名）
        unified_role_dir = os.path.join(target_dir, chinese_name)
        os.makedirs(unified_role_dir, exist_ok=True)

        # 复制图片到统一目录
        total_images = 0
        copied_count = 0
        for src_dir in existing_dirs:
            for filename in os.listdir(src_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    src_path = os.path.join(src_dir, filename)
                    # 避免重复文件名
                    base_name, ext = os.path.splitext(filename)
                    dest_path = os.path.join(unified_role_dir, filename)
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(unified_role_dir, f"{base_name}_{counter}{ext}")
                        counter += 1
                    shutil.copy2(src_path, dest_path)
                    copied_count += 1
                    total_images += 1

        processed_count += 1
        print(
            f"✅ [{processed_count}/{total_roles}] {chinese_name}: 合并 {len(existing_dirs)} 个目录，共 {copied_count} 张图片"
        )

    print("\n" + "=" * 70)
    print(f"统一完成!")
    print(f"  - 处理角色: {processed_count} 个")
    print(f"  - 跳过角色: {skipped_count} 个")
    print(f"  - 输出目录: {target_dir}")
    print("=" * 70)


if __name__ == "__main__":
    unify_role_images()
