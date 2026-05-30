import os
import shutil


def unify_url_naming():
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

    # URL目录
    url_dir = "spider_image_system/data/img_url"
    new_url_dir = "spider_image_system/data/img_url_english"

    # 创建新目录
    os.makedirs(new_url_dir, exist_ok=True)

    print("=" * 70)
    print("          URL文件统一命名为英文")
    print("=" * 70)

    renamed_count = 0
    skipped_count = 0
    merged_count = 0

    # 建立角色到URL文件的映射
    role_files = {}

    for role in roles:
        chinese = role["chinese"]
        english = role["english"]

        if not english:
            skipped_count += 1
            print(f"⏭️ {chinese}: 无英文名，跳过")
            continue

        # 标准化英文名
        normalized_english = english.replace(" ", "_").replace("·", "_")

        # 收集该角色的所有URL文件
        files_to_merge = []

        # 拼音文件
        if chinese in pinyin_map:
            pinyin = pinyin_map[chinese]
            pinyin_file = f"{pinyin}_img.txt"
            pinyin_path = os.path.join(url_dir, pinyin_file)
            if os.path.exists(pinyin_path):
                files_to_merge.append(pinyin_path)

        # 英文名文件
        english_lower = english.lower().replace(" ", "_")
        english_file = f"{english_lower}_img.txt"
        english_path = os.path.join(url_dir, english_file)
        if os.path.exists(english_path) and english_path not in files_to_merge:
            files_to_merge.append(english_path)

        # 日文名文件
        if role["japanese"]:
            japanese_file = f'{role["japanese"]}_img.txt'
            japanese_path = os.path.join(url_dir, japanese_file)
            if os.path.exists(japanese_path) and japanese_path not in files_to_merge:
                files_to_merge.append(japanese_path)

        # 直接英文名大写文件
        english_upper_file = f"{english}_img.txt"
        english_upper_path = os.path.join(url_dir, english_upper_file)
        if os.path.exists(english_upper_path) and english_upper_path not in files_to_merge:
            files_to_merge.append(english_upper_path)

        if not files_to_merge:
            skipped_count += 1
            print(f"⏭️ {chinese} ({english}): 无URL文件，跳过")
            continue

        # 合并URL文件
        new_file_name = f"{normalized_english}_img.txt"
        new_file_path = os.path.join(new_url_dir, new_file_name)

        # 去重合并
        all_urls = set()
        for src_file in files_to_merge:
            with open(src_file, "r") as f:
                all_urls.update(f.read().splitlines())

        with open(new_file_path, "w") as f:
            for url in sorted(all_urls):
                f.write(url + "\n")

        if len(files_to_merge) > 1:
            merged_count += 1
            print(
                f"🔄 {chinese} ({english}): 合并 {len(files_to_merge)} 个文件 -> {new_file_name} ({len(all_urls)} URLs)"
            )
        else:
            renamed_count += 1
            print(
                f"✅ {chinese} ({english}): {files_to_merge[0].split('/')[-1]} -> {new_file_name} ({len(all_urls)} URLs)"
            )

    print("\n" + "=" * 70)
    print(f"重命名完成!")
    print(f"  - 重命名文件: {renamed_count} 个")
    print(f"  - 合并文件: {merged_count} 个")
    print(f"  - 跳过角色: {skipped_count} 个")
    print(f"  - 输出目录: {new_url_dir}")
    print("=" * 70)


if __name__ == "__main__":
    unify_url_naming()
