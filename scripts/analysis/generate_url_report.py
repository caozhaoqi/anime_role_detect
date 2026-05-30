import os


def generate_report():
    # 读取角色名单
    roles = []
    with open("auto_spider_img/loli-role.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # 格式: 1→阿洛娜 蔚蓝档案 Arona アロナ
                # 先去掉序号部分
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

    # 获取所有URL文件
    url_dir = "spider_image_system/data/img_url"
    url_files = [f for f in os.listdir(url_dir) if f.endswith("_img.txt")]
    url_file_names = [f.replace("_img.txt", "") for f in url_files]

    print("=" * 70)
    print("           角色URL文件统一优化报告")
    print("=" * 70)

    # 统计信息
    print("\n【一、基本统计】")
    print(f"  角色名单总数: {len(roles)} 个")
    print(f"  URL文件总数: {len(url_files)} 个")

    # 命名类型分析
    pinyin_count = sum(1 for f in url_file_names if f.islower() and any(c.isdigit() for c in f))
    english_count = sum(
        1 for f in url_file_names if f[0].isupper() and not any(c.isdigit() for c in f)
    )
    japanese_count = sum(1 for f in url_file_names if any("\u3040" <= c <= "\u30ff" for c in f))
    mixed_count = len(url_file_names) - pinyin_count - english_count - japanese_count

    print(f"\n  命名类型分布:")
    print(f"    ├── 拼音命名: {pinyin_count} 个")
    print(f"    ├── 英文命名: {english_count} 个")
    print(f"    ├── 日文命名: {japanese_count} 个")
    print(f"    └── 混合命名: {mixed_count} 个")

    # URL数量统计
    total_urls = 0
    max_urls = 0
    max_file = ""
    for f in url_files:
        with open(os.path.join(url_dir, f), "r") as file:
            count = len(file.readlines())
            total_urls += count
            if count > max_urls:
                max_urls = count
                max_file = f

    print(f"\n  URL内容统计:")
    print(f"    ├── 总URL数量: {total_urls:,} 个")
    print(f"    ├── 平均每文件: {total_urls // len(url_files)} 个")
    print(f"    └── 最大文件: {max_file} ({max_urls} URLs)")

    # 角色匹配分析
    print("\n【二、角色-URL文件匹配分析】")
    print(f"  正在分析 {len(roles)} 个角色...")

    matched = []
    unmatched = []

    # 构建拼音映射表
    pinyin_map = {
        "阿洛娜": "a1luo4na4",
        "纳西妲": "na4xi1da2",
        "黑塔": "hei1ta3",
        "符玄": "fu2xuan2",
        "瑶瑶": "yao2yao2",
        "迪奥娜": "di2ao4na4",
        "安可": "an1ke3",
        "晓美焰": "xiao3mei3yan4",
        "蕾姆": "lei2mu3",
        "拉姆": "la1mu3",
        "神乐": "shen1yue4",
        "阿尼亚": "a1ni4ya4",
        "白上吹雪": "bai2shang4chui1xue3",
        "布洛妮娅": "bu4luo4ni2ya4",
        "维普蕾": "wei2pu3lei3",
        "莉塔拉": "li4ta3la1",
        "月千夜": "yue4qian1ye4",
        "纳甘": "na4gan1",
        "寇尔芙": "kou4er3fu2",
        "克罗丽科": "ke4luo2li4ke1",
        "佩里缇亚": "pei4li3ti2ya4",
        "科谢尼娅": "ke1xie4ni2ya4",
        "夏克里": "sha1wu4",
        "芙兰": "fu2lan2",
        "菲米莉丝": "fei1mi3li4si1",
        "杏": "kan1",
        "伊瑟琳": "yi1se4lin2",
        "灶门祢豆子": "ni2dou4zi5",
        "雏鹤爱": "chu2he4ai4",
        "普拉娜": "pu3la1na4",
        "砂狼白子": "sha1lang2bai2zi3",
        "派蒙": "pai4meng2",
        "火花": "hua1huo3",
        "香风智乃": "xiang1feng1zhi4nai3",
        "希儿": "xi1er3",
        "铃兰": "ling2lan2",
        "白咲花": "bai2xiao4hua1",
        "星野日向": "xing1ye3ri4xiang4",
        "姬坂乃爱": "ji1ban3nai4ai4",
        "种村小依": "zhong3cun1xiao3yi1",
        "小之森夏音": "xiao3zhi1sen1xia4yin1",
        "夜叉神天衣": "ye4cha1shen2tian1yi1",
        "空银子": "kong1yin2zi3",
        "早濑优香": "zao3lai4you1xiang1",
        "一之濑明日奈": "yi1zhi1lai4ming2ri4nai4",
        "空崎日奈": "kong1qi2ri4nai4",
        "圣园未花": "sheng4yuan2wei4hua1",
        "小鸟游星野": "xiao3niao3you2xing1ye3",
        "四糸乃": "si4mi4nai3",
        "康娜": "kang1na4",
        "凯露": "kai3lu4",
        "伊莉雅": "yi1li4ya3",
        "忍野忍": "ren3ye3ren3",
        "小埋": "xiao3mai2",
        "纱雾": "sha1wu4",
        "猫宫又奈": "mao1gong1you4nai4",
        "德丽莎": "de2li4sha1",
        "可琳": "ke3lin2",
        "缇宝": "ti2bao3",
        "可莉": "ke3li4",
        "希格雯": "xi1ge2wen2",
        "蕾贝": "lei3bei4",
        "七七": "qi1qi1",
        "早柚": "zao3you4",
        "多莉": "duo1li4",
        "卡齐娜": "ka3qi2na4",
        "三月七": "san1yue4qi1",
        "花火": "hua1huo3",
        "银狼": "yin2lang2",
        "天童爱丽丝": "tian1tong2ai4li4si1",
        "早雾": "zao3wu4",
        "维里奈": "wei2li3nai4",
        "釉瑚": "you4hu2",
        "鹿目圆": "lu4mu4yuan2",
        "血小板": "xue4xiao3ban3",
        "克拉拉": "ke1la1la1",
    }

    for role in roles:
        matched_file = None
        matched_by = ""

        # 尝试英文名精确匹配
        if role["english"]:
            english_lower = role["english"].lower().replace(" ", "_")
            for url_name in url_file_names:
                if url_name.lower() == english_lower:
                    matched_file = url_name + "_img.txt"
                    matched_by = "英文名"
                    break

        # 尝试英文名前缀匹配
        if not matched_file and role["english"]:
            english_lower = role["english"].lower().replace(" ", "_")
            for url_name in url_file_names:
                if url_name.lower().startswith(english_lower.split("_")[0]):
                    matched_file = url_name + "_img.txt"
                    matched_by = "英文名前缀"
                    break

        # 尝试日文名匹配
        if not matched_file and role["japanese"]:
            for url_name in url_file_names:
                if url_name == role["japanese"]:
                    matched_file = url_name + "_img.txt"
                    matched_by = "日文名"
                    break

        # 尝试拼音匹配
        if not matched_file and role["chinese"] in pinyin_map:
            pinyin_name = pinyin_map[role["chinese"]]
            if pinyin_name + "_img.txt" in url_files:
                matched_file = pinyin_name + "_img.txt"
                matched_by = "拼音"

        if matched_file:
            with open(os.path.join(url_dir, matched_file), "r") as f:
                url_count = len(f.readlines())
            matched.append(
                (
                    role["chinese"],
                    role["english"],
                    role["japanese"],
                    matched_file,
                    url_count,
                    matched_by,
                )
            )
        else:
            unmatched.append((role["chinese"], role["english"], role["japanese"]))

    print(f"\n  ✓ 已匹配角色: {len(matched)} 个")
    print(f"  ✗ 未匹配角色: {len(unmatched)} 个")

    print("\n【三、已匹配角色详情】")
    print("-" * 75)
    print(f"{'角色中文名':<12} {'英文名':<15} {'匹配方式':<8} {'URL文件':<20} {'URL数量':>8}")
    print("-" * 75)
    for chn, eng, jpn, file, cnt, by in matched:
        print(f"{chn:<12} {eng[:14]:<15} {by:<8} {file:<20} {cnt:>8}")

    print("\n【四、未匹配角色列表】")
    print("-" * 55)
    print(f"{'序号':<4} {'角色中文名':<12} {'英文名':<15} {'日文名':<12}")
    print("-" * 55)
    for i, (chn, eng, jpn) in enumerate(unmatched, 1):
        print(f"{i:<4} {chn:<12} {eng[:14]:<15} {jpn[:11]:<12}")

    print("\n【五、优化建议】")
    print("=" * 70)
    print("  1. 统一命名规范:")
    print("     └── 建议使用英文名作为主标识符，如 Arona、Nahida、Herta")
    print("     └── 避免拼音编码问题，提高跨平台兼容性")
    print()
    print("  2. 清理重复文件:")
    print("     └── 存在多个同名角色的不同变体文件")
    print("     └── 如 Aris_img.txt 和 Aris wei4lan2dang4an4_img.txt")
    print()
    print("  3. URL质量过滤:")
    print("     └── 部分URL文件包含大量无效链接（SVG图标等）")
    print("     └── 建议下载前进行格式过滤")
    print()
    print("  4. 角色映射表:")
    print("     └── 创建标准化的角色名映射表")
    print("     └── 支持中文名、英文名、日文名互查")

    print("\n" + "=" * 70)
    print("                   报告结束")
    print("=" * 70)


if __name__ == "__main__":
    generate_report()
