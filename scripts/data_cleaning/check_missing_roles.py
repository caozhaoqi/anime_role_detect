#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pypinyin import lazy_pinyin, Style

# 角色列表文件
ROLE_FILE = "auto_spider_img/loli-role.txt"
# URL目录
URL_DIR = "spider_image_system/data/img_url_english"

# 读取角色列表
roles = []
with open(ROLE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(" ")
            chinese_name = parts[0]
            # 英文名是第3个字段（索引2）
            if len(parts) >= 3:
                en_name = parts[2]
            else:
                en_name = chinese_name
            roles.append({"chinese": chinese_name, "english": en_name})

print(f"总角色数: {len(roles)}")

# 获取已有的URL文件
existing_files = set()
if os.path.exists(URL_DIR):
    for filename in os.listdir(URL_DIR):
        if filename.endswith("_img.txt"):
            en_name = filename.replace("_img.txt", "")
            existing_files.add(en_name)

print(f"已有URL文件: {len(existing_files)} 个")

# 找出缺少的角色
missing_roles = []
for role in roles:
    if role["english"] not in existing_files:
        missing_roles.append(role)

print(f"\n缺少URL的角色: {len(missing_roles)} 个")
print("=" * 60)
for i, role in enumerate(missing_roles, 1):
    print(f"{i:2d}. {role['chinese']} ({role['english']})")

# 保存缺少的角色到文件
if missing_roles:
    with open("data/missing_roles.txt", "w", encoding="utf-8") as f:
        for role in missing_roles:
            f.write(f"{role['chinese']} {role['english']}\n")
    print(f"\n已保存到 data/missing_roles.txt")
