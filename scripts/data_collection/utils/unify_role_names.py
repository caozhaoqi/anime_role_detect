#!/usr/bin/env python3
"""
统一角色名称脚本
- 实现汉字拼音转化后的与汉字的角色名一致
- 更新数据库中的display_name字段
- 确保系统中角色名称的一致性
"""

import os
import json
import sqlite3
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="unify_role_names.log",
    filemode="a",
)
logger = logging.getLogger(__name__)

# 全局配置
GLOBAL_CONFIG = {
    "database_file": "../../data/role_images.db",
    "annotation_dir": "../../data/annotations",
    "role_images_dir": "../../data/role_images",
}

# 拼音到汉字的映射表（手动维护）
PINYIN_TO_CHINESE = {
    "a1luo2na4": "阿罗娜",
    "a1lu4": "阿鲁",
    "an1ka3xi1ya3": "安卡西亚",
    "dong1you1zi": "东云梓",
    "hua1yin1": "华音",
    "jia1dai4zi": "加代子",
    "jing4hua2": "京华",
    "kan1": "坎",
    "lei3bei4": "雷贝",
    "li3shi4": "丽诗",
    "li4li4ya4·a1lin2": "莉莉亚·艾琳",
    "ling2yin1": "铃音",
    "mu4yue4": "暮月",
    "pu3la1na4": "普拉娜",
    "qian1xia4": "千夏",
    "qing1que4": "青雀",
    "qiu1nai3": "秋奈",
    "ren3": "伦",
    "shen1yue4": "神乐",
    "yi1zhi1": "一只",
    "xiao3xia4": "小夏",
    "xing1ye3": "星野",
    "ya4zi": "亚子",
    "ri4nai4": "日奈",
    "wu4yu3mo2li3sha1": "吴雨茉莉莎",
    "di2ao4na4": "迪奥娜",
    "fei1xie4er3": "菲谢尔",
    "gu3ming2di4lian4": "古明地恋",
    "hei1ta3": "黑塔",
    "ke3li2": "可莉",
    "ke3lin2_wei1ke4si1": "科林·威克斯",
    "cong2yu3": "聪语",
    "fu2xuan2": "符玄",
}


def get_chinese_name_from_annotations():
    """从标注文件中提取汉字角色名"""
    logger.info("从标注文件中提取汉字角色名")

    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotation_dir = os.path.join(script_dir, GLOBAL_CONFIG["annotation_dir"])

    pinyin_to_chinese = {}

    # 遍历角色目录
    for role_name in os.listdir(annotation_dir):
        role_dir = os.path.join(annotation_dir, role_name)
        if not os.path.isdir(role_dir):
            continue

        # 遍历标注文件
        for file_name in os.listdir(role_dir):
            if not file_name.endswith(".json"):
                continue

            annotation_file = os.path.join(role_dir, file_name)
            try:
                with open(annotation_file, "r", encoding="utf-8") as f:
                    annotation_data = json.load(f)

                # 提取汉字名称
                chinese_name = annotation_data.get("name")
                if chinese_name and chinese_name != role_name:
                    pinyin_to_chinese[role_name] = chinese_name
                    logger.info(f"从标注文件中提取: {role_name} -> {chinese_name}")
                    # 找到一个就可以了，不需要遍历所有文件
                    break
            except Exception as e:
                logger.warning(f"读取标注文件失败: {annotation_file} - {str(e)}")

    return pinyin_to_chinese


def update_database_display_names():
    """更新数据库中的display_name字段"""
    logger.info("更新数据库中的display_name字段")

    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_file = os.path.join(script_dir, GLOBAL_CONFIG["database_file"])

    # 连接数据库
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # 从标注文件中提取汉字名称
    annotations_mapping = get_chinese_name_from_annotations()

    # 合并手动映射和从标注文件中提取的映射
    combined_mapping = {**PINYIN_TO_CHINESE, **annotations_mapping}

    # 更新数据库
    updated_count = 0
    for pinyin_name, chinese_name in combined_mapping.items():
        try:
            cursor.execute(
                """
            UPDATE roles SET display_name = ? WHERE name = ?
            """,
                (chinese_name, pinyin_name),
            )

            if cursor.rowcount > 0:
                updated_count += 1
                logger.info(f"更新角色显示名称: {pinyin_name} -> {chinese_name}")
        except Exception as e:
            logger.warning(f"更新数据库失败: {pinyin_name} - {str(e)}")

    # 提交更改
    conn.commit()
    conn.close()

    logger.info(f"更新了 {updated_count} 个角色的显示名称")
    return updated_count


def create_alias_system():
    """创建角色别名系统，支持通过拼音或汉字查询角色"""
    logger.info("创建角色别名系统")

    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_file = os.path.join(script_dir, GLOBAL_CONFIG["database_file"])

    # 连接数据库
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # 检查是否存在alias表
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS role_aliases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role_id INTEGER,
        alias TEXT UNIQUE,
        type TEXT,
        FOREIGN KEY (role_id) REFERENCES roles(id)
    )
    """
    )

    # 从标注文件中提取汉字名称
    annotations_mapping = get_chinese_name_from_annotations()

    # 合并手动映射和从标注文件中提取的映射
    combined_mapping = {**PINYIN_TO_CHINESE, **annotations_mapping}

    # 为每个角色创建别名
    added_aliases = 0
    for pinyin_name, chinese_name in combined_mapping.items():
        # 获取角色ID
        cursor.execute("SELECT id FROM roles WHERE name = ?", (pinyin_name,))
        role_result = cursor.fetchone()
        if not role_result:
            logger.warning(f"角色 {pinyin_name} 在数据库中不存在")
            continue
        role_id = role_result[0]

        # 添加拼音别名
        try:
            cursor.execute(
                """
            INSERT OR IGNORE INTO role_aliases (role_id, alias, type)
            VALUES (?, ?, ?)
            """,
                (role_id, pinyin_name, "pinyin"),
            )
            added_aliases += cursor.rowcount
        except Exception as e:
            logger.warning(f"添加拼音别名失败: {pinyin_name} - {str(e)}")

        # 添加汉字别名
        try:
            cursor.execute(
                """
            INSERT OR IGNORE INTO role_aliases (role_id, alias, type)
            VALUES (?, ?, ?)
            """,
                (role_id, chinese_name, "chinese"),
            )
            added_aliases += cursor.rowcount
        except Exception as e:
            logger.warning(f"添加汉字别名失败: {chinese_name} - {str(e)}")

    # 提交更改
    conn.commit()
    conn.close()

    logger.info(f"添加了 {added_aliases} 个角色别名")
    return added_aliases


def update_annotations_display_name():
    """更新标注文件中的显示名称"""
    logger.info("更新标注文件中的显示名称")

    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotation_dir = os.path.join(script_dir, GLOBAL_CONFIG["annotation_dir"])

    # 从标注文件中提取汉字名称
    annotations_mapping = get_chinese_name_from_annotations()

    # 合并手动映射和从标注文件中提取的映射
    combined_mapping = {**PINYIN_TO_CHINESE, **annotations_mapping}

    updated_count = 0
    # 遍历角色目录
    for role_name in os.listdir(annotation_dir):
        role_dir = os.path.join(annotation_dir, role_name)
        if not os.path.isdir(role_dir):
            continue

        # 获取汉字名称
        chinese_name = combined_mapping.get(role_name)
        if not chinese_name:
            logger.warning(f"角色 {role_name} 没有对应的汉字名称")
            continue

        # 遍历标注文件
        for file_name in os.listdir(role_dir):
            if not file_name.endswith(".json"):
                continue

            annotation_file = os.path.join(role_dir, file_name)
            try:
                with open(annotation_file, "r", encoding="utf-8") as f:
                    annotation_data = json.load(f)

                # 更新名称
                if annotation_data.get("name") != chinese_name:
                    annotation_data["name"] = chinese_name

                    with open(annotation_file, "w", encoding="utf-8") as f:
                        json.dump(annotation_data, f, indent=2, ensure_ascii=False)

                    updated_count += 1
                    logger.info(f"更新标注文件: {file_name} - {chinese_name}")
            except Exception as e:
                logger.warning(f"更新标注文件失败: {annotation_file} - {str(e)}")

    logger.info(f"更新了 {updated_count} 个标注文件")
    return updated_count


def main():
    """主函数"""
    logger.info("============================================================")
    logger.info("开始统一角色名称")
    logger.info("============================================================")

    # 更新数据库中的display_name字段
    update_database_display_names()

    # 创建角色别名系统
    create_alias_system()

    # 更新标注文件中的显示名称
    update_annotations_display_name()

    logger.info("\n============================================================")
    logger.info("角色名称统一完成")
    logger.info("============================================================")


if __name__ == "__main__":
    main()
