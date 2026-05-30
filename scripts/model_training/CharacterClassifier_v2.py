#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
角色分类器 - 增强版
支持多数据源、本地缓存和手动数据录入
"""

import requests
import re
import json
import os


class CharacterClassifier:
    def __init__(self, cache_file="character_cache.json"):
        # 萌娘百科 API 地址
        self.moegirl_api_url = "https://zh.moegirl.org.cn/api.php"

        # 本地缓存文件
        self.cache_file = cache_file
        self.cache = self.load_cache()

        # 定义 NLP 关键词权重字典 (特征工程)
        self.keywords_weights = {
            "萝莉": 5.0,
            "合法萝莉": 5.0,
            "万年萝莉": 5.0,
            "幼女": 4.0,
            "体型娇小": 2.0,
            "娇小": 1.5,
            "小女孩": 2.0,
            "平胸": 1.0,
            "小学生": 2.0,
            "童颜": 1.5,
        }

        # 反向特征（降低该分类概率的特征）
        self.negative_keywords = {
            "御姐": -4.0,
            "人妻": -3.0,
            "巨乳": -3.0,
            "熟女": -4.0,
            "成年女性": -3.0,
        }

    def load_cache(self):
        """
        加载本地缓存
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载缓存失败: {e}")
                return {}
        return {}

    def save_cache(self):
        """
        保存本地缓存
        """
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存失败: {e}")

    def fetch_from_moegirl(self, character_name):
        """
        从萌娘百科获取角色数据
        """
        params = {
            "action": "query",
            "prop": "extracts|categories",
            "titles": character_name,
            "format": "json",
            "exintro": True,
            "explaintext": True,
            "cllimit": "max",
        }

        try:
            response = requests.get(self.moegirl_api_url, params=params, timeout=10)
            data = response.json()
            pages = data.get("query", {}).get("pages", {})

            for page_id, page_info in pages.items():
                if page_id == "-1":
                    return None
                return page_info
        except Exception as e:
            print(f"萌娘百科请求错误: {e}")
            return None

    def fetch_character_data(self, character_name):
        """
        获取角色数据，优先使用缓存
        """
        # 检查缓存
        if character_name in self.cache:
            print(f"从缓存获取角色: {character_name}")
            return self.cache[character_name]

        # 从萌娘百科获取
        page_info = self.fetch_from_moegirl(character_name)

        if page_info:
            # 保存到缓存
            self.cache[character_name] = page_info
            self.save_cache()
            print(f"从萌娘百科找到角色: {character_name}")
            return page_info

        print(f"未找到角色: {character_name}")
        return None

    def analyze_height(self, text):
        """
        分析文本中的身高信息
        """
        # 匹配身高模式：XXXcm、XXX厘米、身高XXX等
        height_patterns = [
            r"身高[：:]\s*(\d+\.?\d*)\s*[cm厘米]",
            r"(\d+\.?\d*)\s*[cm厘米]",
            r"(\d+\.?\d*)\s*米",
        ]

        for pattern in height_patterns:
            match = re.search(pattern, text)
            if match:
                height_str = match.group(1)
                try:
                    height = float(height_str)
                    # 如果是米，转换为厘米
                    if height < 3:
                        height = height * 100
                    return height
                except ValueError:
                    continue

        return None

    def extract_features(self, page_info):
        """
        从角色信息中提取特征
        """
        if not page_info:
            return []

        text = page_info.get("extract", "")
        categories = page_info.get("categories", [])

        # 提取分类信息
        category_names = []
        for cat in categories:
            title = cat.get("title", "")
            # 移除分类命名空间
            if title.startswith("分类:"):
                title = title[3:]
            category_names.append(title)

        # 合并文本和分类
        all_text = text + " " + " ".join(category_names)

        # 提取特征关键词
        matched_tags = []

        # 检查正向关键词
        for keyword, weight in self.keywords_weights.items():
            if keyword in all_text:
                matched_tags.append(keyword)

        # 检查反向关键词
        for keyword, weight in self.negative_keywords.items():
            if keyword in all_text:
                matched_tags.append(f"{keyword}(-)")

        # 分析身高
        height = self.analyze_height(all_text)
        if height:
            if height < 150:
                matched_tags.append("身高偏矮")
            elif height > 165:
                matched_tags.append("身高偏高")

        return matched_tags

    def calculate_score(self, matched_tags):
        """
        根据匹配的特征计算综合得分
        """
        score = 0.0

        for tag in matched_tags:
            # 检查是否是反向特征
            if tag.endswith("(-)"):
                keyword = tag[:-2]
                if keyword in self.negative_keywords:
                    score += self.negative_keywords[keyword]
            else:
                if tag in self.keywords_weights:
                    score += self.keywords_weights[tag]
                elif tag == "身高偏矮":
                    score += 1.5
                elif tag == "身高偏高":
                    score -= 1.0

        return score

    def classify(self, character_name):
        """
        对角色进行分类
        """
        print(f"正在分析角色: 【{character_name}】...")

        # 获取角色数据
        page_info = self.fetch_character_data(character_name)

        if not page_info:
            return f"未在数据库中找到该角色，或角色名不准确。", [], "未找到角色"

        # 提取特征
        matched_tags = self.extract_features(page_info)

        if not matched_tags:
            return "无明显特征", [], "无明显特征"

        # 计算综合得分
        score = self.calculate_score(matched_tags)

        print(f"提取到的特征: {matched_tags if matched_tags else '无明显特征'}")
        print(f"综合特征得分: {score}")

        # 根据得分进行分类
        if score >= 4.0:
            return f"✅ 判定结果：【{character_name}】 属于 萝莉 分类。", matched_tags, "萝莉"
        elif score > 0:
            return (
                f"❓ 判定结果：【{character_name}】 可能具有部分萝莉特征，但证据不足。",
                matched_tags,
                "可能具有部分萝莉特征",
            )
        else:
            return (
                f"❌ 判定结果：【{character_name}】 不属于 萝莉 分类。",
                matched_tags,
                "不属于萝莉",
            )

    def add_manual_data(self, character_name, features):
        """
        手动添加角色数据
        """
        # 创建模拟的页面信息
        page_info = {"extract": " ".join(features), "categories": []}

        # 保存到缓存
        self.cache[character_name] = page_info
        self.save_cache()
        print(f"已手动添加角色数据: {character_name}")

    def export_cache(self, output_file):
        """
        导出缓存数据
        """
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            print(f"缓存数据已导出到: {output_file}")
        except Exception as e:
            print(f"导出缓存失败: {e}")

    def import_cache(self, input_file):
        """
        导入缓存数据
        """
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.cache.update(data)
            self.save_cache()
            print(f"缓存数据已从 {input_file} 导入")
        except Exception as e:
            print(f"导入缓存失败: {e}")


if __name__ == "__main__":
    classifier = CharacterClassifier()

    # 测试分类
    test_characters = ["可莉", "纳西妲", "青雀", "古明地恋"]

    for name in test_characters:
        result, tags, category = classifier.classify(name)
        print(result)
        print("-" * 40)
