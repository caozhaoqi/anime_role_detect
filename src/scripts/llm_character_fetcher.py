#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用大模型获取角色名脚本
参考CharacterClassifier_v2.py，结合大模型API获取角色信息
"""

import os
import json
import requests
from datetime import datetime

class LLMCharacterFetcher:
    def __init__(self, cache_file="llm_character_cache.json"):
        # 本地缓存文件
        self.cache_file = cache_file
        self.cache = self.load_cache()
        
        # 大模型API配置（示例）
        self.llm_api_url = "https://api.example.com/chat/completions"  # 替换为真实API地址
        self.api_key = "your_api_key"  # 替换为真实API密钥
    
    def load_cache(self):
        """
        加载本地缓存
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
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
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存失败: {e}")
    
    def call_llm_api(self, prompt):
        """
        调用大模型API获取角色信息
        """
        # 这里使用一个模拟的响应
        # 实际使用时需要替换为真实的API调用
        
        # 模拟响应 - 原神角色
        if "原神" in prompt:
            return {
                "characters": [
                    "琴", "安柏", "丽莎", "芭芭拉", "可莉", "诺艾尔", "菲谢尔", "砂糖", "莫娜", "迪奥娜",
                    "罗莎莉亚", "优菈", "闲云", "瑶瑶", "夜兰", "申鹤", "云堇", "北斗", "凝光", "香菱",
                    "刻晴", "七七", "辛焱", "甘雨", "胡桃", "烟绯", "神里绫华", "宵宫", "早柚", "雷电将军",
                    "八重神子", "九条裟罗", "珊瑚宫心海", "娜维娅", "芙宁娜", "千织", "久岐忍", "珐露珊", "莱依拉", "妮露",
                    "坎蒂丝", "多莉", "柯莱", "绮良良", "纳西妲", "迪希雅", "迪娜泽黛", "派蒙", "夏沃蕾", "夏洛蒂",
                    "琳妮特", "希格雯", "克洛琳德", "归终", "萍姥姥", "荧"
                ],
                "source": "LLM"
            }
        
        # 模拟响应 - 星穹铁道角色
        elif "星穹铁道" in prompt:
            return {
                "characters": [
                    "艾丝妲", "三月七", "希露瓦", "黑塔", "银狼", "希儿", "卡芙卡", "素裳", "姬子", "布洛妮娅",
                    "克拉拉", "佩拉", "虎克", "黑天鹅", "花火", "阮梅", "娜塔莎", "寒鸦", "镜流", "雪衣",
                    "黄泉", "符玄", "白露", "霍霍", "玲妮", "青雀", "停云", "托帕", "驭空"
                ],
                "source": "LLM"
            }
        
        # 模拟响应 - 崩坏3角色
        elif "崩坏3" in prompt:
            return {
                "characters": [
                    "布洛妮娅", "符华", "希儿", "格蕾修", "丽塔", "爱莉希雅", "琪亚娜", "雷电芽衣", "识之律者", "雷之律者",
                    "空之律者", "死生之律者", "薪炎之律者", "始源之律者", "人之律者", "无量塔姬子", "八重樱", "德丽莎·阿波卡利斯", "卡莲·卡斯兰娜", "丽塔·洛丝薇瑟",
                    "希儿·芙乐艾", "萝莎莉娅·阿琳", "莉莉娅·阿琳", "时雨绮罗", "普罗米修斯", "米丝忒琳·沙尼亚特", "苏莎娜", "爱衣·休伯利安Λ", "李素裳", "维尔薇",
                    "梅比乌斯", "帕朵菲莉丝", "阿波尼亚", "伊甸"
                ],
                "source": "LLM"
            }
        
        # 默认响应
        else:
            return {
                "characters": [],
                "source": "LLM"
            }
    
    def fetch_characters(self, game_name):
        """
        获取指定游戏的角色列表
        """
        # 检查缓存
        if game_name in self.cache:
            print(f"从缓存获取 {game_name} 角色列表")
            return self.cache[game_name]
        
        # 构建prompt
        prompt = f"请列出{game_name}中的所有角色名称，只返回角色名列表，不要包含其他内容。"
        
        # 调用大模型API
        print(f"向大模型询问 {game_name} 的角色名")
        response = self.call_llm_api(prompt)
        
        if response and response.get("characters"):
            # 保存到缓存
            self.cache[game_name] = response
            self.save_cache()
            print(f"从大模型获取到 {len(response['characters'])} 个 {game_name} 角色")
            return response
        
        print(f"未从大模型获取到 {game_name} 的角色信息")
        return {"characters": [], "source": "LLM"}
    
    def update_character_file(self, game_name, file_path):
        """
        更新角色文件
        """
        # 获取角色列表
        response = self.fetch_characters(game_name)
        characters = response.get("characters", [])
        
        if not characters:
            print(f"未获取到 {game_name} 的角色列表，跳过更新")
            return False
        
        # 读取现有角色
        existing_roles = set()
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        role = line.strip()
                        if role:
                            existing_roles.add(role)
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
        
        # 过滤出新角色
        new_roles = [role for role in characters if role not in existing_roles]
        
        if new_roles:
            # 追加新角色
            try:
                with open(file_path, 'a', encoding='utf-8') as f:
                    for role in new_roles:
                        f.write(f"{role}\n")
                print(f"已为 {game_name} 添加 {len(new_roles)} 个新角色")
                return True
            except Exception as e:
                print(f"写入文件 {file_path} 时出错: {e}")
                return False
        else:
            print(f"{game_name} 无新角色更新")
            return False
    
    def export_cache(self, output_file):
        """
        导出缓存数据
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            print(f"缓存数据已导出到: {output_file}")
        except Exception as e:
            print(f"导出缓存失败: {e}")
    
    def import_cache(self, input_file):
        """
        导入缓存数据
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.cache.update(data)
            self.save_cache()
            print(f"缓存数据已从 {input_file} 导入")
        except Exception as e:
            print(f"导入缓存失败: {e}")

def main():
    """
    主函数
    """
    # 初始化大模型角色获取器
    fetcher = LLMCharacterFetcher()
    
    # 游戏配置
    game_configs = [
        {
            "name": "原神",
            "file_path": "auto_spider_img/txt/1_genshin_chinese_spider_img_keyword.txt"
        },
        {
            "name": "星穹铁道",
            "file_path": "auto_spider_img/txt/3_star_rail_chinese_spider_img_keyword.txt"
        },
        {
            "name": "崩坏3",
            "file_path": "auto_spider_img/txt/6_honkai3_chinese_spider_img_keyword.txt"
        }
    ]
    
    # 处理每个游戏
    for config in game_configs:
        game_name = config["name"]
        file_path = config["file_path"]
        
        print(f"\n处理游戏: {game_name}")
        print(f"更新文件: {file_path}")
        
        # 更新角色文件
        success = fetcher.update_character_file(game_name, file_path)
        
        if success:
            print(f"{game_name} 角色文件更新成功")
        else:
            print(f"{game_name} 角色文件更新失败")
    
    print("\n角色获取完成！")

if __name__ == "__main__":
    main()