import sys
import os

# 添加脚本所在目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CharacterClassifier import CharacterClassifier

def classify_loli_characters():
    """
    读取loli_characters.txt文件中的角色列表，使用CharacterClassifier进行分类
    """
    # 读取角色列表文件
    characters_file = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/lolis/loli_characters.txt"
    
    try:
        with open(characters_file, 'r', encoding='utf-8') as f:
            characters = [line.strip() for line in f if line.strip()]
        
        print(f"共读取到 {len(characters)} 个角色")
        print("=" * 80)
        
        # 初始化分类器
        classifier = CharacterClassifier()
        
        # 分类结果统计
        results = {
            "萝莉": 0,
            "可能具有部分萝莉特征": 0,
            "不属于萝莉": 0,
            "未找到角色": 0
        }
        
        # 对每个角色进行分类
        for character in characters:
            result = classifier.classify(character)
            print(result)
            print("-" * 40)
            
            # 统计结果
            if "属于 萝莉" in result:
                results["萝莉"] += 1
            elif "可能具有部分萝莉特征" in result:
                results["可能具有部分萝莉特征"] += 1
            elif "不属于 萝莉" in result:
                results["不属于萝莉"] += 1
            elif "未在数据库中找到该角色" in result:
                results["未找到角色"] += 1
        
        # 输出统计结果
        print("=" * 80)
        print("分类结果统计:")
        for key, value in results.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    classify_loli_characters()
