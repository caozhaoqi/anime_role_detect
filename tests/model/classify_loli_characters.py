import sys
import os

#  CharacterClassifier Python
_script_dir = os.path.dirname(os.path.abspath(__file__))

from CharacterClassifier import CharacterClassifier


def classify_loli_characters():
    """
    loli_characters.txtCharacterClassifier
    """
    # 
    characters_file = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/lolis/loli_characters.txt"

    try:
        with open(characters_file, "r", encoding="utf-8") as f:
            characters = [line.strip() for line in f if line.strip()]

        print(f" {len(characters)} ")
        print("=" * 80)

        # 
        classifier = CharacterClassifier()

        # 
        results = {"": 0, "": 0, "": 0, "": 0}

        # 
        for character in characters:
            result = classifier.classify(character)
            print(result)
            print("-" * 40)

            # 
            if " " in result:
                results[""] += 1
            elif "" in result:
                results[""] += 1
            elif " " in result:
                results[""] += 1
            elif "" in result:
                results[""] += 1

        # 
        print("=" * 80)
        print(":")
        for key, value in results.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f": {e}")


if __name__ == "__main__":
    classify_loli_characters()
