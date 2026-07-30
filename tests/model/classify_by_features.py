#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import os


def classify_by_features(input_file):
    """
    

    Args:
        input_file: 
    """
    # 
    feature_categories = {
        "": [],
        "": [],
        "": [],
        "": [],
        "": [],
        "": [],
        "": [],
    }

    # 
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f" {len(lines)} ")

    # 
    current_category = None
    for i, line in enumerate(lines):
        line = line.strip()

        # 
        print(f" {i+1} : {line}")

        # 
        if not line or line.startswith(":") or line.startswith(":"):
            continue

        # 
        if line.endswith(":"):
            current_category = line[:-1]
            print(f": {current_category}")
            continue

        # 
        if current_category and " - : " in line:
            # 
            parts = line.split(" - : ")
            if len(parts) == 2:
                character_name = parts[0].strip()
                features_str = parts[1]

                # 
                if features_str == "":
                    features = []
                else:
                    #  [] 
                    features_str = features_str.strip("[]")
                    # 
                    features = []
                    # 
                    if "', '" in features_str:
                        for f in features_str.split("', '"):
                            feature = f.strip().strip("'").strip('"')
                            if feature:
                                features.append(feature)
                    else:
                        # 
                        feature = features_str.strip().strip("'").strip('"')
                        if feature:
                            features.append(feature)

                # 
                print(f": {character_name}, : {features}")

                # 
                for feature in features:
                    # 
                    if feature.endswith("(-)"):
                        feature_name = feature[:-3]
                        if feature_name in feature_categories:
                            feature_categories[feature_name].append(character_name)
                            print(f" {feature_name}: {character_name}")
                    else:
                        if feature in feature_categories:
                            feature_categories[feature].append(character_name)
                            print(f" {feature}: {character_name}")
                        # 
                        for category in feature_categories:
                            if category in feature:
                                feature_categories[category].append(character_name)
                                print(f" {category}: {character_name}")

    # 
    output_dir = os.path.dirname(input_file)
    feature_output_dir = os.path.join(output_dir, "features")
    if not os.path.exists(feature_output_dir):
        os.makedirs(feature_output_dir)

    # 
    with open(
        os.path.join(feature_output_dir, "feature_classification_results.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(":\n")
        for feature, characters in feature_categories.items():
            f.write(f"\n{feature}: {len(characters)} \n")
            for character in characters:
                f.write(f"  {character}\n")

    # 
    for feature, characters in feature_categories.items():
        if characters:
            with open(
                os.path.join(feature_output_dir, f"{feature}.txt"), "w", encoding="utf-8"
            ) as f:
                for character in characters:
                    f.write(f"{character}\n")

    print(f" {feature_output_dir} ")

    # 
    print("\n:")
    for feature, characters in feature_categories.items():
        print(f"{feature}: {len(characters)} ")


if __name__ == "__main__":
    input_file = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/lolis/classified/classification_results.txt"
    print(f": {input_file}")
    print(f": {os.path.exists(input_file)}")
    if os.path.exists(input_file):
        print(f": {os.path.getsize(input_file)} ")
    classify_by_features(input_file)
