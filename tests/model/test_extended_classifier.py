#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CharacterClassifier
"""

import sys
import os

#  CharacterClassifier Python
_script_dir = os.path.dirname(os.path.abspath(__file__))

from CharacterClassifier import CharacterClassifier


def test_extended_classifier():
    """
    
    """
    # 
    classifier = CharacterClassifier()

    # 
    test_characters = [
        "Yoimiya",  # 
        "Klee",  # 
        "Nahida",  # 
        "Raiden Shogun",  # 
        "Yae Miko",  # 
        "Kokomi",  # 
        "Furina",  # 
        "Navia",  # 
        "Clara",  # 
        "Seele",  # 
        "Bronya",  # 
        "Kafka",  # 
        "Himeko",  # 
        "Silver Wolf",  # 
        "Sparkle",  # 
        "Black Swan",  # 
        "Acheron",  # 
        "Firefly",  # 
        "Robin",  # 
    ]

    print("=" * 80)
    print("CharacterClassifier")
    print("=" * 80)

    # 
    results = {"": 0, "": 0, "": 0, "": 0}

    for name in test_characters:
        result, tags, category = classifier.classify(name)
        print(result)
        print("-" * 40)

        # 
        results[category] += 1

    # 
    print("=" * 80)
    print(":")
    for key, value in results.items():
        print(f"{key}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    test_extended_classifier()
