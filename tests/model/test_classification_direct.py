#!/usr/bin/env python3
"""

"""

import sys
import os
import numpy as np
from pathlib import Path

# Python
project_root = Path(__file__).parent.parent.parent.parent  # tests/manual/model_testing/classification -> 

from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification


def test_classification():
    """"""
    print("=" * 60)
    print("")
    print("=" * 60)

    # 
    preprocessor = Preprocessing()
    extractor = FeatureExtraction()
    classifier = Classification(index_path="role_index", threshold=0.4)

    # 
    test_images = [
        ("data/train//_000.jpg", ""),
        ("data/train//_000.jpg", ""),
        ("data/train//_000.jpg", ""),
    ]

    for image_path, expected_role in test_images:
        full_path = project_root / image_path
        print(f"\n: {full_path}")
        print(f": {expected_role}")

        try:
            # 
            normalized_img, boxes = preprocessor.process(str(full_path))
            print(f" {len(boxes)} ")

            # 
            feature = extractor.extract_features(normalized_img)
            print(f": {feature.shape}")

            # 
            role, similarity = classifier.classify(feature)
            print(f": ={role}, ={similarity:.4f}")

        except Exception as e:
            print(f": {e}")

    print("\n" + "=" * 60)
    print("")
    print("=" * 60)


if __name__ == "__main__":
    test_classification()
