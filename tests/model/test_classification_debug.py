#!/usr/bin/env python3
"""
unknown
"""

import sys
import os
import numpy as np
from PIL import Image
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent  # tests/manual/model_testing/classification -> 

from core.feature_extraction.feature_extraction import FeatureExtraction
from core.classification.classification import Classification
from core.logging.global_logger import get_logger

logger = get_logger("test_classification_debug")


def test_classification():
    """"""
    logger.info("=" * 60)
    logger.info(" - ")
    logger.info("=" * 60)

    # 
    logger.info("\n1. ...")
    extractor = FeatureExtraction(quantize=False)

    # 
    logger.info("\n2. ...")
    index_path = project_root / "role_index_augmented.faiss"
    logger.info(f": {index_path}")
    logger.info(f": {index_path.exists()}")

    mapping_path = project_root / "role_index_augmented_mapping.json"
    logger.info(f": {mapping_path.exists()}")

    classifier = Classification(index_path=str(index_path), threshold=0.4)

    if classifier.index is None:
        logger.error("")
        return

    logger.info(f" {len(classifier.role_mapping)} ")
    logger.info(f": {set(classifier.role_mapping)}")

    # 1: 
    logger.info("\n3. 1: ")
    test_image_path = project_root / "scripts" / "test_images" / "sample.jpg"
    if test_image_path.exists():
        logger.info(f": {test_image_path}")
        image = Image.open(test_image_path).convert("RGB")
        logger.info(f": {image.size}")

        # 
        features = extractor.extract_features(image)
        logger.info(f": {features.shape}")
        logger.info(f": {np.linalg.norm(features):.4f}")

        # 
        role, similarity = classifier.classify(features, top_k=10)
        logger.info(f": ={role}, ={similarity:.4f}")
    else:
        logger.warning(f": {test_image_path}")

    # 3: API
    logger.info("\n5. 3: ")
    test_image = Image.new("RGB", (224, 224), color="red")
    logger.info(f": 224x224 ")

    # 
    features = extractor.extract_features(test_image)
    logger.info(f": {features.shape}")
    logger.info(f": {np.linalg.norm(features):.4f}")
    logger.info(f"10: {features[:10]}")

    # 
    role, similarity = classifier.classify(features, top_k=10)
    logger.info(f": ={role}, ={similarity:.4f}")

    # 3: 
    logger.info("\n5. ")
    logger.info(f": {classifier.index.ntotal}")
    logger.info(f": {classifier.index.d}")

    # 
    if classifier.index.ntotal > 0:
        sample_vector = classifier.index.reconstruct(0)
        logger.info(f": {sample_vector.shape}")
        logger.info(f": {np.linalg.norm(sample_vector):.4f}")
        logger.info(f"10: {sample_vector[:10]}")

    logger.info("\n" + "=" * 60)
    logger.info("")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_classification()
