from PIL import Image
from src.core.classification.classification import Classification
from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction
import numpy as np


class SimpleImageSearchService:
    def __init__(self, index_path="./models/efficientnet_b3_loli_optimized_v2_20260529_133654"):
        self.index_path = index_path
        self.classifier = None
        self.preprocessor = None
        self.feature_extractor = None
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            self.classifier = Classification(self.index_path, threshold=0.1)
            self.preprocessor = Preprocessing()
            self.feature_extractor = FeatureExtraction()
            self._initialized = True

    def search(self, image: Image.Image, top_k: int = 10):
        self._ensure_initialized()
        try:
            processed_image = self.preprocessor.preprocess(image)
            if processed_image is None:
                return []

            feature = self.feature_extractor.extract_features(processed_image)

            if self.classifier and self.classifier.index is not None:
                role, similarity = self.classifier.classify(feature, top_k)
                results = []
                for r, s in zip(
                    role if isinstance(role, list) else [role],
                    similarity if isinstance(similarity, list) else [similarity],
                ):
                    results.append((r, s))
                return results
            return []
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
