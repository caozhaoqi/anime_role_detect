from PIL import Image
from src.core.recognition.character_retriever import CharacterRetriever
import numpy as np
import os


class SimpleImageSearchService:
    def __init__(self, index_path=None):
        self.index_path = index_path
        self.retriever = None
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            try:
                self.retriever = CharacterRetriever(
                    clip_model_name="ViT-B/32",
                    feature_store_path="data/feature_store/character_index.faiss",
                    metadata_path="data/feature_store/character_metadata.json",
                    similarity_threshold=0.3,
                )
                self._initialized = True
            except Exception as e:
                print(f"初始化搜索服务失败: {e}")
                self.retriever = None

    def search(self, image: Image.Image, top_k: int = 10):
        self._ensure_initialized()
        
        if self.retriever is None:
            return []

        try:
            results = self.retriever.identify(image, top_k=top_k)
            
            search_results = []
            for result in results:
                character = result.get("character", "unknown")
                similarity = result.get("similarity", 0.0)
                search_results.append((character, similarity))
            
            return search_results
        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def build_index(self, dataset_dir: str):
        self._ensure_initialized()
        
        if self.retriever is None:
            return {"success": False, "error": "检索器未初始化"}

        try:
            results = self.retriever.register_characters_from_dataset(
                dataset_dir=dataset_dir,
                max_samples_per_character=30,
                use_prototype=True,
            )
            
            saved = self.retriever.save()
            
            added_count = sum(1 for r in results if r.get("success") and not r.get("skipped"))
            
            return {
                "success": True,
                "dataset_dir": dataset_dir,
                "added_images": added_count,
                "index_stats": self.retriever.get_stats(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}