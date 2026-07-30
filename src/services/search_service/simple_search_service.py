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
            # 索引文件不存在时跳过初始化：避免无谓加载 CLIP(ViT-B/32 ~600MB)
            # 导致 multimedia 进程 OOM 被杀（以图搜图请求直接让服务断开连接）。
            index_file = self.index_path or "data/feature_store/character_index.faiss"
            metadata_file = "data/feature_store/character_metadata.json"
            if not (os.path.exists(index_file) and os.path.exists(metadata_file)):
                print(f"特征索引不存在({index_file} / {metadata_file})，跳过检索服务初始化，search 将返回空结果")
                self.retriever = None
                self._initialized = True
                return
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

    def get_index_stats(self):
        """返回索引统计信息；索引未初始化时返回安全默认值。

        注意：只读 feature_store 统计，**不调用 CharacterRetriever.get_stats()**
        （后者会触发 initialize() 加载 CLIP 模型，可能让 multimedia 进程 OOM）。
        修复：multimedia 路由此前调用了不存在的 get_index_stats() 方法，
        导致 /search/stats 抛 AttributeError。
        """
        self._ensure_initialized()
        if self.retriever is None:
            return {"index_count": 0, "status": "not_initialized", "model_name": None}
        try:
            fs_stats = self.retriever.feature_store.get_stats()
            if not isinstance(fs_stats, dict):
                fs_stats = {}
            return {
                "index_count": fs_stats.get("total_vectors", fs_stats.get("count", 0)),
                "status": "running",
                "model_name": "ViT-B/32",
                "dimension": fs_stats.get("dimension"),
                "index_type": fs_stats.get("index_type"),
                "characters": fs_stats.get("characters", []),
            }
        except Exception as e:
            return {"index_count": 0, "status": "error", "error": str(e)}

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