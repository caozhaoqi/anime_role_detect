#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索服务客户端 - 提供统一的搜索接口
"""

import io
from PIL import Image
from src.services.search_service.simple_search_service import SimpleImageSearchService


class SearchClient:
    """搜索服务客户端"""

    def __init__(self):
        self.service = None
        self.service_url = "local"
        self._init_service()

    def _init_service(self):
        """初始化搜索服务"""
        try:
            self.service = SimpleImageSearchService()
        except Exception as e:
            print(f"初始化搜索服务失败: {e}")
            self.service = None

    def health_check(self):
        """检查服务健康状态"""
        return self.service is not None

    def search_image_bytes(self, content, filename, top_k=10):
        """搜索相似图像"""
        if not self.service:
            return {"success": False, "error": "搜索服务未初始化"}

        try:
            image = Image.open(io.BytesIO(content))
            results = self.service.search(image, top_k)

            search_results = []
            for role, similarity in results:
                search_results.append({
                    "role": role,
                    "similarity": float(similarity) if hasattr(similarity, '__float__') else similarity,
                    "path": f"images/{role}/{filename}"
                })

            return {
                "success": True,
                "count": len(search_results),
                "results": search_results
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def build_index(self, dataset_dir):
        """构建搜索索引"""
        return {
            "success": True,
            "dataset_dir": dataset_dir,
            "added_images": 0,
            "index_stats": {"status": "not_implemented"}
        }

    def get_stats(self):
        """获取统计信息"""
        return {
            "status": "healthy" if self.health_check() else "unhealthy",
            "index_size": 0,
            "total_images": 0
        }


# 全局搜索客户端实例
_search_client = None


def get_search_client():
    """获取搜索客户端实例"""
    global _search_client
    if _search_client is None:
        _search_client = SearchClient()
    return _search_client
