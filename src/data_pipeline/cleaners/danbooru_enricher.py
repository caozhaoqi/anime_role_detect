#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Danbooru数据增强器
使用Danbooru标签丰富动漫角色数据
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import requests
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("danbooru_enricher")


class DanbooruEnricher:
    """
    Danbooru数据增强器
    
    功能：
    1. 从Danbooru/Yande.re获取角色相关标签
    2. 为本地图片添加标签元数据
    3. 识别并过滤非角色图片
    4. 补充角色变体/形态信息
    """
    
    def __init__(
        self,
        mirror_site: str = "yande.re",
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        cache_dir: str = "data/danbooru_cache",
        rate_limit: float = 1.0,
    ):
        """
        初始化增强器
        
        Args:
            mirror_site: 镜像站点 (yande.re, lolibooru, konachan)
            api_key: API密钥（用于官方Danbooru）
            user_id: 用户ID（用于官方Danbooru）
            cache_dir: 缓存目录
            rate_limit: 请求间隔（秒）
        """
        self.mirror_site = mirror_site
        self.api_key = api_key
        self.user_id = user_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        
        # 站点配置
        self.site_config = {
            "yande.re": {
                "url": "https://yande.re",
                "post_url": "https://yande.re/post.json",
                "tags_url": "https://yande.re/tag.json",
            },
            "lolibooru": {
                "url": "https://lolibooru.moe",
                "post_url": "https://lolibooru.moe/post.json",
                "tags_url": "https://lolibooru.moe/tag.json",
            },
            "konachan": {
                "url": "https://konachan.com",
                "post_url": "https://konachan.com/post.json",
                "tags_url": "https://konachan.com/tag.json",
            },
        }
        
        # 缓存
        self.tag_cache: Dict[str, List[str]] = {}
        self.last_request_time = 0
        
        logger.info(f"Danbooru增强器初始化: 站点={mirror_site}")
    
    def _rate_limit(self):
        """请求限速"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _get_cache_path(self, tag: str) -> Path:
        """获取缓存文件路径"""
        safe_tag = tag.lower().replace(" ", "_").replace("/", "_")
        return self.cache_dir / f"{safe_tag}.json"
    
    def search_posts(
        self,
        tags: List[str],
        limit: int = 100,
    ) -> List[Dict]:
        """
        搜索帖子
        
        Args:
            tags: 搜索标签列表
            limit: 返回数量限制
            
        Returns:
            帖子列表
        """
        cache_path = self._get_cache_path("_".join(tags))
        
        # 检查缓存
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if cached.get("count", 0) >= 50:
                    logger.info(f"使用缓存: {tags}")
                    return cached.get("posts", [])
        
        # 构建URL
        params = {
            "tags": "+".join(tags),
            "limit": min(limit, 200),
        }
        
        try:
            self._rate_limit()
            
            response = requests.get(
                self.site_config[self.mirror_site]["post_url"],
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            
            posts = response.json()
            
            # 缓存结果
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"tags": tags, "count": len(posts), "posts": posts}, f)
            
            logger.info(f"搜索成功: {tags} -> {len(posts)} 条结果")
            return posts
            
        except Exception as e:
            logger.error(f"搜索失败 {tags}: {e}")
            return []
    
    def get_character_tags(
        self,
        character_name: str,
        work_title: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        获取角色相关标签
        """
        tags = {
            "character": [],
            "artist": [],
            "copyright": [],
            "general": [],
        }
        
        search_tags = []
        if work_title:
            search_tags.append(work_title.lower().replace(" ", "_"))
        search_tags.append(character_name.lower().replace(" ", "_"))
        
        posts = self.search_posts(search_tags, limit=50)
        
        if not posts:
            return tags
        
        for post in posts:
            post_tags = post.get("tags", "").split()
            
            for tag in post_tags:
                if tag.startswith("character:") or tag.endswith(f"({character_name.lower().replace(' ', '_')})"):
                    if tag not in tags["character"]:
                        tags["character"].append(tag)
                elif tag.startswith("artist:"):
                    if tag not in tags["artist"]:
                        tags["artist"].append(tag)
                elif tag.startswith("copyright:") or tag.startswith("series:") or tag.startswith("game:"):
                    if tag not in tags["copyright"]:
                        tags["copyright"].append(tag)
                else:
                    if tag not in tags["general"]:
                        tags["general"].append(tag)
        
        return tags
    
    def enrich_image_metadata(
        self,
        image_path: str,
        character_name: str,
        work_title: Optional[str] = None,
    ) -> Dict:
        """
        为图片添加元数据
        """
        metadata = {
            "image_path": image_path,
            "character": character_name,
            "work_title": work_title,
            "danbooru_tags": {},
            "suggested_tags": [],
            "confidence": 0.0,
        }
        
        tags = self.get_character_tags(character_name, work_title)
        metadata["danbooru_tags"] = tags
        
        recommended = []
        for tag in tags["character"]:
            if character_name.lower().replace(" ", "_") in tag:
                recommended.append(tag)
        
        important_tags = [
            "solo", "1girl", "animal_ears", "blush", "smile",
            "blue_eyes", "brown_hair", "long_hair", "school_uniform",
            "twintails", "ponytail", "short_hair", "silver_hair",
            "ahoge", "hair_ornament", "bow", "ribbon",
        ]
        
        for tag in tags["general"]:
            tag_name = tag.split(":")[-1] if ":" in tag else tag
            if tag_name in important_tags and tag not in recommended:
                recommended.append(tag)
        
        metadata["suggested_tags"] = recommended[:20]
        metadata["confidence"] = min(1.0, len(tags["character"]) / 3 + len(recommended) / 20)
        
        return metadata
    
    def enrich_directory(
        self,
        directory: str,
        character_name: str,
        work_title: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict:
        """
        增强目录中所有图片的元数据
        """
        dir_path = Path(directory)
        
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            image_paths.extend(list(dir_path.glob(ext)))
        
        if not image_paths:
            return {"message": "目录中没有图片"}
        
        logger.info(f"开始增强 {character_name}: {len(image_paths)} 张图片")
        
        results = []
        for img_path in tqdm(image_paths, desc=f"增强 {character_name}"):
            metadata = self.enrich_image_metadata(str(img_path), character_name, work_title)
            results.append(metadata)
            
            if not dry_run:
                meta_path = img_path.with_suffix(img_path.suffix + ".json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        avg_confidence = np.mean([r["confidence"] for r in results]) if results else 0
        
        return {
            "character": character_name,
            "total_images": len(image_paths),
            "avg_confidence": avg_confidence,
            "tag_coverage": {
                "character": len(results[0]["danbooru_tags"]["character"]) if results else 0,
                "general": len(results[0]["danbooru_tags"]["general"]) if results else 0,
            },
            "results": results,
        }
    
    def find_related_characters(
        self,
        character_name: str,
        work_title: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """
        查找相关角色
        """
        if work_title:
            search_tags = [work_title.lower().replace(" ", "_")]
        else:
            search_tags = [character_name.lower().replace(" ", "_")]
        
        posts = self.search_posts(search_tags, limit=100)
        
        related_chars = {}
        for post in posts:
            tags = post.get("tags", "").split()
            
            for tag in tags:
                if tag.startswith("character:") and character_name.lower().replace(" ", "_") not in tag:
                    char = tag.replace("character:", "")
                    if char not in related_chars:
                        related_chars[char] = 0
                    related_chars[char] += 1
        
        sorted_chars = sorted(related_chars.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"character": char, "count": count}
            for char, count in sorted_chars[:limit]
        ]
    
    def download_reference_images(
        self,
        character_name: str,
        work_title: Optional[str] = None,
        save_dir: str = "data/references",
        limit: int = 20,
    ) -> List[str]:
        """
        下载角色参考图
        """
        tags = []
        if work_title:
            tags.append(work_title.lower().replace(" ", "_"))
        tags.append(character_name.lower().replace(" ", "_"))
        
        posts = self.search_posts(tags, limit=limit)
        
        save_path = Path(save_dir) / character_name.replace(" ", "_")
        save_path.mkdir(parents=True, exist_ok=True)
        
        downloaded = []
        for post in tqdm(posts[:limit], desc=f"下载 {character_name}"):
            try:
                image_url = post.get("file_url")
                if not image_url:
                    continue
                
                ext = Path(post.get("file_ext", "jpg")).suffix
                filename = f"{post['id']}{ext}"
                filepath = save_path / filename
                
                if filepath.exists():
                    continue
                
                self._rate_limit()
                response = requests.get(image_url, timeout=60)
                response.raise_for_status()
                
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                downloaded.append(str(filepath))
                
            except Exception as e:
                logger.warning(f"下载失败: {e}")
        
        logger.info(f"下载完成: {len(downloaded)}/{len(posts[:limit])} 张图片")
        return downloaded


class DanbooruTagMatcher:
    """
    Danbooru标签匹配器
    
    将本地图片与Danbooru标签匹配，过滤不相关图片
    """
    
    def __init__(self, enricher: DanbooruEnricher = None):
        self.enricher = enricher or DanbooruEnricher()
        self.character_tags: Dict[str, List[str]] = {}
    
    def load_character_tags(
        self,
        character_mapping: Dict[str, Tuple[str, Optional[str]]],
    ):
        """
        加载角色标签
        
        Args:
            character_mapping: {角色目录: (角色名, 作品名)}
        """
        for char_dir, (char_name, work_title) in tqdm(character_mapping.items()):
            tags = self.enricher.get_character_tags(char_name, work_title)
            all_tags = (
                tags["character"] + 
                tags["artist"] + 
                tags["copyright"] + 
                tags["general"]
            )
            self.character_tags[char_dir] = all_tags
            logger.info(f"加载标签: {char_name} -> {len(all_tags)} 个标签")
    
    def match_image_to_tags(
        self,
        image_path: str,
        candidate_tags: List[str],
    ) -> float:
        """
        计算图片与标签的匹配度
        
        Returns:
            匹配度分数 (0-1)
        """
        try:
            from src.core.recognition.clip_embedder_cached import CLIPEmbedderCached
            
            embedder = CLIPEmbedderCached()
            
            image_feat = embedder.embed_image(image_path)
            if image_feat is None:
                return 0.0
            
            text_feats = embedder.embed_texts(candidate_tags[:10])
            if text_feats is None:
                return 0.0
            
            avg_sim = 0.0
            for tf in text_feats:
                tf_norm = tf / (np.linalg.norm(tf) + 1e-8)
                sim = float(np.dot(image_feat, tf_norm))
                avg_sim += sim
            
            return avg_sim / len(text_feats) if text_feats else 0.0
            
        except Exception as e:
            logger.warning(f"匹配失败 {image_path}: {e}")
            return 0.0
    
    def filter_with_tags(
        self,
        directory: str,
        character_dir: str,
        threshold: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """
        使用标签过滤图片
        
        Returns:
            [(图片路径, 匹配度)] 列表
        """
        tags = self.character_tags.get(character_dir, [])
        if not tags:
            return []
        
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            image_paths.extend([str(p) for p in Path(directory).glob(ext)])
        
        results = []
        for path in tqdm(image_paths, desc=f"匹配 {character_dir}"):
            score = self.match_image_to_tags(path, tags)
            results.append((path, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Danbooru数据增强")
    parser.add_argument("--character", "-c", required=True, help="角色名称")
    parser.add_argument("--work", "-w", help="作品名称")
    parser.add_argument("--output", "-o", help="保存目录")
    parser.add_argument("--site", default="yande.re", choices=["yande.re", "lolibooru", "konachan"])
    
    args = parser.parse_args()
    
    enricher = DanbooruEnricher(mirror_site=args.site)
    
    # 获取标签
    tags = enricher.get_character_tags(args.character, args.work)
    print(f"角色标签: {json.dumps(tags, indent=2, ensure_ascii=False)}")
    
    # 查找相关角色
    related = enricher.find_related_characters(args.character, args.work)
    print(f"相关角色: {json.dumps(related, indent=2, ensure_ascii=False)}")
    
    # 下载参考图
    if args.output:
        downloaded = enricher.download_reference_images(
            args.character, args.work, args.output
        )
        print(f"下载完成: {len(downloaded)} 张图片")
