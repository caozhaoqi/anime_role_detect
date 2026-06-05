"""
CLIP去重系统
CLIP-based Deduplication System
"""
# 必须在导入任何其他模块之前设置环境变量
import os
import sys
import platform
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Mac平台禁用CUDA，避免mutex错误
if platform.system() == "Darwin":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["FORCE_CPU"] = "1"

import numpy as np
import torch
import clip
from PIL import Image
import imagehash


class CLIPDeduplicator:
    """CLIP去重器"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        初始化CLIP去重器
        
        Args:
            model_name: CLIP模型名称，如"ViT-B/32"、"ViT-L/14"
            device: 运行设备，None表示自动选择
        """
        if device is not None:
            self.device = device
        elif platform.system() == "Darwin":
            # Mac平台优先使用MPS，否则使用CPU
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_name = model_name
        
        # 加载CLIP模型
        print(f"📥 正在加载CLIP模型: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        print(f"✅ CLIP模型加载完成，运行设备: {self.device}")
    
    def compute_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        计算单张图片的CLIP向量
        
        Args:
            image_path: 图片路径
        
        Returns:
            图片向量，形状为(512,)或(768,)，失败返回None
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
            
            # 归一化并转换为numpy数组
            embedding = embedding.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        
        except Exception as e:
            print(f"⚠️ 计算图片向量失败 {image_path}: {str(e)}")
            return None
    
    def compute_embeddings(self, image_paths: List[str], batch_size: int = 32) -> List[Tuple[str, np.ndarray]]:
        """
        批量计算图片向量
        
        Args:
            image_paths: 图片路径列表
            batch_size: 批处理大小
        
        Returns:
            成功计算的图片路径和向量列表
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    tensor = self.preprocess(image).unsqueeze(0)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"⚠️ 加载图片失败 {path}: {str(e)}")
            
            if batch_tensors:
                batch = torch.cat(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    embeddings = self.model.encode_image(batch)
                
                embeddings = embeddings.cpu().numpy()
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                # 匹配成功加载的图片
                idx = 0
                for path in batch_paths:
                    try:
                        Image.open(path).convert("RGB")
                        results.append((path, embeddings[idx]))
                        idx += 1
                    except:
                        pass
        
        return results
    
    def compute_phash(self, image_path: str) -> Optional[str]:
        """
        计算图片的感知哈希
        
        Args:
            image_path: 图片路径
        
        Returns:
            哈希值字符串，失败返回None
        """
        try:
            image = Image.open(image_path)
            phash = imagehash.phash(image)
            return str(phash)
        except Exception as e:
            print(f"⚠️ 计算感知哈希失败 {image_path}: {str(e)}")
            return None
    
    def deduplicate_by_phash(self, image_paths: List[str], threshold: int = 5) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        使用感知哈希去重
        
        Args:
            image_paths: 图片路径列表
            threshold: 哈希差异阈值
        
        Returns:
            (保留的图片路径, 重复对列表)
        """
        hash_map = {}
        duplicates = []
        retained = []
        
        for path in image_paths:
            phash = self.compute_phash(path)
            if phash is None:
                continue
            
            found_duplicate = False
            for existing_hash, existing_path in hash_map.items():
                # 计算汉明距离
                hamming_dist = sum(c1 != c2 for c1, c2 in zip(phash, existing_hash))
                if hamming_dist <= threshold:
                    duplicates.append((path, existing_path))
                    found_duplicate = True
                    break
            
            if not found_duplicate:
                hash_map[phash] = path
                retained.append(path)
        
        return retained, duplicates
    
    def deduplicate_by_clip(self, image_embeddings: List[Tuple[str, np.ndarray]], 
                            threshold: float = 0.98) -> Tuple[List[str], List[Tuple[str, str, float]]]:
        """
        使用CLIP向量去重
        
        Args:
            image_embeddings: (图片路径, 向量)列表
            threshold: 相似度阈值
        
        Returns:
            (保留的图片路径, 重复对列表)
        """
        if len(image_embeddings) < 2:
            return [path for path, _ in image_embeddings], []
        
        paths = [path for path, _ in image_embeddings]
        embeddings = np.array([emb for _, emb in image_embeddings])
        
        # 计算相似度矩阵
        similarity_matrix = embeddings @ embeddings.T
        
        # 找出重复对
        n = len(paths)
        visited = set()
        duplicates = []
        retained = []
        
        for i in range(n):
            if i in visited:
                continue
            
            retained.append(paths[i])
            visited.add(i)
            
            for j in range(i + 1, n):
                if j in visited:
                    continue
                
                similarity = similarity_matrix[i, j]
                if similarity >= threshold:
                    duplicates.append((paths[j], paths[i], float(similarity)))
                    visited.add(j)
        
        return retained, duplicates
    
    def deduplicate(self, image_paths: List[str], phash_threshold: int = 5, 
                    clip_threshold: float = 0.98, batch_size: int = 32) -> Tuple[List[str], Dict]:
        """
        完整去重流程：先使用感知哈希快速过滤，再使用CLIP精细去重
        
        Args:
            image_paths: 图片路径列表
            phash_threshold: 感知哈希阈值
            clip_threshold: CLIP相似度阈值
            batch_size: CLIP批处理大小
        
        Returns:
            (保留的图片路径, 去重统计信息)
        """
        print(f"🔍 开始去重，共 {len(image_paths)} 张图片")
        
        # 阶段1: 感知哈希去重
        print("📊 阶段1: 感知哈希去重...")
        after_phash, phash_duplicates = self.deduplicate_by_phash(image_paths, phash_threshold)
        print(f"   感知哈希去重完成，保留 {len(after_phash)} 张，去除 {len(phash_duplicates)} 对重复")
        
        # 阶段2: CLIP向量去重
        print("📊 阶段2: CLIP向量去重...")
        if len(after_phash) > 0:
            embeddings = self.compute_embeddings(after_phash, batch_size)
            after_clip, clip_duplicates = self.deduplicate_by_clip(embeddings, clip_threshold)
        else:
            after_clip = []
            clip_duplicates = []
        
        print(f"   CLIP去重完成，保留 {len(after_clip)} 张，去除 {len(clip_duplicates)} 对重复")
        
        # 统计信息
        stats = {
            "original_count": len(image_paths),
            "after_phash_count": len(after_phash),
            "after_clip_count": len(after_clip),
            "phash_duplicates_count": len(phash_duplicates),
            "clip_duplicates_count": len(clip_duplicates),
            "total_removed": len(image_paths) - len(after_clip),
            "phash_duplicates": phash_duplicates,
            "clip_duplicates": clip_duplicates
        }
        
        print(f"✅ 去重完成！共去除 {stats['total_removed']} 张重复图片")
        
        return after_clip, stats


# 示例用法
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CLIP去重工具")
    parser.add_argument("-i", "--input", required=True, help="图片目录")
    parser.add_argument("-o", "--output", help="去重后保存目录")
    parser.add_argument("--phash-threshold", type=int, default=5, help="感知哈希阈值")
    parser.add_argument("--clip-threshold", type=float, default=0.98, help="CLIP相似度阈值")
    parser.add_argument("--batch-size", type=int, default=32, help="批处理大小")
    
    args = parser.parse_args()
    
    # 获取图片列表
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
    image_paths = [
        str(p) for p in Path(args.input).rglob('*') 
        if p.suffix.lower() in image_extensions
    ]
    
    print(f"📁 找到 {len(image_paths)} 张图片")
    
    # 创建去重器
    deduplicator = CLIPDeduplicator()
    
    # 执行去重
    retained, stats = deduplicator.deduplicate(
        image_paths,
        phash_threshold=args.phash_threshold,
        clip_threshold=args.clip_threshold,
        batch_size=args.batch_size
    )
    
    # 打印统计
    print("\n📊 去重统计:")
    print(f"   原始数量: {stats['original_count']}")
    print(f"   感知哈希去重后: {stats['after_phash_count']}")
    print(f"   CLIP去重后: {stats['after_clip_count']}")
    print(f"   总共去除: {stats['total_removed']}")
    
    # 如果指定了输出目录，复制保留的图片
    if args.output:
        import shutil
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for path in retained:
            filename = Path(path).name
            shutil.copy(path, output_dir / filename)
        
        print(f"✅ 已将 {len(retained)} 张图片复制到 {output_dir}")