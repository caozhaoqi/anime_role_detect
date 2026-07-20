#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗流水线
整合所有清洗模块，从原始数据到清洗后数据的完整流程
"""

import os
import sys
import json
import time
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
from PIL import Image
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent.parent

from src.data_pipeline.cleaners import (
    CLIPDeduplicator,
    CharacterConsistencyFilter,
    HDBSCANClusterFilter,
    MislabeledDetector,
    DanbooruEnricher,
)
from src.core.recognition.clip_embedder_cached import CLIPEmbedderCached

logger = logging.getLogger("cleaning_pipeline")


@dataclass
class CleaningConfig:
    """清洗配置"""
    # 阶段开关
    enable_deduplication: bool = True
    enable_consistency_filter: bool = True
    enable_cluster_filter: bool = True
    enable_mislabeled_detector: bool = True
    enable_danbooru_enrichment: bool = False
    
    # CLIP去重
    similarity_threshold: float = 0.95
    dedup_dry_run: bool = False
    
    # 角色一致性
    consistency_threshold: float = 0.25
    consistency_dry_run: bool = False
    
    # HDBSCAN聚类
    min_cluster_size: int = 5
    outlier_threshold: float = 0.7
    cluster_dry_run: bool = False
    
    # 错误标签检测
    text_threshold: float = 0.2
    confusion_gap: float = 0.08
    outlier_score_threshold: float = 0.7
    
    # Danbooru增强
    mirror_site: str = "yande.re"
    
    # 通用
    min_images_per_character: int = 10
    max_workers: int = 4


@dataclass
class CharacterCleaningResult:
    """角色清洗结果"""
    character: str
    original_count: int = 0
    after_dedup: int = 0
    after_consistency: int = 0
    after_cluster: int = 0
    after_mislabeled: int = 0
    
    # 详细统计
    duplicate_pairs: int = 0
    low_consistency_count: int = 0
    cluster_outliers: int = 0
    mislabeled_count: int = 0
    
    # Danbooru元数据
    danbooru_tags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    # 状态
    status: str = "pending"
    error: Optional[str] = None


@dataclass
class PipelineReport:
    """流水线报告"""
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0
    
    # 总体统计
    total_characters: int = 0
    total_original_images: int = 0
    total_cleaned_images: int = 0
    total_removed_images: int = 0
    overall_keep_rate: float = 0
    
    # 各阶段统计
    preprocessing_stats: dict = field(default_factory=dict)
    dedup_removed: int = 0
    consistency_removed: int = 0
    cluster_removed: int = 0
    mislabeled_removed: int = 0
    
    # 角色详细结果
    character_results: Dict[str, dict] = field(default_factory=dict)
    
    # 配置
    config: dict = field(default_factory=dict)


class CleaningPipeline:
    """
    数据清洗流水线
    
    流程：
    0. 预处理 - 过滤损坏图片和完全重复图片（MD5哈希）
    1. CLIP去重 - 去除重复/高度相似图片
    2. 角色一致性过滤 - 过滤与角色标注不一致的图片
    3. HDBSCAN聚类过滤 - 去除角色内异常点
    4. 错误标签检测 - 综合检测标注错误的图片
    5. Danbooru增强 - 补充标签元数据（可选）
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        config: Optional[CleaningConfig] = None,
    ):
        """
        初始化清洗流水线
        
        Args:
            input_dir: 原始数据目录（包含角色子目录）
            output_dir: 清洗后数据输出目录
            config: 清洗配置
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or CleaningConfig()
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "_filtered").mkdir(exist_ok=True)
        
        # 初始化模块
        self._init_modules()
        
        # 结果
        self.results: Dict[str, CharacterCleaningResult] = {}
        self.report: Optional[PipelineReport] = None
        
        logger.info(f"清洗流水线初始化: {input_dir} -> {output_dir}")
    
    def _init_modules(self):
        """初始化各清洗模块"""
        logger.info("初始化清洗模块...")
        
        # CLIP Embedder
        self.embedder = CLIPEmbedderCached(
            model_name="ViT-B/32",
            cache_dir=str(project_root / "clip_cache"),
            use_huggingface=False,  # macOS上使用OpenAI CLIP避免MKL死锁
        )
        
        # CLIP去重器
        if self.config.enable_deduplication:
            self.deduplicator = CLIPDeduplicator(
                similarity_threshold=self.config.similarity_threshold,
                embedder=self.embedder,
            )
        
        # 角色一致性过滤器
        if self.config.enable_consistency_filter:
            self.consistency_filter = CharacterConsistencyFilter(
                consistency_threshold=self.config.consistency_threshold,
                embedder=self.embedder,
            )
        
        # HDBSCAN聚类过滤器
        if self.config.enable_cluster_filter:
            self.cluster_filter = HDBSCANClusterFilter(
                min_cluster_size=self.config.min_cluster_size,
                embedder=self.embedder,
            )
        
        # 错误标签检测器
        if self.config.enable_mislabeled_detector:
            self.mislabeled_detector = MislabeledDetector(embedder=self.embedder)
        
        # Danbooru增强器
        if self.config.enable_danbooru_enrichment:
            self.danbooru_enricher = DanbooruEnricher(mirror_site=self.config.mirror_site)
        
        logger.info("模块初始化完成")
    
    def get_character_images(self, char_dir: Path) -> List[str]:
        """获取角色目录下的所有图片"""
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            images.extend([str(p) for p in char_dir.glob(ext)])
        return sorted(images)
    
    def _compute_file_hash(self, filepath: Path) -> Optional[str]:
        """计算文件MD5哈希"""
        try:
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return None
    
    def _is_valid_image(self, filepath: Path) -> Tuple[bool, str]:
        """
        检查图片是否有效
        
        Returns:
            (是否有效, 原因)
        """
        try:
            # 检查文件大小
            filesize = os.path.getsize(filepath)
            if filesize < 1024:
                return False, f"文件太小 ({filesize} bytes)"
            
            # 尝试打开并验证
            with Image.open(filepath) as img:
                img.verify()
            
            # 重新打开检查尺寸和模式
            with Image.open(filepath) as img:
                width, height = img.size
                if width < 50 or height < 50:
                    return False, f"尺寸太小 ({width}x{height})"
                if img.mode not in ['RGB', 'RGBA', 'L', 'RGBX']:
                    return False, f"不支持的模式: {img.mode}"
            
            return True, "OK"
        except Exception as e:
            return False, str(e)
    
    def preprocess(self) -> dict:
        """
        预处理：过滤损坏图片和完全重复图片
        
        在进入CLIP特征提取前，先用简单方法过滤无效图片，
        大幅减少后续处理时间和CLIP调用次数
        
        Returns:
            预处理统计信息
        """
        logger.info("="*60)
        logger.info("步骤1: 预处理（过滤损坏和重复图片）")
        logger.info("="*60)
        
        # 创建临时预处理目录
        preprocessed_dir = self.output_dir / "_preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            "total_scanned": 0,
            "valid": 0,
            "corrupt": 0,
            "duplicate": 0,
            "by_character": {}
        }
        
        # 全局哈希缓存（跨角色去重）
        global_hash_cache = set()
        # 角色级哈希缓存
        char_hash_cache = set()
        
        char_dirs = sorted([d for d in self.input_dir.iterdir() if d.is_dir()])
        
        for char_dir in tqdm(char_dirs, desc="预处理角色"):
            char_name = char_dir.name
            out_char_dir = preprocessed_dir / char_name
            out_char_dir.mkdir(exist_ok=True)
            
            char_stats = {"scanned": 0, "valid": 0, "corrupt": 0, "duplicate": 0}
            char_hash_cache.clear()
            
            for img_file in char_dir.iterdir():
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                    continue
                
                char_stats["scanned"] += 1
                stats["total_scanned"] += 1
                
                # 检查图片有效性
                is_ok, reason = self._is_valid_image(img_file)
                if not is_ok:
                    char_stats["corrupt"] += 1
                    stats["corrupt"] += 1
                    logger.debug(f"  损坏: {char_name}/{img_file.name} - {reason}")
                    continue
                
                # 检查完全重复（MD5哈希）
                file_hash = self._compute_file_hash(img_file)
                if file_hash is None:
                    char_stats["corrupt"] += 1
                    stats["corrupt"] += 1
                    continue
                
                # 跨角色去重
                if file_hash in global_hash_cache:
                    char_stats["duplicate"] += 1
                    stats["duplicate"] += 1
                    logger.debug(f"  跨角色重复: {char_name}/{img_file.name}")
                    continue
                
                # 角色内去重
                if file_hash in char_hash_cache:
                    char_stats["duplicate"] += 1
                    stats["duplicate"] += 1
                    logger.debug(f"  角色内重复: {char_name}/{img_file.name}")
                    continue
                
                # 保留图片
                global_hash_cache.add(file_hash)
                char_hash_cache.add(file_hash)
                shutil.copy(img_file, out_char_dir / img_file.name)
                char_stats["valid"] += 1
                stats["valid"] += 1
            
            stats["by_character"][char_name] = char_stats
            if char_stats["corrupt"] > 0 or char_stats["duplicate"] > 0:
                logger.info(f"  {char_name}: 有效 {char_stats['valid']}/{char_stats['scanned']}, "
                          f"损坏 {char_stats['corrupt']}, 重复 {char_stats['duplicate']}")
        
        # 打印汇总
        logger.info(f"\n预处理完成:")
        logger.info(f"  扫描总数: {stats['total_scanned']}")
        logger.info(f"  有效保留: {stats['valid']}")
        logger.info(f"  移除损坏: {stats['corrupt']}")
        logger.info(f"  移除重复: {stats['duplicate']}")
        logger.info(f"  保留率: {stats['valid']/stats['total_scanned']*100:.1f}%")
        
        return stats
    
    def clean_character(self, char_dir: Path) -> CharacterCleaningResult:
        """
        清洗单个角色
        
        Args:
            char_dir: 角色目录
            
        Returns:
            清洗结果
        """
        char_name = char_dir.name
        result = CharacterCleaningResult(character=char_name)
        
        try:
            images = self.get_character_images(char_dir)
            result.original_count = len(images)
            
            if len(images) < self.config.min_images_per_character:
                result.status = "skipped"
                result.error = f"图片数不足 ({len(images)} < {self.config.min_images_per_character})"
                logger.warning(f"{char_name}: {result.error}")
                return result
            
            logger.info(f"\n{'='*50}")
            logger.info(f"清洗角色: {char_name} ({len(images)} 张图片)")
            logger.info(f"{'='*50}")
            
            # 创建工作目录
            work_dir = self.output_dir / char_name
            work_dir.mkdir(exist_ok=True)
            
            # 复制原始图片到工作目录
            import shutil
            for img_path in images:
                dst = work_dir / Path(img_path).name
                if not dst.exists():
                    shutil.copy(img_path, dst)
            
            current_images = [str(work_dir / Path(p).name) for p in images]
            
            # ===== 阶段1: CLIP去重 =====
            if self.config.enable_deduplication:
                logger.info(f"[{char_name}] 阶段1: CLIP去重")
                
                dedup_result = self.deduplicator.deduplicate_directory(
                    str(work_dir),
                    recursive=False,
                    dry_run=True,  # 干运行获取统计
                )
                
                result.duplicate_pairs = dedup_result.get("duplicate_pairs", 0)
                
                if not self.config.dedup_dry_run:
                    dedup_result = self.deduplicator.deduplicate_directory(
                        str(work_dir),
                        recursive=False,
                        dry_run=False,
                    )
                
                result.after_dedup = dedup_result.get("kept_images", len(current_images))
                result.dedup_removed = result.original_count - result.after_dedup
                current_images = self.get_character_images(work_dir)
                
                logger.info(f"  去重后: {result.after_dedup} 张 (移除 {result.dedup_removed})")
            else:
                result.after_dedup = len(current_images)
            
            # ===== 阶段2: 角色一致性过滤 =====
            if self.config.enable_consistency_filter and len(current_images) >= 5:
                logger.info(f"[{char_name}] 阶段2: 角色一致性过滤")
                
                # 先构建特征库用于一致性检测
                scores = self.consistency_filter.filter_character_images(
                    current_images,
                    char_name,
                    return_scores=True,
                )
                
                # 统计低一致性图片
                low_consistency = [p for p, s in scores if s < self.config.consistency_threshold]
                result.low_consistency_count = len(low_consistency)
                
                if not self.config.consistency_dry_run and low_consistency:
                    # 移动低一致性图片
                    filtered_dir = work_dir / "_low_consistency"
                    filtered_dir.mkdir(exist_ok=True)
                    
                    for img_path in low_consistency:
                        dst = filtered_dir / Path(img_path).name
                        os.rename(img_path, dst)
                
                result.after_consistency = len(current_images) - len(low_consistency)
                result.consistency_removed = result.after_dedup - result.after_consistency
                current_images = self.get_character_images(work_dir)
                
                logger.info(f"  一致性过滤后: {result.after_consistency} 张 (移除 {result.consistency_removed})")
            else:
                result.after_consistency = len(current_images)
            
            # ===== 阶段3: HDBSCAN聚类过滤 =====
            if self.config.enable_cluster_filter and len(current_images) >= self.config.min_cluster_size:
                logger.info(f"[{char_name}] 阶段3: HDBSCAN聚类过滤")
                
                analysis = self.cluster_filter.analyze_clusters(current_images)
                
                # 收集异常点
                outliers = []
                for label, stats in analysis.get("cluster_stats", {}).items():
                    if label == -1:  # 噪声点
                        outliers.extend([img["path"] for img in stats["images"]])
                    else:
                        # 检查簇内异常分数高的
                        for img in stats["images"]:
                            if img["outlier_score"] >= self.config.outlier_threshold:
                                outliers.append(img["path"])
                
                result.cluster_outliers = len(outliers)
                
                if not self.config.cluster_dry_run and outliers:
                    outlier_dir = work_dir / "_outliers"
                    outlier_dir.mkdir(exist_ok=True)
                    
                    for img_path in outliers:
                        dst = outlier_dir / Path(img_path).name
                        os.rename(img_path, dst)
                
                result.after_cluster = len(current_images) - len(outliers)
                result.cluster_removed = result.after_consistency - result.after_cluster
                current_images = self.get_character_images(work_dir)
                
                logger.info(f"  聚类过滤后: {result.after_cluster} 张 (移除 {result.cluster_removed})")
            else:
                result.after_cluster = len(current_images)
            
            # ===== 阶段4: 错误标签检测 =====
            if self.config.enable_mislabeled_detector and len(current_images) >= 5:
                logger.info(f"[{char_name}] 阶段4: 错误标签检测")
                
                # 构建特征库
                self.mislabeled_detector.build_feature_library(str(work_dir))
                
                # 检测可疑图片
                thresholds = {
                    "text_similarity": self.config.text_threshold,
                    "confusion_gap": self.config.confusion_gap,
                    "outlier_score": self.config.outlier_score_threshold,
                }
                
                suspicious = self.mislabeled_detector.scan_directory(
                    str(work_dir),
                    thresholds=thresholds,
                )
                
                mislabeled = [s["path"] for s in suspicious if s["suspicious"]]
                result.mislabeled_count = len(mislabeled)
                
                if mislabeled:
                    mislabeled_dir = work_dir / "_mislabeled"
                    mislabeled_dir.mkdir(exist_ok=True)
                    
                    for img_path in mislabeled:
                        dst = mislabeled_dir / Path(img_path).name
                        os.rename(img_path, dst)
                
                result.after_mislabeled = len(current_images) - len(mislabeled)
                result.mislabeled_removed = max(0, result.after_cluster - result.after_mislabeled)
                
                logger.info(f"  错误标签检测后: {result.after_mislabeled} 张 (移除 {result.mislabeled_removed})")
            else:
                result.after_mislabeled = len(current_images)
            
            # ===== 阶段5: Danbooru增强 =====
            if self.config.enable_danbooru_enrichment:
                logger.info(f"[{char_name}] 阶段5: Danbooru标签增强")
                
                try:
                    tags = self.danbooru_enricher.get_character_tags(char_name)
                    result.danbooru_tags = tags.get("character", []) + tags.get("general", [])[:10]
                    result.confidence = min(1.0, len(result.danbooru_tags) / 20)
                    
                    # 保存元数据
                    metadata = {
                        "character": char_name,
                        "danbooru_tags": tags,
                        "cleaning_stats": asdict(result),
                    }
                    
                    meta_path = work_dir / "metadata.json"
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"  Danbooru标签: {len(result.danbooru_tags)} 个")
                except Exception as e:
                    logger.warning(f"  Danbooru增强失败: {e}")
            
            result.status = "completed"
            logger.info(f"✅ {char_name} 清洗完成: {result.original_count} -> {result.after_mislabeled}")
            
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            logger.error(f"❌ {char_name} 清洗失败: {e}")
        
        return result
    
    def run(self, skip_preprocess: bool = False) -> PipelineReport:
        """
        运行清洗流水线
        
        Args:
            skip_preprocess: 是否跳过预处理步骤（已预处理过的数据可跳过）
            
        Returns:
            流水线报告
        """
        start_time = time.time()
        
        report = PipelineReport(
            start_time=datetime.now().isoformat(),
            config=asdict(self.config),
        )
        
        logger.info("="*60)
        logger.info("数据清洗流水线启动")
        logger.info("="*60)
        logger.info(f"输入目录: {self.input_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"配置: {json.dumps(asdict(self.config), indent=2)}")
        
        # 保存原始输入目录
        original_input_dir = self.input_dir
        
        # 步骤1: 预处理（可选）
        preprocessed_dir = self.output_dir / "_preprocessed"
        if skip_preprocess and preprocessed_dir.exists():
            logger.info("跳过预处理（已存在预处理数据）")
            self.input_dir = preprocessed_dir
        else:
            # 运行预处理
            preprocess_stats = self.preprocess()
            report.preprocessing_stats = preprocess_stats
            
            # 将输入目录切换到预处理后的目录
            self.input_dir = preprocessed_dir
        
        # 获取所有角色目录
        char_dirs = sorted([d for d in self.input_dir.iterdir() if d.is_dir()])
        report.total_characters = len(char_dirs)
        
        logger.info(f"找到 {len(char_dirs)} 个角色目录")
        logger.info(f"使用 {self.config.max_workers} 个并发线程")
        
        # 预初始化CLIP模型（多线程安全：在进入线程池前先初始化）
        # 避免多个线程同时触发CLIP懒加载导致的PyTorch C++ mutex死锁
        if self.config.enable_deduplication or self.config.enable_consistency_filter:
            logger.info("预初始化CLIP模型...")
            self.embedder.embedder.initialize()
            logger.info("CLIP模型预初始化完成")
        
        # 清洗每个角色（支持并发）
        if self.config.max_workers > 1:
            # 并发处理
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(self.clean_character, char_dir): char_dir.name 
                          for char_dir in char_dirs}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="清洗角色"):
                    char_name = futures[future]
                    try:
                        result = future.result()
                        self.results[char_name] = result
                    except Exception as e:
                        logger.error(f"角色 {char_name} 处理异常: {e}")
                        self.results[char_name] = CharacterCleaningResult(
                            character=char_name,
                            status="failed",
                            error=str(e)
                        )
        else:
            # 顺序处理
            for char_dir in tqdm(char_dirs, desc="清洗角色"):
                result = self.clean_character(char_dir)
                self.results[char_dir.name] = result
        
        # 生成报告
        end_time = time.time()
        report.end_time = datetime.now().isoformat()
        report.duration_seconds = end_time - start_time
        
        # 汇总统计
        total_original = sum(r.original_count for r in self.results.values())
        total_cleaned = sum(r.after_mislabeled for r in self.results.values() if r.status == "completed")
        
        report.total_original_images = total_original
        report.total_cleaned_images = total_cleaned
        report.total_removed_images = total_original - total_cleaned
        report.overall_keep_rate = total_cleaned / total_original if total_original > 0 else 0
        
        report.dedup_removed = sum(r.dedup_removed for r in self.results.values())
        report.consistency_removed = sum(r.consistency_removed for r in self.results.values())
        report.cluster_removed = sum(r.cluster_removed for r in self.results.values())
        report.mislabeled_removed = sum(r.mislabeled_removed for r in self.results.values())
        
        # 角色详细结果
        report.character_results = {
            name: asdict(result) for name, result in self.results.items()
        }
        
        self.report = report
        
        # 保存报告
        self._save_report()
        
        # 打印汇总
        self._print_summary()
        
        return report
    
    def _save_report(self):
        """保存报告"""
        report_path = self.output_dir / "cleaning_report.json"
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.report), f, indent=2, ensure_ascii=False)
        
        logger.info(f"报告已保存: {report_path}")
    
    def _print_summary(self):
        """打印汇总"""
        report = self.report
        
        print("\n" + "="*60)
        print("📊 数据清洗流水线报告")
        print("="*60)
        
        print(f"\n⏱️  运行时间: {report.duration_seconds:.1f} 秒")
        print(f"📁 输入目录: {self.input_dir}")
        print(f"📁 输出目录: {self.output_dir}")
        
        print(f"\n📈 总体统计:")
        print(f"   角色数量: {report.total_characters}")
        print(f"   原始图片: {report.total_original_images}")
        print(f"   清洗后图片: {report.total_cleaned_images}")
        print(f"   移除图片: {report.total_removed_images}")
        print(f"   总体保留率: {report.overall_keep_rate:.1%}")
        
        print(f"\n🔍 各阶段移除统计:")
        print(f"   CLIP去重: {report.dedup_removed}")
        print(f"   一致性过滤: {report.consistency_removed}")
        print(f"   聚类过滤: {report.cluster_removed}")
        print(f"   错误标签检测: {report.mislabeled_removed}")
        
        print(f"\n👤 角色详细结果:")
        for name, result in sorted(self.results.items()):
            keep_rate = result.after_mislabeled / result.original_count if result.original_count > 0 else 0
            status_icon = "✅" if result.status == "completed" else "❌"
            print(f"   {status_icon} {name}: {result.original_count} -> {result.after_mislabeled} ({keep_rate:.0%})")
        
        print("\n" + "="*60)
        print(f"报告已保存到: {self.output_dir}/cleaning_report.json")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="数据清洗流水线")
    parser.add_argument("--input", "-i", required=True, help="输入目录")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    
    # 阶段开关
    parser.add_argument("--no-dedup", action="store_true", help="跳过去重阶段")
    parser.add_argument("--no-consistency", action="store_true", help="跳过一致性过滤")
    parser.add_argument("--no-cluster", action="store_true", help="跳过聚类过滤")
    parser.add_argument("--no-mislabeled", action="store_true", help="跳过错误标签检测")
    parser.add_argument("--enable-danbooru", action="store_true", help="启用Danbooru增强")
    
    # 参数
    parser.add_argument("--similarity", type=float, default=0.95, help="相似度阈值")
    parser.add_argument("--consistency", type=float, default=0.25, help="一致性阈值")
    parser.add_argument("--outlier", type=float, default=0.7, help="异常阈值")
    
    # 干运行
    parser.add_argument("--dry-run", action="store_true", help="干运行（不删除文件）")
    
    args = parser.parse_args()
    
    # 构建配置
    config = CleaningConfig(
        enable_deduplication=not args.no_dedup,
        enable_consistency_filter=not args.no_consistency,
        enable_cluster_filter=not args.no_cluster,
        enable_mislabeled_detector=not args.no_mislabeled,
        enable_danbooru_enrichment=args.enable_danbooru,
        similarity_threshold=args.similarity,
        consistency_threshold=args.consistency,
        outlier_threshold=args.outlier,
        dedup_dry_run=args.dry_run,
        consistency_dry_run=args.dry_run,
        cluster_dry_run=args.dry_run,
    )
    
    # 运行流水线
    pipeline = CleaningPipeline(args.input, args.output, config)
    report = pipeline.run()
    
    return 0 if report.total_removed_images > 0 or report.total_cleaned_images > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
