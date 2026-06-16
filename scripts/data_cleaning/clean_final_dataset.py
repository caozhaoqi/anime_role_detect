#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_dataset 自动清洗脚本 v2

检测器: LBPCascade AnimeFace（专为动漫人脸优化）

过滤策略:
  1. 无人脸      → bad_noface
  2. 多人脸(>=2)  → bad_multiface
  3. 远景(人脸面积 < 3%) → bad_farface
  4. Q版(人脸面积 > 40%) → bad_chibi
  5. 其余 → 原地保留（good）

输出:
  - clean_report_*.json  汇总报告
  - cleaned_fd/bad_{type}/角色/  坏图
"""

import os
import sys
import json
import time
import shutil
import logging
import urllib.request
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

# 允许处理超大图
Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"clean_fd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ====== 配置 ======
FD_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
OUT_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/cleaned_fd")
CASCADE_DIR = Path(__file__).parent / "cascades"
ANIME_CASCADE_URL = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"

# 过滤阈值
FAR_FACE_RATIO = 0.03      # 人脸面积 < 3% → 远景
CHIBI_FACE_RATIO = 0.40     # 人脸面积 > 40% → Q版
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def get_anime_cascade():
    """下载并加载动漫人脸级联分类器"""
    CASCADE_DIR.mkdir(parents=True, exist_ok=True)
    cascade_path = CASCADE_DIR / "lbpcascade_animeface.xml"

    if not cascade_path.exists():
        logger.info("下载动漫人脸检测模型 (lbpcascade_animeface)...")
        try:
            urllib.request.urlretrieve(ANIME_CASCADE_URL, str(cascade_path))
            logger.info(f"下载完成: {cascade_path}")
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return None

    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        logger.error("加载级联分类器失败")
        return None
    logger.info("动漫人脸检测器加载完成")
    return cascade


def classify_image(img_path_str: str, cascade, img_size_limit: int = 2000) -> dict:
    """对单张图片分类"""
    pp = Path(img_path_str)
    result = {
        "path": img_path_str,
        "role": pp.parent.name,
        "filename": pp.name,
        "category": "unknown",
        "face_count": 0,
        "face_ratio": 0.0,
        "error": None,
    }

    # 读取图片
    try:
        pil_img = Image.open(img_path_str).convert("RGB")
        # 超大图缩放加速
        w, h = pil_img.size
        scale = 1.0
        if max(w, h) > img_size_limit:
            scale = img_size_limit / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            w, h = new_w, new_h
        img = np.array(pil_img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        result["error"] = f"read_failed: {e}"
        result["category"] = "error"
        return result

    img_area = h * w

    # 检测人脸
    try:
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
    except Exception as e:
        result["error"] = f"detect_failed: {e}"
        result["category"] = "error"
        return result

    result["face_count"] = len(faces)

    if len(faces) == 0:
        result["category"] = "noface"
        return result

    if len(faces) >= 2:
        result["category"] = "multiface"
        return result

    # 单人脸 → 计算人脸占比
    x, y, fw, fh = faces[0]
    face_area = fw * fh
    face_ratio = face_area / img_area if img_area > 0 else 0
    result["face_ratio"] = round(face_ratio, 4)

    if face_ratio < FAR_FACE_RATIO:
        result["category"] = "farface"
    elif face_ratio > CHIBI_FACE_RATIO:
        result["category"] = "chibi"
    else:
        result["category"] = "good"

    return result


def main():
    logger.info("=" * 60)
    logger.info("final_dataset 自动清洗 v2 (LBPCascade AnimeFace)")
    logger.info(f"阈值: farface<{FAR_FACE_RATIO*100}%  chibi>{CHIBI_FACE_RATIO*100}%")
    logger.info("=" * 60)

    if not FD_DIR.exists():
        logger.error(f"目录不存在: {FD_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载检测器
    cascade = get_anime_cascade()
    if cascade is None:
        logger.error("检测器加载失败，退出")
        return

    roles = sorted([d for d in FD_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])

    stats = {"total": 0, "good": 0, "noface": 0, "multiface": 0,
             "farface": 0, "chibi": 0, "error": 0}
    role_details = {}
    start = time.time()

    for role_dir in roles:
        role_name = role_dir.name
        images = sorted([f for f in role_dir.iterdir()
                         if f.is_file() and f.suffix.lower() in IMG_EXTS])
        if not images:
            continue

        stats["total"] += len(images)
        logger.info(f"处理角色: {role_name} ({len(images)} 张)")

        role_stats = {"total": len(images), "good": 0, "noface": 0,
                      "multiface": 0, "farface": 0, "chibi": 0, "error": 0}

        for img_path in images:
            res = classify_image(str(img_path), cascade)
            cat = res["category"]
            role_stats[cat] += 1
            stats[cat] += 1

            if cat == "good":
                continue

            # 移动坏图
            bad_dir = OUT_DIR / f"bad_{cat}" / role_name
            bad_dir.mkdir(parents=True, exist_ok=True)
            dst = bad_dir / img_path.name
            try:
                shutil.move(str(img_path), str(dst))
            except Exception as e:
                logger.warning(f"  移动失败 {img_path.name}: {e}")

        # 角色摘要
        g, n, m, f, c, e = (role_stats["good"], role_stats["noface"],
                            role_stats["multiface"], role_stats["farface"],
                            role_stats["chibi"], role_stats["error"])
        logger.info(f"  ✅{g} ❌noface={n} multi={m} far={f} chibi={c} err={e}")
        role_details[role_name] = role_stats

        # 空目录清理
        remaining = [f for f in role_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in IMG_EXTS]
        if not remaining:
            try:
                role_dir.rmdir()
                logger.info(f"  {role_name} 目录已空，自动删除")
            except Exception:
                pass

    elapsed = time.time() - start

    # 汇总
    total = stats["total"]
    logger.info("\n" + "=" * 60)
    logger.info(f"清洗完成！耗时: {elapsed:.0f}s")
    logger.info(f"总计: {total} 张")
    for k in ["good", "noface", "multiface", "farface", "chibi", "error"]:
        v = stats[k]
        pct = v / total * 100 if total else 0
        icon = "✅" if k == "good" else "❌"
        logger.info(f"  {icon} {k:10s}: {v:>5} ({pct:.1f}%)")

    # 保存报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(FD_DIR),
        "total": total,
        "filtered": {k: stats[k] for k in ["good", "noface", "multiface", "farface", "chibi", "error"]},
        "thresholds": {"far_face_ratio": FAR_FACE_RATIO, "chibi_face_ratio": CHIBI_FACE_RATIO},
        "detector": "lbpcascade_animeface",
        "elapsed_seconds": elapsed,
        "role_details": role_details,
    }

    report_file = f"clean_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"\n报告: {report_file}")
    logger.info(f"输出: {OUT_DIR}/")


if __name__ == "__main__":
    main()