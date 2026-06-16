#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTCNN 重跑 final_dataset 清洗

用 MTCNN 替换 LBPCascade，重新对 final_dataset 进行分类：
  - good:      单人近景（face_ratio ≥ 1% 且 < 40%）
  - noface:    无人脸
  - multiface: 多人（≥2 个人脸）
  - farface:   远景（face_ratio < 1%）
  - chibi:     Q版（face_ratio ≥ 40%）

修改：
  - 远景阈值从 3% 降到 1%（动漫图更合理）
  - 检测器从 LBPCascade 换成 MTCNN
"""

import json
import time
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from facenet_pytorch import MTCNN

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ====== 路径 ======
FD_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
CLEANED_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/cleaned_fd_mtcnn")
REPORT_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data")

# ====== 阈值 ======
FAR_FACE_RATIO = 0.01   # 远景：人脸面积 < 1%
CHIBI_FACE_RATIO = 0.40  # Q版：人脸面积 > 40%
IMG_SIZE_LIMIT = 2000     # 超图缩放上限
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def init_mtcnn():
    return MTCNN(keep_all=True, device="cpu", min_face_size=20)


def classify_image(img_path: Path, mtcnn) -> dict:
    result = {
        "path": str(img_path),
        "role": img_path.parent.name,
        "filename": img_path.name,
        "category": "unknown",
        "face_count": 0,
        "face_ratio": 0.0,
        "error": None,
    }

    try:
        pil_img = Image.open(img_path).convert("RGB")
        w, h = pil_img.size

        # 缩放超大图
        if max(w, h) > IMG_SIZE_LIMIT:
            scale = IMG_SIZE_LIMIT / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h = pil_img.size

        img_np = np.array(pil_img)

        # MTCNN 检测
        boxes, _ = mtcnn.detect(img_np)
        if boxes is None:
            result["category"] = "noface"
            result["face_count"] = 0
            return result

        faces = boxes.tolist()
        result["face_count"] = len(faces)

        if len(faces) >= 2:
            result["category"] = "multiface"
            return result

        # 单人：计算人脸面积比
        x1, y1, x2, y2 = faces[0]
        face_area = (x2 - x1) * (y2 - y1)
        img_area = h * w
        face_ratio = face_area / img_area if img_area > 0 else 0
        result["face_ratio"] = round(face_ratio, 4)

        if face_ratio < FAR_FACE_RATIO:
            result["category"] = "farface"
        elif face_ratio > CHIBI_FACE_RATIO:
            result["category"] = "chibi"
        else:
            result["category"] = "good"

    except UnidentifiedImageError:
        result["category"] = "error"
        result["error"] = "cannot_identify"
    except Exception as e:
        result["category"] = "error"
        result["error"] = str(e)

    return result


def move_to(img_path: Path, target_dir: Path):
    """移动图片到目标目录（保持角色子目录结构）"""
    role = img_path.parent.name
    dst_dir = target_dir / role
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / img_path.name
    # 处理重名
    if dst.exists():
        stem = img_path.stem
        suffix = img_path.suffix
        idx = 1
        while dst.exists():
            dst = dst_dir / f"{stem}_{idx}{suffix}"
            idx += 1
    img_path.rename(dst)


def main():
    logger.info("=" * 60)
    logger.info("MTCNN 重跑 final_dataset 清洗")
    logger.info("=" * 60)

    mtcnn = init_mtcnn()
    logger.info("MTCNN 初始化完成")

    # 统计
    categories = {"good": 0, "noface": 0, "multiface": 0, "farface": 0, "chibi": 0, "error": 0}
    role_stats = {}
    total = 0
    t0 = time.time()

    # 收集所有图片
    all_images = []
    for role_dir in sorted(FD_DIR.iterdir()):
        if not role_dir.is_dir() or role_dir.name.startswith("."):
            continue
        for f in role_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                all_images.append(f)

    logger.info(f"共 {len(all_images)} 张图片需处理\n")

    # 逐张检测
    for i, img_path in enumerate(all_images):
        role = img_path.parent.name
        result = classify_image(img_path, mtcnn)
        cat = result["category"]
        categories[cat] = categories.get(cat, 0) + 1
        total += 1

        if role not in role_stats:
            role_stats[role] = {"total": 0, "good": 0}
        role_stats[role]["total"] += 1
        if cat == "good":
            role_stats[role]["good"] += 1

        # 非 good 的移动走（good 保留在原地）
        if cat != "good":
            bad_dir = CLEANED_DIR / f"bad_{cat}"
            move_to(img_path, bad_dir)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(f"  进度: {i+1}/{len(all_images)} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    logger.info(f"\n处理完成! 耗时: {elapsed:.0f}s")
    logger.info("=" * 60)
    logger.info("分类统计:")
    logger.info(f"  ✅ good:      {categories['good']:>5} ({categories['good']/total*100:.1f}%)")
    logger.info(f"  ❌ noface:    {categories['noface']:>5} ({categories['noface']/total*100:.1f}%)")
    logger.info(f"  ❌ multiface: {categories['multiface']:>5} ({categories['multiface']/total*100:.1f}%)")
    logger.info(f"  ❌ farface:   {categories['farface']:>5} ({categories['farface']/total*100:.1f}%)")
    logger.info(f"  ❌ chibi:     {categories['chibi']:>5} ({categories['chibi']/total*100:.1f}%)")
    logger.info(f"  ⚠️ error:     {categories['error']:>5} ({categories['error']/total*100:.1f}%)")
    logger.info(f"  ─────────────────────")
    logger.info(f"  总计: {total}")

    # 各角色 good 率
    logger.info("\n各角色保留统计 (good/total):")
    low_roles = []
    for role, s in sorted(role_stats.items()):
        pct = s["good"] / s["total"] * 100
        logger.info(f"  {role:20s} {s['good']:3d}/{s['total']:<3d} ({pct:5.1f}%)")
        if pct < 20:
            low_roles.append((role, pct))

    if low_roles:
        logger.info(f"\n⚠️ 保留率 < 20% 的角色:")
        for r, p in low_roles:
            logger.info(f"  {r}: {p:.1f}%")

    # 保存报告
    report = {
        "detector": "MTCNN",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "total": total,
        "categories": categories,
        "far_face_ratio_threshold": FAR_FACE_RATIO,
        "chibi_face_ratio_threshold": CHIBI_FACE_RATIO,
        "role_stats": {k: v for k, v in sorted(role_stats.items())},
    }
    report_file = REPORT_DIR / f"mtcnn_clean_report_{report['timestamp']}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\n报告已保存: {report_file}")

    # 对比 LBP 旧结果
    logger.info("\n对比 LBP 旧清洗:")
    logger.info(f"  LBP   good: 592 (23.6%)")
    logger.info(f"  MTCNN good: {categories['good']} ({categories['good']/total*100:.1f}%)")


if __name__ == "__main__":
    main()