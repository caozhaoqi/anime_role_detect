#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测器对比测试 — 评估 bad_noface 中的漏检率

对比两个检测器:
  A. LBPCascade AnimeFace（当前清洗用的）
  B. MTCNN (facenet-pytorch)

从 bad_noface 中随机抽 N 张，分别用两个检测器检测，
统计"实际有人脸但被 LBPCascade 漏检"的比例。
"""

import random
import json
import time
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ====== 配置 ======
SAMPLE_SIZE = 200  # 检测样本数
RANDOM_SEED = 42
BAD_NOFACE_DIR = Path(
    "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/cleaned_fd/bad_noface"
)
CASCADE_PATH = Path(
    "/Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/data_cleaning/cascades/lbpcascade_animeface.xml"
)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

random.seed(RANDOM_SEED)


# ====== 检测器 A: LBPCascade ======
def init_lbpcascade():
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError("无法加载 LBPCascade")
    return cascade


def detect_lbpcascade(cascade, gray):
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    return faces


# ====== 检测器 B: MTCNN ======
def init_mtcnn():
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=True, device="cpu", min_face_size=20)
    return mtcnn


def detect_mtcnn(mtcnn, img_rgb):
    boxes, _ = mtcnn.detect(img_rgb)
    if boxes is None:
        return []
    return boxes


def main():
    logger.info("=" * 60)
    logger.info("检测器对比测试 — bad_noface 漏检评估")
    logger.info("=" * 60)

    # 收集图片
    images = []
    for role_dir in sorted(BAD_NOFACE_DIR.iterdir()):
        if not role_dir.is_dir():
            continue
        for f in role_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                images.append(f)
        if len(images) >= SAMPLE_SIZE * 2:  # 多收集些以便随机抽样
            break

    if len(images) <= SAMPLE_SIZE:
        sampled = images
    else:
        sampled = random.sample(images, SAMPLE_SIZE)

    logger.info(f"共收集 {len(images)} 张 bad_noface 图片")
    logger.info(f"随机抽样 {len(sampled)} 张进行检测器对比")
    logger.info("")

    # 初始化检测器
    cascade = init_lbpcascade()
    mtcnn = init_mtcnn()

    # 对比测试
    results = {
        "total": len(sampled),
        "lbpcascade_found": 0,
        "mtcnn_found": 0,
        "both_found": 0,
        "lbpcascade_only": 0,
        "mtcnn_only": 0,
        "neither": 0,
    }

    detail = []

    for i, img_path in enumerate(sampled):
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            logger.warning(f"  读取失败 [{i+1}/{len(sampled)}]: {img_path.name} - {e}")
            continue

        # LBPCascade
        lbp_faces = detect_lbpcascade(cascade, gray)
        lbp_ok = len(lbp_faces) > 0

        # MTCNN
        mtcnn_ok = False
        try:
            mtcnn_faces = detect_mtcnn(mtcnn, img_np)
            mtcnn_ok = len(mtcnn_faces) > 0
        except Exception as e:
            pass

        if lbp_ok:
            results["lbpcascade_found"] += 1
        if mtcnn_ok:
            results["mtcnn_found"] += 1
        if lbp_ok and mtcnn_ok:
            results["both_found"] += 1
        if lbp_ok and not mtcnn_ok:
            results["lbpcascade_only"] += 1
        if not lbp_ok and mtcnn_ok:
            results["mtcnn_only"] += 1
        if not lbp_ok and not mtcnn_ok:
            results["neither"] += 1

        detail.append({
            "file": str(img_path.relative_to(BAD_NOFACE_DIR)),
            "lbpcascade": bool(lbp_ok),
            "mtcnn": mtcnn_ok,
        })

        if (i + 1) % 50 == 0:
            logger.info(f"  进度: {i+1}/{len(sampled)}")

    # 报告
    total = len(sampled)
    logger.info("\n" + "=" * 60)
    logger.info("对比结果:")
    logger.info(f"  总测试数: {total}")
    logger.info(f"  LBPCascade 检出: {results['lbpcascade_found']} ({results['lbpcascade_found']/total*100:.1f}%)")
    logger.info(f"  MTCNN 检出:     {results['mtcnn_found']} ({results['mtcnn_found']/total*100:.1f}%)")
    logger.info(f"  两者都检出:     {results['both_found']}")
    logger.info(f"  仅 LBPCascade:  {results['lbpcascade_only']}")
    logger.info(f"  仅 MTCNN:       {results['mtcnn_only']} ← 这是LBPCascade漏检数")
    logger.info(f"  两者都未检出:   {results['neither']}")
    logger.info("")

    # 漏检率
    only_mtcnn = results["mtcnn_only"]
    false_negative_rate = only_mtcnn / total * 100
    logger.info(f"→ LBPCascade 漏检率 (MTCNN能检但LBPCascde不能): {false_negative_rate:.1f}%")
    logger.info(f"→ 修正后 good 预计: {592 + int(1193 * (results['mtcnn_found']/total))} 张"
                " (将漏检的从bad_noface召回)")
    logger.info("")

    # 直观统计: MTCNN认为实际有脸的
    actual_have_face = results["mtcnn_found"]
    logger.info(f"→ bad_noface {total} 张中，MTCNN认为有脸的: {actual_have_face} ({actual_have_face/total*100:.1f}%)")

    # 保存明细
    report = {
        "config": {"sample_size": SAMPLE_SIZE, "source": str(BAD_NOFACE_DIR)},
        "results": results,
        "false_negative_rate_pct": round(false_negative_rate, 1),
        "detail": detail,
    }
    report_file = "detector_comparison.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"明细已保存: {report_file}")


if __name__ == "__main__":
    main()