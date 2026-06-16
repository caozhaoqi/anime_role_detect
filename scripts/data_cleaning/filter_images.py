#!/usr/bin/env python3
"""
自动过滤垃圾图
- YOLOv8 → 人体检测 (person class=0)
- Anime Face Cascade + 备用人脸检测
- 过滤: 无人脸/多人/角色过小 → 将 2610 张压缩到 ~1500 张高质量图
"""

import os
import shutil
import time
import logging
import json
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("filter")

# ===== 路径 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "final_dataset"
FILTERED_DIR = PROJECT_ROOT / "data" / "filtered_dataset"
REPORT_FILE = PROJECT_ROOT / "data" / "filtered_report.md"
CASCADE_ANIME = PROJECT_ROOT / "scripts" / "data_cleaning" / "cascades" / "lbpcascade_animeface.xml"
CASCADE_DEFAULT = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
STATS_FILE = Path("/tmp") / "filter_stats.json"

# ===== 阈值 =====
PERSON_CONF = 0.15         # YOLO person 置信度 (低阈值: 挽救更多图片)
PERSON_AREA_MIN = 0.01     # 人体最小面积比 (1%)
FACE_AREA_MIN = 0.001      # 人脸最小面积比 (0.1%)

# 多进程 — 每个 worker 会加载一个 YOLO 模型, 开太多会内存爆炸
N_WORKERS = 2


def init_worker():
    """Worker 初始化 — 每个进程加载自己的 YOLO + Cascade"""
    global _yolo, _cascade_anime, _cascade_default
    import warnings
    warnings.filterwarnings("ignore")

    from ultralytics import YOLO
    _yolo = YOLO("yolov8n.pt")
    _cascade_anime = cv2.CascadeClassifier(str(CASCADE_ANIME))
    _cascade_default = cv2.CascadeClassifier(CASCADE_DEFAULT)


def detect_person(img_np):
    """YOLO 检测 person"""
    global _yolo
    results = _yolo(img_np, verbose=False, workers=0)
    h, w = img_np.shape[:2]
    persons = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0
            if cls_id == 0 and conf >= PERSON_CONF:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                persons.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "area_ratio": (x2 - x1) * (y2 - y1) / (w * h),
                })
    return persons


def detect_faces_multi(img_np):
    """多级人脸检测: Anime Cascade → Default Haar → 返回合并结果"""
    global _cascade_anime, _cascade_default
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    h, w = img_np.shape[:2]
    all_faces = []

    # 1. Anime-specific cascade (严格参数, 减少误报)
    faces = _cascade_anime.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(30, 30))
    for (fx, fy, fw, fh) in faces:
        all_faces.append({"bbox": [fx, fy, fx + fw, fy + fh], "area_ratio": fw * fh / (w * h), "source": "anime"})

    # 2. 通用 haar cascade
    if len(all_faces) == 0:
        faces = _cascade_default.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(40, 40))
        for (fx, fy, fw, fh) in faces:
            all_faces.append({"bbox": [fx, fy, fx + fw, fy + fh], "area_ratio": fw * fh / (w * h), "source": "default"})

    return all_faces


def process_one(img_path_str: str) -> dict:
    """处理单张图片 — 被 worker 调用"""
    global _yolo
    img = cv2.imread(img_path_str)
    if img is None:
        return {"path": img_path_str, "status": "error", "reason": "unreadable"}

    # 1. YOLO person
    persons = detect_person(img)
    pc = len(persons)
    par = sum(p["area_ratio"] for p in persons) if persons else 0
    mpr = max((p["area_ratio"] for p in persons), default=0)

    # 2. Face cascade
    faces = detect_faces_multi(img)
    fc = len(faces)
    far = sum(f["area_ratio"] for f in faces) if faces else 0
    mfr = max((f["area_ratio"] for f in faces), default=0)

    return {
        "path": img_path_str,
        "status": "ok",
        "person_count": pc,
        "person_area_ratio": round(par, 4),
        "max_person_ratio": round(mpr, 4),
        "face_count": fc,
        "face_area_ratio": round(far, 4),
        "max_face_ratio": round(mfr, 4),
    }


def decide(result: dict) -> tuple:
    """
    YOLO 为主、cascade 为辅的判决逻辑
    三级策略:
      Level 1 (首选): YOLO 检出 1 个 person → 保留
      Level 2 (宽松): YOLO 检出 2 persons 但主要人物够大 → 保留 (可能小配角)
      Level 3 (挽救): 无人无脸但图像非空白 → 保留 (纯色/空白图才丢弃)
    """
    if result["status"] != "ok":
        return ("delete", result.get("reason", "error"))

    fc = result["face_count"]
    pc = result["person_count"]
    mpr = result["max_person_ratio"]
    mfr = result["max_face_ratio"]

    # Level 1: YOLO 单人 → 保留
    if pc == 1:
        return ("keep", f"单人(pc=1,fc={fc},par={mpr:.3f})")

    # Level 2: YOLO 两人, 主要人物够大 → 保留 (可能主+小配角)
    if pc == 2 and mpr >= 0.08:
        return ("keep", f"双人但主角色大(pc=2,par={mpr:.3f})")

    # YOLO > 2 人 → 跳过
    if pc >= 2:
        return ("skip", f"多人体(pc={pc})")

    # YOLO 没检出 person → 靠人脸检测兜底
    if fc == 1:
        if mfr >= FACE_AREA_MIN:
            return ("keep", f"仅人脸(far={mfr:.3f})")
        else:
            return ("skip", f"人脸过小(far={mfr:.3f})")
    elif fc == 0:
        return ("skip", "无人无脸")
    else:
        return ("skip", f"无人但多脸(fc={fc})")


def main():
    logger.info("=" * 60)
    logger.info("自动过滤垃圾图")
    logger.info(f"数据源: {DATA_DIR}")
    logger.info(f"输出: {FILTERED_DIR}")
    logger.info(f"阈值: PERSON_CONF={PERSON_CONF}, PERSON_AREA≥{PERSON_AREA_MIN}, FACE_AREA≥{FACE_AREA_MIN}")
    logger.info(f"Worker: {N_WORKERS}")
    logger.info("=" * 60)

    # 扫描图片
    char_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    all_images = []
    for d in char_dirs:
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                all_images.append(f)
    total = len(all_images)
    logger.info(f"共扫描到 {total} 张图片")

    os.makedirs(FILTERED_DIR, exist_ok=True)

    # 多进程处理
    img_paths = [str(p) for p in all_images]
    pool = mp.Pool(N_WORKERS, initializer=init_worker)
    start = time.time()

    stats = {
        "total": total, "keep": 0, "skip": 0, "delete": 0,
        "by_reason": defaultdict(int),
        "by_char": defaultdict(lambda: {"total": 0, "keep": 0, "skip": 0}),
        "face_stats": {"face_ok": 0, "face_no_but_person": 0, "no_face_no_person": 0},
    }

    # imap_unordered: 边处理边拿结果
    for idx, result in enumerate(pool.imap_unordered(process_one, img_paths)):
        char_name = Path(result["path"]).parent.name
        stats["by_char"][char_name]["total"] += 1

        action, reason = decide(result)

        if action == "keep":
            stats["keep"] += 1
            stats["by_char"][char_name]["keep"] += 1
            # copy
            src = Path(result["path"])
            out_dir = FILTERED_DIR / char_name
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(str(src), str(out_dir / src.name))
        elif action == "skip":
            stats["skip"] += 1
            stats["by_char"][char_name]["skip"] += 1
            stats["by_reason"][reason] += 1
        elif action == "delete":
            stats["delete"] += 1

        # 进度
        if (idx + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"  进度: {idx+1}/{total} | {rate:.1f} img/s | keep={stats['keep']} skip={stats['skip']} | 预计剩余 {(total-idx-1)/rate:.0f}s" if rate > 0 else "")

    pool.close()
    pool.join()
    elapsed = time.time() - start

    # === 结果 ===
    logger.info("=" * 60)
    logger.info("过滤完成!")
    logger.info(f"耗时: {elapsed:.1f}s ({elapsed/total:.2f}s/img)")
    logger.info(f"图片总数: {stats['total']}")
    logger.info(f"保留(keep): {stats['keep']} ({stats['keep']/stats['total']*100:.1f}%)")
    logger.info(f"跳过(skip): {stats['skip']}")
    logger.info(f"删除(error): {stats['delete']}")

    logger.info("\n跳过原因分布:")
    for reason, cnt in sorted(stats["by_reason"].items(), key=lambda x: -x[1]):
        logger.info(f"  {reason}: {cnt}")

    # 保存统计
    with open(STATS_FILE, "w") as f:
        json.dump({
            "total": stats["total"], "keep": stats["keep"], "skip": stats["skip"],
            "by_reason": dict(stats["by_reason"]),
            "by_char": {k: dict(v) for k, v in stats["by_char"].items()},
            "elapsed": elapsed,
        }, f, ensure_ascii=False, indent=2)

    generate_report(stats)
    logger.info(f"报告: {REPORT_FILE}")


def generate_report(stats):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = stats["total"]
    keep = stats["keep"]
    skip = stats["skip"]

    lines = []
    lines.append("# 数据集过滤报告\n")
    lines.append(f"**生成时间**: {now}\n")
    lines.append(f"**过滤阈值**: PERSON_CONF={PERSON_CONF}, PERSON_AREA≥{PERSON_AREA_MIN}, FACE_AREA≥{FACE_AREA_MIN}\n")
    lines.append("---\n")
    lines.append("## 一、总体概览\n")
    lines.append("| 指标 | 数值 | 占比 |")
    lines.append("|------|------|------|")
    lines.append(f"| 原始图片 | {total} | 100% |")
    lines.append(f"| 保留 | {keep} | {keep/total*100:.1f}% |")
    lines.append(f"| 跳过 | {skip} | {skip/total*100:.1f}% |")
    lines.append(f"| 出错 | {stats['delete']} | {stats['delete']/total*100:.1f}% |")
    lines.append("")
    lines.append("## 二、跳过原因分布\n")
    lines.append("| 原因 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for reason, cnt in sorted(stats["by_reason"].items(), key=lambda x: -x[1]):
        lines.append(f"| {reason} | {cnt} | {cnt/total*100:.1f}% |")
    lines.append("")
    lines.append("## 三、各角色通过率\n")
    lines.append("| 角色 | 总数 | 保留 | 跳过 | 通过率 |")
    lines.append("|------|------|------|------|--------|")
    for name, c in sorted(stats["by_char"].items(), key=lambda x: x[1]["keep"], reverse=True):
        rate = c["keep"] / c["total"] * 100 if c["total"] > 0 else 0
        lines.append(f"| {name} | {c['total']} | {c['keep']} | {c['skip']} | {rate:.0f}% |")
    lines.append("")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()