#!/usr/bin/env python3
"""
training_dataset 数据质量清洗工具（安全版）

设计原则:
  1. 默认 DRY-RUN —— 不加 --apply 绝不动任何文件, 只打印将要做什么。
  2. 绝不 rm —— --apply 模式下所有"删除"都是移动到隔离目录, 可随时恢复。
  3. 不自动裁决歧义 —— 跨类重复图(同一张图被标成两个角色)只生成待裁决清单,
     由人决定哪个类才是正确标签, 脚本不猜。

处理的质量问题:
  A. 真实格式与后缀不符 (GIF/WebP/PNG 挂 .jpg) —— PIL 只读第 0 帧, 常是标题页/黑场,
     等于给模型喂噪声标签。动作: 抽取中间帧转 RGB JPEG, 原文件隔离。
  B. 截断/损坏图 —— verify() + load() 双重检测。动作: 隔离。
  C. 跨类字节完全相同重复 —— 同图两标签, 既是标签矛盾也是切分泄漏源。
     动作: 仅生成待裁决清单 JSON, 不自动删。
  D. 非 RGB 模式 (RGBA/P/L/CMYK) —— 仅统计上报。文件不动, 因为
     src/common/preprocess.py 已在读取层统一 convert('RGB')。

用法:
    # 预演(默认, 只读)
    .venv/bin/python scripts/data_processing/clean_training_dataset.py

    # 真正执行(隔离坏文件 + 转码 GIF)
    .venv/bin/python scripts/data_processing/clean_training_dataset.py --apply

    # 换数据集
    .venv/bin/python scripts/data_processing/clean_training_dataset.py \
        --dataset-dir data/final_dataset --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageFile

# 与训练侧保持一致: 容忍截断图, 放宽解压炸弹阈值(仅用于扫描阶段判定)
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 200_000_000

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# PIL format -> 该 format 合法的后缀集合
FORMAT_OK_EXTS = {
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "WEBP": {".webp"},
    "GIF": {".gif"},
    "BMP": {".bmp"},
    "MPO": {".jpg", ".jpeg"},  # 多图 JPEG, 后缀仍是 jpg
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_images(root: Path):
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                yield cls_dir.name, f


def probe(path: Path) -> dict:
    """返回 {format, mode, size, n_frames, truncated, error}"""
    info = {
        "format": None,
        "mode": None,
        "size": None,
        "n_frames": 1,
        "truncated": False,
        "error": None,
    }
    try:
        with Image.open(path) as im:
            info["format"] = im.format
            info["mode"] = im.mode
            info["size"] = list(im.size)
            info["n_frames"] = getattr(im, "n_frames", 1)
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        return info

    # verify() 会消耗文件句柄, 必须重开; 且它对某些截断漏检, 所以再 load() 一次
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception as e:  # noqa: BLE001
        info["truncated"] = True
        info["error"] = f"verify: {type(e).__name__}: {e}"
        return info

    try:
        # 关掉容错, 让真实截断暴露出来
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(path) as im:
            im.load()
    except Exception as e:  # noqa: BLE001
        info["truncated"] = True
        info["error"] = f"load: {type(e).__name__}: {e}"
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    return info


def convert_animated_to_jpeg(src: Path, dst: Path) -> bool:
    """抽取动图中间帧(避开常见的标题页/黑场首帧), 转 RGB JPEG 落到 dst。"""
    try:
        with Image.open(src) as im:
            n = getattr(im, "n_frames", 1)
            im.seek(n // 2 if n > 1 else 0)
            frame = im.convert("RGB")
            dst.parent.mkdir(parents=True, exist_ok=True)
            frame.save(dst, format="JPEG", quality=95)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    [!] 转码失败 {src}: {e}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="training_dataset 安全清洗工具")
    ap.add_argument("--dataset-dir", default="data/training_dataset")
    ap.add_argument("--quarantine-dir", default=None,
                    help="隔离目录, 默认 data/_quarantine/<name>_<ts>")
    ap.add_argument("--apply", action="store_true",
                    help="真正执行。不加则只预演(只读)")
    ap.add_argument("--report", default=None, help="JSON 报告输出路径")
    args = ap.parse_args()

    root = Path(args.dataset_dir).resolve()
    if not root.is_dir():
        print(f"数据集目录不存在: {root}", file=sys.stderr)
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine = Path(args.quarantine_dir) if args.quarantine_dir else \
        root.parent / "_quarantine" / f"{root.name}_{ts}"
    report_path = Path(args.report) if args.report else \
        root.parent.parent / "outputs" / f"clean_{root.name}_{ts}.json"

    mode_label = "APPLY(真正执行)" if args.apply else "DRY-RUN(只读预演)"
    print("=" * 78)
    print(f"数据集清洗 · {mode_label}")
    print(f"  数据集   : {root}")
    print(f"  隔离目录 : {quarantine}")
    print(f"  报告     : {report_path}")
    print("=" * 78)

    misnamed, truncated = [], []
    by_hash: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    mode_counter: dict[str, int] = defaultdict(int)
    fmt_counter: dict[str, int] = defaultdict(int)
    per_class: dict[str, int] = defaultdict(int)
    total = 0

    print("\n[1/3] 扫描中 ...")
    for cls, path in iter_images(root):
        total += 1
        per_class[cls] += 1
        info = probe(path)
        fmt = info["format"] or "UNREADABLE"
        fmt_counter[fmt] += 1
        if info["mode"]:
            mode_counter[info["mode"]] += 1

        if info["truncated"] or (info["error"] and info["format"] is None):
            truncated.append({"cls": cls, "path": str(path), "error": info["error"]})
        elif fmt in FORMAT_OK_EXTS and path.suffix.lower() not in FORMAT_OK_EXTS[fmt]:
            misnamed.append({
                "cls": cls, "path": str(path), "real_format": fmt,
                "suffix": path.suffix.lower(), "n_frames": info["n_frames"],
            })

        try:
            by_hash[sha256_of(path)].append((cls, path))
        except OSError:
            pass

        if total % 500 == 0:
            print(f"    ... {total} 张")

    cross_class_dups = [
        {"sha256": h, "entries": [{"cls": c, "path": str(p)} for c, p in items]}
        for h, items in by_hash.items()
        if len({c for c, _ in items}) > 1
    ]
    same_class_dups = [
        {"sha256": h, "entries": [{"cls": c, "path": str(p)} for c, p in items]}
        for h, items in by_hash.items()
        if len(items) > 1 and len({c for c, _ in items}) == 1
    ]
    non_rgb = sum(v for k, v in mode_counter.items() if k != "RGB")

    print(f"\n[2/3] 扫描完成: {len(per_class)} 类 / {total} 张")
    print(f"  格式分布      : {dict(sorted(fmt_counter.items()))}")
    print(f"  模式分布      : {dict(sorted(mode_counter.items()))}")
    print(f"  非 RGB        : {non_rgb} 张 ({non_rgb / max(total, 1):.1%})")
    print(f"  后缀不符      : {len(misnamed)} 张  <- 会被转码修复")
    print(f"  截断/损坏     : {len(truncated)} 张  <- 会被隔离")
    print(f"  跨类重复(组)  : {len(cross_class_dups)}  <- 只出清单, 需人工裁决")
    print(f"  同类重复(组)  : {len(same_class_dups)}  <- 会隔离多余副本")

    print(f"\n[3/3] 执行动作 ({mode_label}) ...")
    actions = []

    # A. 后缀不符 -> 抽中间帧转 JPEG, 原文件隔离
    for item in misnamed:
        src = Path(item["path"])
        new_name = src.with_suffix(f".{item['real_format'].lower()}").name
        dst_img = src.with_name(src.stem + "_frame.jpg")
        q_dst = quarantine / "misnamed" / item["cls"] / new_name
        actions.append({"kind": "misnamed", "src": str(src),
                        "convert_to": str(dst_img), "quarantine": str(q_dst)})
        print(f"  [格式] {item['cls']}/{src.name} "
              f"(真实={item['real_format']}, {item['n_frames']}帧) "
              f"-> 抽帧转 {dst_img.name}, 原图隔离")
        if args.apply:
            if convert_animated_to_jpeg(src, dst_img):
                q_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(q_dst))

    # B. 截断 -> 隔离
    for item in truncated:
        src = Path(item["path"])
        q_dst = quarantine / "truncated" / item["cls"] / src.name
        actions.append({"kind": "truncated", "src": str(src),
                        "quarantine": str(q_dst), "error": item["error"]})
        print(f"  [损坏] {item['cls']}/{src.name} -> 隔离")
        if args.apply:
            q_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(q_dst))

    # C. 同类重复 -> 保留第一个, 其余隔离
    for grp in same_class_dups:
        keep = grp["entries"][0]
        for dup in grp["entries"][1:]:
            src = Path(dup["path"])
            q_dst = quarantine / "same_class_dup" / dup["cls"] / src.name
            actions.append({"kind": "same_class_dup", "src": str(src),
                            "keep": keep["path"], "quarantine": str(q_dst)})
            print(f"  [重复] {dup['cls']}/{src.name} -> 隔离 (保留 {Path(keep['path']).name})")
            if args.apply:
                q_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(q_dst))

    # D. 跨类重复 -> 只出清单
    if cross_class_dups:
        print("\n  [!] 跨类重复 —— 同一张图被标成多个角色, 脚本不自动裁决:")
        for grp in cross_class_dups:
            names = " | ".join(f"{e['cls']}/{Path(e['path']).name}" for e in grp["entries"])
            print(f"      {names}")
        print("      -> 请人工确认正确归属类, 手动移除错误副本; "
              "在裁决前它们既造成标签矛盾, 也会导致切分泄漏。")

    report = {
        "timestamp": ts,
        "dataset_dir": str(root),
        "mode": "apply" if args.apply else "dry-run",
        "quarantine_dir": str(quarantine),
        "summary": {
            "classes": len(per_class), "total_images": total,
            "format_dist": dict(sorted(fmt_counter.items())),
            "mode_dist": dict(sorted(mode_counter.items())),
            "non_rgb": non_rgb,
            "misnamed": len(misnamed), "truncated": len(truncated),
            "cross_class_dup_groups": len(cross_class_dups),
            "same_class_dup_groups": len(same_class_dups),
        },
        "per_class_counts": dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
        "misnamed": misnamed,
        "truncated": truncated,
        "cross_class_dups": cross_class_dups,
        "same_class_dups": same_class_dups,
        "actions": actions,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print("\n" + "=" * 78)
    if args.apply:
        print(f"✅ 已执行。被移走的文件都在: {quarantine}")
        print("   若要回滚, 把隔离目录里的文件移回原类目录即可。")
    else:
        print("🔍 以上为预演, 未改动任何文件。确认无误后加 --apply 执行。")
    print(f"📄 JSON 报告: {report_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
