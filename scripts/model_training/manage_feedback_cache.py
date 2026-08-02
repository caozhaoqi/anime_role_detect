#!/usr/bin/env python3
"""反馈原图缓存清理脚本（纯标准库，无 torch 依赖）。

契约（与训练脚本约定，勿改 schema）：
  - 反馈原图落盘到 data/feedback_images/<recognition_id>.jpg
  - 消费事实源：data/feedback_images/.consumed_manifest.json
        JSON 对象，键为 recognition_id，值为
        {consumed_at, consumed_by, corrected_label, source_jsonl, image_ref}
        该 manifest 由训练脚本在"训练成功且产物落盘后"原子写入。
  - 安全铁律：rid 不在 manifest 中的图片，无论缓存多满，绝不删除。
              不删 manifest 本身，不碰 data/feedback_images/ 之外的任何文件。

默认 dry-run（只收集、不删），--force/--yes 才真正删除；--dry-run 为显式别名
（与 --force 同传时 dry-run 优先）。默认 feedback-dir / manifest 已按仓库根
（脚本上溯三级）解析为绝对路径，无论从哪个 cwd 运行都指向同一份文件，与训练脚本一致。
"""
from __future__ import annotations

import argparse
import sys
import os
import json
import gzip
import shutil
import hashlib
import re
from collections import defaultdict
from datetime import datetime, timezone

DEFAULT_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
DEFAULT_MAX_FILES = 5000
MANIFEST_NAME = ".consumed_manifest.json"
JSONL_RE = re.compile(r"^feedback_\d{4}-\d{2}-\d{2}\.jsonl$")


def _project_root() -> str:
    """仓库根：scripts/model_training/manage_feedback_cache.py 上溯三级。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_consumed_manifest(path: str) -> dict:
    """读取 .consumed_manifest.json；不存在返回 {}；解析失败抛出清晰错误。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"消费清单解析失败 {path}: {e}") from e
    except OSError as e:
        raise ValueError(f"消费清单读取失败 {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"消费清单格式错误（应为 JSON 对象，键为 recognition_id）: {path}")
    return data


def scan_feedback_images(directory: str) -> list:
    """列出 directory 下所有 *.jpg（排除隐藏文件与 manifest），每项 {rid,path,size,mtime}。"""
    results = []
    if not os.path.isdir(directory):
        return results
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        if not name.lower().endswith(".jpg"):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        rid = name[:-4]  # 去掉 ".jpg"
        results.append({
            "rid": rid,
            "path": path,
            "size": st.st_size,
            "mtime": st.st_mtime,
        })
    return results


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def select_for_eviction(images, consumed, max_bytes, max_files, dedup=False) -> list:
    """计算待删列表。

    规则：
      - 仅保留 rid ∈ consumed 的候选；
      - dedup=True：对候选算 sha256，同 hash 仅留 mtime 最新的一份（其余多余副本也进候选）；
      - 候选按 mtime 升序（最旧优先）排序；
      - 从头（最旧）累加快照，直到剩余数量 < max_files 且 剩余大小 < max_bytes 为止；
        超出部分进入待删列表。若当前已低于上限，待删列表为空。
    返回待删列表，每项含 {rid, path, size}。
    """
    # 只接受已被消费的图片（安全铁律：未消费绝不入选）
    pool = [img for img in images if img["rid"] in consumed]
    if not pool:
        return []

    if dedup:
        groups = defaultdict(list)
        for img in pool:
            groups[_sha256(img["path"])].append(img)
        keepers = set()
        for members in groups.values():
            if len(members) > 1:
                newest = max(members, key=lambda x: x["mtime"])
                keepers.add(newest["rid"])
        #  keeper（每组最新一份）永不进入待删候选；多余副本参与淘汰
        pool = [img for img in pool if img["rid"] not in keepers]

    pool_sorted = sorted(pool, key=lambda x: x["mtime"])  # 最旧优先
    total = len(pool_sorted)
    total_size = sum(img["size"] for img in pool_sorted)

    # 已低于上限：不删任何文件
    if total <= max_files and total_size <= max_bytes:
        return []

    to_delete = []
    deleted_size = 0
    for img in pool_sorted:
        to_delete.append(img)
        deleted_size += img["size"]
        remaining = total - len(to_delete)
        remaining_size = total_size - deleted_size
        if remaining < max_files and remaining_size < max_bytes:
            break

    return [{"rid": i["rid"], "path": i["path"], "size": i["size"]} for i in to_delete]


def evict(selected, dry_run=True, force=False) -> tuple:
    """删除 selected 中的文件。默认 dry_run（只打印）；仅 force=True 且非 dry_run 才真正删除。"""
    deleted = []
    freed_bytes = 0
    actually_delete = bool(force) and not bool(dry_run)
    for item in selected:
        if actually_delete:
            _err(f"[DELETE] {item['path']} ({item['size']} bytes)")
            try:
                os.remove(item["path"])
                deleted.append(item["path"])
                freed_bytes += item["size"]
            except OSError as e:
                _err(f"[ERROR] 无法删除 {item['path']}: {e}")
        else:
            _err(f"[DRY-RUN] would delete {item['path']} ({item['size']} bytes)")
    return deleted, freed_bytes


def archive_jsonl(log_dir, archive_dir, manifest, older_than_days=0, force=False) -> list:
    """可选：扫描 feedback_<date>.jsonl，若文件内**所有** recognition_id 都已在 manifest 中，
    则 gzip 归档到 archive_dir（未全消费的不动）。older_than_days>0 时只处理早于该天数的文件。
    返回被归档（或拟归档）的文件名列表。默认 dry-run，仅 force=True 才真正写盘并移除原文件。
    """
    if older_than_days <= 0:
        return []
    archived = []
    if not os.path.isdir(log_dir):
        return archived
    os.makedirs(archive_dir, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    for name in sorted(os.listdir(log_dir)):
        if not JSONL_RE.match(name):
            continue
        path = os.path.join(log_dir, name)
        # 解析文件名中的日期，按 older_than_days 过滤
        try:
            fdate = datetime.strptime(name[len("feedback_"):-len(".jsonl")], "%Y-%m-%d").date()
        except ValueError:
            continue
        if (today - fdate).days < older_than_days:
            continue
        rids = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rids.append(rec.get("recognition_id"))
        except OSError as e:
            _err(f"[ERROR] 无法读取 {path}: {e}")
            continue
        # 仅当所有 recognition_id 都已消费才归档
        if rids and all(r in manifest for r in rids if r is not None):
            arc_name = name + ".gz"
            arc_path = os.path.join(archive_dir, arc_name)
            if force:
                _err(f"[ARCHIVE] {path} -> {arc_path}")
                with open(path, "rb") as f_in, gzip.open(arc_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                try:
                    os.remove(path)
                except OSError as e:
                    _err(f"[ERROR] 归档后无法移除原文件 {path}: {e}")
            else:
                _err(f"[DRY-RUN] would archive {path} -> {arc_path}")
            archived.append(name)
    return archived


def build_summary(scanned, candidates, selected, deleted, freed_bytes) -> dict:
    return {
        "scanned": scanned,
        "candidates": candidates,
        "to_delete": len(selected),
        "bytes_to_free": sum(i["size"] for i in selected),
        "deleted": len(deleted),
        "freed_bytes": freed_bytes,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="反馈原图缓存清理（默认 dry-run，只收集不删除）",
    )
    project_root = _project_root()
    parser.add_argument("--feedback-dir", default=os.path.join(project_root, "data", "feedback_images"),
                        help="反馈原图目录（默认 <仓库根>/data/feedback_images，已按仓库根解析，与训练脚本一致）")
    parser.add_argument("--manifest", default=None,
                        help="消费清单路径（默认 <feedback-dir>/.consumed_manifest.json）")
    parser.add_argument("--max-bytes", type=int, default=None,
                        help=f"缓存体积上限字节数（默认 {DEFAULT_MAX_BYTES}，可由环境变量 FEEDBACK_CACHE_MAX_BYTES 覆盖）")
    parser.add_argument("--max-files", type=int, default=None,
                        help=f"缓存文件数上限（默认 {DEFAULT_MAX_FILES}，可由环境变量 FEEDBACK_CACHE_MAX_FILES 覆盖）")
    parser.add_argument("--dedup", action="store_true", help="开启 sha256 去重（同内容仅保留最新一份）")
    parser.add_argument("--archive-older-than", type=int, default=0,
                        help="归档早于 N 天的全消费日志（默认 0=不归档）")
    parser.add_argument("--dry-run", action="store_true",
                        help="显式声明 dry-run（脚本默认即 dry-run，只收集不删除）；与 --force 同时传入时 dry-run 优先")
    parser.add_argument("--force", "--yes", dest="force", action="store_true",
                        help="真正删除/归档（默认 dry-run）")
    parser.add_argument("--report", default=None, help="将 JSON 摘要写入该路径")
    args = parser.parse_args(argv)

    # env 提供默认值，CLI 可覆盖
    max_bytes = args.max_bytes if args.max_bytes is not None else \
        int(os.environ.get("FEEDBACK_CACHE_MAX_BYTES", DEFAULT_MAX_BYTES))
    max_files = args.max_files if args.max_files is not None else \
        int(os.environ.get("FEEDBACK_CACHE_MAX_FILES", DEFAULT_MAX_FILES))
    manifest_path = args.manifest or os.path.join(args.feedback_dir, MANIFEST_NAME)

    # dry-run 是默认行为；显式 --dry-run 或（非 --force）均为 dry-run；二者同传时 dry-run 优先
    dry_run = args.dry_run or not args.force

    try:
        consumed = load_consumed_manifest(manifest_path)
        if not os.path.exists(manifest_path):
            print(f"[INFO] 消费清单不存在，视为空（不删任何图）: {manifest_path}")
        images = scan_feedback_images(args.feedback_dir)
        candidates = sum(1 for img in images if img["rid"] in consumed)
        selected = select_for_eviction(images, consumed, max_bytes, max_files, dedup=args.dedup)

        deleted, freed_bytes = evict(selected, dry_run=dry_run, force=args.force)

        archived = []
        if args.archive_older_than > 0:
            archive_dir = os.path.join(args.feedback_dir, "..", "feedback", "archive")
            archive_dir = os.path.normpath(archive_dir)
            archived = archive_jsonl(
                os.path.join(os.path.dirname(os.path.normpath(args.feedback_dir)), "feedback"),
                archive_dir, consumed, args.archive_older_than, force=args.force,
            )

        summary = build_summary(len(images), candidates, selected, deleted, freed_bytes)
        summary["dry_run"] = dry_run
        summary["archived"] = archived
        print(json.dumps(summary, ensure_ascii=False))
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        return 0
    except Exception as e:  # noqa: BLE001
        _err(f"[FATAL] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
