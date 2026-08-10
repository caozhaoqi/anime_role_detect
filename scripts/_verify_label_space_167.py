#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 T3-1 清理结果：167 类 / 4 类消失 / 标签无越界 / 其余类分配零变化。

用法: ./.venv/bin/python scripts/_verify_label_space_167.py <backup_dir>

核心断言是最后一条：**逐类逐 split 比对新旧样本路径集合**。只要其余 167 类的
train/val/test 路径集合与备份逐条相同，就证明这次清理只是删了 4 行，没有触发
任何重新分配（这是 v7↔v8 可比的前提）。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.core.data.split_utils import _split_hash  # noqa: E402

SPLIT_DIR = ROOT / "data" / "splits" / "seed42"
DROPPED = {"jean", "rina", "seth", "yae_miko"}
SPLITS = ("train", "val", "test")

failures = []


def check(cond, label, detail=""):
    print(f"{'[PASS]' if cond else '[FAIL]'} {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        failures.append(label)


def load(dirpath):
    return {s: json.loads((Path(dirpath) / f"{s}.json").read_text(encoding="utf-8")) for s in SPLITS}


def by_class(rows_map):
    """{class_name: {split: set(paths)}}，类名取自 path 父目录（与 label 无关）。"""
    out = defaultdict(lambda: defaultdict(set))
    for s, rows in rows_map.items():
        for r in rows:
            out[r["path"].split("/", 1)[0]][s].add(r["path"])
    return out


def main():
    backup_dir = Path(sys.argv[1])
    new = load(SPLIT_DIR)
    old = load(backup_dir)
    summary = json.loads((SPLIT_DIR / "summary.json").read_text(encoding="utf-8"))
    c2i = json.loads((SPLIT_DIR / "class_to_idx.json").read_text(encoding="utf-8"))

    # 1. 标签空间大小 == 167
    check(len(c2i) == 167, "class_to_idx 键数 == 167", f"实际 {len(c2i)}")
    check(summary["num_classes"] == 167, "summary.num_classes == 167", str(summary["num_classes"]))

    # 2. 4 个类彻底消失（class_to_idx / summary / 样本行 三处都不能有）
    check(not (DROPPED & set(c2i)), "4 个类不在 class_to_idx 中")
    check(not (DROPPED & set(summary["per_class"])), "4 个类不在 summary.per_class 中")
    check(not (DROPPED & set(summary["eval_status"])), "4 个类不在 summary.eval_status 中")
    names_in_rows = {r["path"].split("/", 1)[0] for rows in new.values() for r in rows}
    check(not (DROPPED & names_in_rows), "4 个类在 train/val/test 中已无任何样本行")

    # 3. 标签值域连续、无越界：所有样本 label ∈ [0,166]，且 0..166 全部被占用
    all_labels = {r["label"] for rows in new.values() for r in rows}
    oob = sorted(l for l in all_labels if not (0 <= l <= 166))
    check(not oob, "所有样本 label ∈ [0,166]（无越界）", f"越界值 {oob[:5]}" if oob else "")
    check(
        set(c2i.values()) == set(range(167)),
        "class_to_idx 索引恰为连续 0..166",
    )
    check(all_labels <= set(range(167)), "样本 label 全部落在 class_to_idx 索引集合内")
    train_val_labels = {r["label"] for s in ("train", "val") for r in new[s]}
    check(
        train_val_labels == set(range(167)),
        "train∪val 覆盖全部 167 个索引（load_frozen_split 的 num_classes 即为 167）",
        f"实际 {len(train_val_labels)}",
    )

    # 4. label <-> 类名 一一对应且与 class_to_idx 一致
    mismatch = [
        r["path"] for rows in new.values() for r in rows
        if c2i.get(r["path"].split("/", 1)[0]) != r["label"]
    ]
    check(not mismatch, "每条样本的 label 与 class_to_idx[类名] 一致", f"{len(mismatch)} 条不符")

    # 5. split_hash 与新 train 内容自洽
    check(
        summary["split_hash"] == _split_hash(new["train"]),
        "summary.split_hash == sha256(sorted train paths)",
        summary["split_hash"][:16],
    )
    check(
        summary["split_hash"] != "71b7101b47eb266579dea81bd837dded9c55e2e09c6e31124de1133abd733eeb",
        "split_hash 已更新（不再是 v7 的 171 类 hash）",
    )

    # 6. ★核心★ 其余 167 类的每个 split 路径集合与备份逐条相同
    old_by, new_by = by_class(old), by_class(new)
    check(
        set(old_by) - DROPPED == set(new_by),
        "类名集合 == 备份类名集合 - 4",
        f"old {len(old_by)} -> new {len(new_by)}",
    )
    diff = []
    for name in set(new_by):
        for s in SPLITS:
            if old_by[name][s] != new_by[name][s]:
                diff.append(f"{name}/{s}: {len(old_by[name][s])} -> {len(new_by[name][s])}")
    check(not diff, "其余 167 类在 train/val/test 的样本路径集合逐条未变", "; ".join(diff[:5]))

    # 7. 总量：只少了 4 张 train 图
    old_n = {s: len(old[s]) for s in SPLITS}
    new_n = {s: len(new[s]) for s in SPLITS}
    check(
        new_n == {"train": old_n["train"] - 4, "val": old_n["val"], "test": old_n["test"]},
        "样本总数仅 train -4，val/test 不变",
        f"{old_n} -> {new_n}",
    )
    check(summary["split_counts"] == new_n, "summary.split_counts 与实际行数一致")
    check(summary["total_images"] == sum(new_n.values()), "summary.total_images 与实际一致")

    print()
    if failures:
        print(f"[RESULT] FAILED {len(failures)} 项: {failures}")
        return 1
    print("[RESULT] ALL PASS — 167 类标签空间干净、连续、无越界，其余类分配零变化")
    return 0


if __name__ == "__main__":
    sys.exit(main())
