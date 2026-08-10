#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性迁移：把冻结切分的标签空间从 171 类收缩到 167 类（T3-1）。

为什么
------
jean / rina / seth / yae_miko 四个类在全库各只有 **1 张图且全部落在 train**
（val=0, test=0），summary.json 已把它们标成 ``TRAIN_ONLY`` /
``excluded_from_eval_characters``。它们占着分类头的 4 个输出维度却永远无法被
评测，把 Macro-F1 拉出一个 171 口径 / 167 口径的双重账本（v7 就是这么变成
"0.5401(171) vs 0.5533(167)" 两个数的）。v8 之前先把标签空间收干净。

做什么（**只删这 4 类，其余 167 类的 train/val/test 归属逐条不变**）
-------------------------------------------------------------------
1. 从 train/val/test.json 里删掉这 4 个类的样本行（实际只有 train 里 4 行）。
2. 把剩余 167 个类的整数 label 重编号为连续 0..166。这一步是**必须**的：
   切分 JSON 里的 label 是整数索引（不是类名），删类后 75/127/135/158 会留下
   空洞，而分类头输出只有 0..166 —— 不重编号就会出现 label 169 越界。
   重编号保序（旧 label 升序 = 类名字典序升序），所以新 idx = 旧 id 减去
   "被删掉且更小的 id 个数"。
3. 重算 split_hash（``split_utils._split_hash``：sorted train 路径的 sha256），
   并同步 summary.json 的 num_classes / per_class / eval_status / split_counts
   等计数字段。
4. 额外落一份 ``class_to_idx.json``（类名 -> 新索引，167 条）作为新标签空间的
   显式真源 —— 此前切分目录里没有这个文件，类名只能从样本 path 的父目录反推。

不做什么
--------
- **不重跑切分生成**：重跑会让其余 167 类重新分配，v7/v8 就没法对比了。
- 不重算 num_groups / content_merge_stats：它们依赖对全库 10569 张图重做
  sha256 union-find，本脚本无法在不读全量文件的前提下可靠推导。这两个字段
  保持 171 类时期的原值，并在 summary 的 ``label_space_cleanup`` 里显式声明
  "继承自 171 类切分、未重算"，避免被当成新值误读。

可逆：运行前必须已 ``cp -r data/splits/seed42 data/splits/seed42_backup_...``。
本脚本幂等性：对已经是 167 类的切分再跑一次会直接退出（无 4 类可删）。
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.core.data.split_utils import _split_hash  # noqa: E402

SPLIT_DIR = ROOT / "data" / "splits" / "seed42"
REG_PATH = ROOT / "configs" / "class_registry_v2.json"

# 剔除依据：per_class total==1 且 train=1/val=0/test=0，eval_status=TRAIN_ONLY
DROP_CLASSES = ("jean", "rina", "seth", "yae_miko")
SPLITS = ("train", "val", "test")


def main():
    summary = json.loads((SPLIT_DIR / "summary.json").read_text(encoding="utf-8"))
    rows = {s: json.loads((SPLIT_DIR / f"{s}.json").read_text(encoding="utf-8")) for s in SPLITS}

    # ── 类名 <-> 旧 label：直接从样本 path 的父目录取，不依赖任何外部映射文件 ──
    old_label_by_name = {}
    for split_rows in rows.values():
        for r in split_rows:
            name = r["path"].split("/", 1)[0]
            prev = old_label_by_name.setdefault(name, r["label"])
            if prev != r["label"]:
                raise SystemExit(f"[FATAL] 类 {name} 的 label 不唯一: {prev} vs {r['label']}")
    if len(old_label_by_name) != len(set(old_label_by_name.values())):
        raise SystemExit("[FATAL] 存在两个类名共用同一个 label，切分已损坏")

    missing = [c for c in DROP_CLASSES if c not in old_label_by_name]
    if missing:
        raise SystemExit(f"[FATAL] 待剔除的类不在切分里（是否已清理过？）: {missing}")

    # 剔除前做一次前提校验：这 4 个类必须确实只在 train 且各 1 张
    for c in DROP_CLASSES:
        pc = summary["per_class"].get(c, {})
        if (pc.get("val"), pc.get("test")) != (0, 0):
            raise SystemExit(f"[FATAL] {c} 在 val/test 里有样本 {pc}，不满足剔除前提，已中止")

    # ── 新标签空间：剩余类按旧 label 升序（= 类名字典序）重编号 0..N-1 ──
    keep = sorted(
        (lbl, name) for name, lbl in old_label_by_name.items() if name not in DROP_CLASSES
    )
    old2new = {lbl: i for i, (lbl, _) in enumerate(keep)}
    class_to_idx = {name: old2new[lbl] for lbl, name in keep}
    drop_labels = {old_label_by_name[c] for c in DROP_CLASSES}

    # ── 重写三个切分文件：删行 + 重编号 ──
    removed = Counter()
    new_rows = {}
    for s in SPLITS:
        kept = []
        for r in rows[s]:
            if r["label"] in drop_labels:
                removed[s] += 1
                continue
            kept.append({"path": r["path"], "label": old2new[r["label"]]})
        new_rows[s] = kept

    n_classes = len(class_to_idx)
    new_hash = _split_hash(new_rows["train"])

    # ── summary：只改随标签空间变化的字段，其余原样保留 ──
    summary["num_classes"] = n_classes
    summary["num_classes_with_images"] = n_classes
    summary["total_images"] = sum(len(v) for v in new_rows.values())
    summary["split_counts"] = {s: len(new_rows[s]) for s in SPLITS}
    summary["per_class"] = {k: v for k, v in summary["per_class"].items() if k not in DROP_CLASSES}
    summary["eval_status"] = {
        k: v for k, v in summary["eval_status"].items() if k not in DROP_CLASSES
    }
    summary["excluded_from_eval_characters"] = [
        c for c in summary.get("excluded_from_eval_characters", []) if c not in DROP_CLASSES
    ]
    summary["num_excluded_from_eval"] = len(summary["excluded_from_eval_characters"])
    summary["split_hash"] = new_hash
    summary["label_space_cleanup"] = {
        "applied": "T3-1",
        "from_num_classes": n_classes + len(DROP_CLASSES),
        "to_num_classes": n_classes,
        "dropped_classes": list(DROP_CLASSES),
        "drop_reason": "total==1 且全部在 train（val=0/test=0），永远不可评测",
        "relabeled": "剩余类保序重编号为连续 0..%d" % (n_classes - 1),
        "other_classes_reassigned": False,
        "previous_split_hash": "71b7101b47eb266579dea81bd837dded9c55e2e09c6e31124de1133abd733eeb",
        "stale_fields": {
            "num_groups": "继承自 171 类切分，未重算（需对全库重做 sha256 union-find）",
            "content_merge_stats": "同上，继承自 171 类切分",
        },
    }

    # 序列化格式与 split_utils.make_character_grouped_split 的写盘保持逐字节一致：
    # 三个切分文件紧凑无缩进，summary.json indent=2。
    for s in SPLITS:
        (SPLIT_DIR / f"{s}.json").write_text(
            json.dumps(new_rows[s], ensure_ascii=False), encoding="utf-8"
        )
    (SPLIT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (SPLIT_DIR / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── 注册表：id 是描述性字段（无消费方拿它当模型输出下标），但仍与标签空间
    #    对齐更安全。剩余类 id 重编号，4 个被剔除的类标记为非 ACTIVE 且 id=None。
    reg = json.loads(REG_PATH.read_text(encoding="utf-8"))
    for c in reg["classes"]:
        if c["name"] in DROP_CLASSES:
            c["id"] = None
            c["status"] = "EXCLUDED_FROM_LABEL_SPACE"
            c["eval_status"] = "NOT_IN_LABEL_SPACE"
            c["suggested_action"] = "补图到 >=3 张（含 val/test 各 >=1）后方可重新入选标签空间"
        else:
            c["id"] = class_to_idx[c["name"]]
    reg["split_hash"] = new_hash
    REG_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dropped classes : {list(DROP_CLASSES)} (labels {sorted(drop_labels)})")
    print(f"removed rows    : {dict(removed)}")
    print(f"num_classes     : {n_classes}")
    print(f"split_counts    : {summary['split_counts']}")
    print(f"new split_hash  : {new_hash}")
    print(f"wrote           : {SPLIT_DIR}/[train|val|test|summary|class_to_idx].json")
    print(f"registry        : {REG_PATH} (ACTIVE ids 重编号 0..{n_classes - 1})")


if __name__ == "__main__":
    main()
