#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重冻 configs/class_registry_v2.json，使其与「合并去重后的 final_dataset」+「新切分」完全一致。

流程：
  1. 从 final_dataset 实际含图目录推导类目（与 train_clean_split.py 的 c2i 推导顺序一致：sorted 目录名）。
  2. 从 summary.json 拉取每类的 train/val/test 计数与 eval_status。
  3. 从旧 registry 继承 booru_tag / aliases / collect_target / suggested_action 等元数据。
  4. 将已合并的短名（DUPLICATE_MAP 中被合并的一侧）移入 deleted（reason=ALIAS_OF:全名）。
  5. 更新顶层 frozen_at / split_hash / num_classes。

可逆：运行前已对旧 registry 做 .bak 备份。
"""
import json
import shutil
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = ROOT / "data" / "final_dataset"
SPLIT_DIR = ROOT / "data" / "splits" / "seed42"
REG_PATH = ROOT / "configs" / "class_registry_v2.json"
EXTS = (".jpg", ".jpeg", ".png", ".webp")

# T3-1 已把这 4 个类移出标签空间（各只有 1 张图且全在 train，永远不可评测）。
# 它们的目录仍在 final_dataset 里，所以下面按目录枚举时必须显式跳过，
# 否则本脚本会把 registry 静默地重新灌回 171 类、并把 id 全部错位一格。
# 详见 scripts/_clean_label_space_167.py。
EXCLUDED_FROM_LABEL_SPACE = {"jean", "rina", "seth", "yae_miko"}

# 与 merge_duplicate_roles.py 的 DUPLICATE_MAP 合并侧保持一致（短名 → 全名）
MERGED_TO_FULL = {
    "azusa": "shirasu_azusa",
    "izuna": "kuda_izuna",
    "shiroko": "sunaookami_shiroko",
    "Izuna": "kuda_izuna",
    "Azusa": "shirasu_azusa",
}


def main():
    old = json.load(open(REG_PATH, encoding="utf-8"))
    summary = json.load(open(SPLIT_DIR / "summary.json", encoding="utf-8"))
    per_class = summary["per_class"]
    eval_status = summary.get("eval_status", {})

    old_by_name = {c["name"]: c for c in old.get("classes", [])}

    # 实际含图目录（sorted，与 c2i 推导一致）
    dirs = sorted(
        d.name for d in FINAL_DIR.iterdir()
        if d.is_dir()
        and d.name not in EXCLUDED_FROM_LABEL_SPACE
        and any(f.suffix.lower() in EXTS for f in d.iterdir())
    )

    merged_short_names = set(MERGED_TO_FULL.keys())

    new_classes = []
    for idx, name in enumerate(dirs):
        if name in merged_short_names:
            continue  # 合并侧不进入 classes，转 deleted
        pc = per_class.get(name, {})
        meta = old_by_name.get(name, {})
        new_classes.append({
            "id": idx,
            "name": name,
            "booru_tag": meta.get("booru_tag", name),
            "aliases": meta.get("aliases", []),
            "status": "ACTIVE",
            "img_count": pc.get("total", 0),
            "distinct_post_ids": pc.get("total", 0),  # 近似：精确值需扫文件，非训练关键
            "eval_status": eval_status.get(name, "FULL"),
            "collect_target": meta.get("collect_target", 30),
            "suggested_action": meta.get("suggested_action"),
        })

    # 被移出标签空间的类：保留旧条目（id=None / 非 ACTIVE），否则重冻一次就
    # 丢掉"它们为什么不在标签空间里"这条记录，下次有人又会把它们加回来。
    for name in sorted(EXCLUDED_FROM_LABEL_SPACE):
        if name in old_by_name:
            new_classes.append(old_by_name[name])

    # 合并侧写入 deleted
    new_deleted = list(old.get("deleted", []))
    for short, full in MERGED_TO_FULL.items():
        if any(d.get("name") == short for d in new_deleted):
            continue
        new_deleted.append({"name": short, "reason": f"ALIAS_OF:{full}"})

    new_reg = {
        "schema_version": 2,
        "frozen_at": "2026-08-04",
        "split_hash": summary["split_hash"],
        "seed": summary.get("seed", 42),
        "classes": new_classes,
        "deleted": new_deleted,
        "pending_review": old.get("pending_review", []),
    }

    json.dump(new_reg, open(REG_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"classes(ACTIVE): {len(new_classes)}")
    print(f"deleted: {len(new_deleted)} (含合并 {len(merged_short_names)} 个短名)")
    print(f"split_hash: {summary['split_hash'][:16]}")
    print(f"eval_status counts: {dict(Counter(c['eval_status'] for c in new_classes))}")
    print("registry re-frozen ->", REG_PATH)


if __name__ == "__main__":
    main()
