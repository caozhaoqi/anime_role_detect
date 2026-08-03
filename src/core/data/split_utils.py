"""
Grouped (character-level) dataset splitting utilities.

!!! MUST group by character to avoid leakage !!!
================================================
For anime/game CHARACTER RECOGNITION the "group" is the CHARACTER (the
sub-directory under ``data/final_dataset/<character>/``). Every image of a
given character MUST land in exactly ONE split (train / val / test). Never
split a character's own images across train and test — that is the data
leakage that inflated the old EfficientNet-B3 Top-1 from a real ~0.42 to a
fake ~0.84 ("training/test from the same source").

Use ``grouped_split`` to obtain leak-free index lists, or
``make_character_grouped_split`` to scan ``final_dataset`` and emit the same
``{train,val,test}.json`` manifest schema consumed by
``scripts/model_evaluation/train_clean_split.py``.

This module is intentionally torch-free so it can be imported and unit-tested
without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sklearn.model_selection import GroupShuffleSplit

ImageSample = Dict[str, object]
SplitName = str


def grouped_split(
    items: Sequence,
    groups: Sequence,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split ``items`` so that every group falls into exactly one split.

    Args:
        items: ordered collection of samples (paths, dicts, ...).
        groups: parallel sequence; the group identity for each item. For
            character recognition this is the character name. Two items with
            the same group value will NEVER end up in different splits.
        ratios: (train, val, test) fractions; must sum to ~1.0.
        seed: random seed for reproducible splits.

    Returns:
        ``(train_idx, val_idx, test_idx)`` — lists of integer positions into
        ``items``. Every index appears in exactly one list; every group
        appears in exactly one list.

    Raises:
        ValueError: if inputs are malformed or there are fewer than 2 groups
            (a split cannot be leak-free with a single group).
    """
    if not items:
        raise ValueError("items is empty")
    if len(items) != len(groups):
        raise ValueError("items and groups must have equal length")
    if len(ratios) != 3:
        raise ValueError("ratios must be a 3-tuple (train, val, test)")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    train_r, val_r, test_r = (float(x) for x in ratios)
    unique_groups: List = list(dict.fromkeys(groups))  # stable order, dedup
    n_groups = len(unique_groups)
    if n_groups < 2:
        raise ValueError(
            f"need at least 2 groups to split without leakage, got {n_groups}"
        )

    all_idx = list(range(len(items)))

    # Stage 1: train vs (val + test) — grouped.
    gss = GroupShuffleSplit(
        n_splits=1, test_size=(val_r + test_r), random_state=seed
    )
    train_idx, rest_idx = next(gss.split(all_idx, groups=list(groups)))

    # Stage 2: split the held-out groups into val vs test — also grouped.
    rest_groups = [groups[i] for i in rest_idx]
    rest_local = list(range(len(rest_idx)))
    if val_r + test_r <= 0:
        val_local, test_local = rest_local, []
    else:
        inner_test = test_r / (val_r + test_r)
        try:
            gss2 = GroupShuffleSplit(
                n_splits=1, test_size=inner_test, random_state=seed + 1
            )
            val_local, test_local = next(
                gss2.split(rest_local, groups=rest_groups)
            )
        except ValueError:
            # Degenerate case: the held-out set is a single group and cannot
            # be sub-split. Place it in the larger of val/test to keep the
            # "one group -> one split" guarantee intact.
            if test_r >= val_r:
                val_local, test_local = [], rest_local
            else:
                val_local, test_local = rest_local, []

    val_idx = [rest_idx[i] for i in val_local]
    test_idx = [rest_idx[i] for i in test_local]

    # Sanity: a group must not appear in two splits.
    _assert_no_group_leak(groups, train_idx, val_idx, test_idx)
    return list(train_idx), list(val_idx), list(test_idx)


def _assert_no_group_leak(
    groups: Sequence,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
) -> None:
    def chars_in(idxs: Sequence[int]):
        return {groups[i] for i in idxs}

    t, v, e = chars_in(train_idx), chars_in(val_idx), chars_in(test_idx)
    overlap = (t & v) | (t & e) | (v & e)
    if overlap:
        raise AssertionError(
            f"LEAKAGE: character(s) span multiple splits: {overlap}"
        )


def _scan_character_images(
    dataset_dir: Path, label_map: Optional[Dict[str, int]]
) -> Tuple[List[str], List[str], List[int]]:
    """Return (relative_paths, groups, labels) for every image under
    ``dataset_dir/<character>/``. Only characters present in ``label_map``
    are included (mirrors the old make_split.py behaviour)."""
    items: List[str] = []
    groups: List[str] = []
    labels: List[int] = []
    exts = (".jpg", ".jpeg", ".png", ".webp")
    for character in sorted(label_map or {}):
        char_dir = dataset_dir / character
        if not char_dir.is_dir():
            continue
        for f in sorted(char_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                items.append(f"{character}/{f.name}")
                groups.append(character)
                labels.append(label_map[character])
    return items, groups, labels


def make_character_grouped_split(
    dataset_dir,
    out_dir=None,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    label_map: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """Scan ``final_dataset`` and emit leak-free, character-grouped splits.

    Args:
        dataset_dir: path to ``data/final_dataset``.
        out_dir: if given, writes ``train.json`` / ``val.json`` / ``test.json``
            (list of ``{"path": "<character>/<file>", "label": <idx>}``) and a
            ``summary.json``. Same schema as the old make_split.py so
            ``train_clean_split.py`` keeps working unchanged.
        ratios: (train, val, test).
        seed: reproducibility seed.
        label_map: ``{character: class_idx}``; only these characters are used.

    Returns:
        dict with keys ``train``, ``val``, ``test`` (lists of samples) and
        ``summary``.
    """
    dataset_dir = Path(dataset_dir)
    if label_map is None:
        # Fallback: derive classes from subdirectories (0..N-1 by sorted name).
        label_map = {
            d.name: i
            for i, d in enumerate(sorted(p for p in dataset_dir.iterdir() if p.is_dir()))
        }

    items, groups, labels = _scan_character_images(dataset_dir, label_map)
    if not items:
        raise ValueError(f"no images found under {dataset_dir}")

    train_idx, val_idx, test_idx = grouped_split(
        items, groups, ratios=ratios, seed=seed
    )

    def build(idxs: Sequence[int]) -> List[ImageSample]:
        return [
            {"path": items[i], "label": labels[i]} for i in idxs
        ]

    splits = {
        "train": build(train_idx),
        "val": build(val_idx),
        "test": build(test_idx),
    }

    # Per-character accounting for the summary.
    per_class: Dict[str, Dict[str, int]] = {}
    for i in train_idx:
        per_class.setdefault(groups[i], {"total": 0, "train": 0, "val": 0, "test": 0})
        per_class[groups[i]]["train"] += 1
    for i in val_idx:
        per_class.setdefault(groups[i], {"total": 0, "train": 0, "val": 0, "test": 0})
        per_class[groups[i]]["val"] += 1
    for i in test_idx:
        per_class.setdefault(groups[i], {"total": 0, "train": 0, "val": 0, "test": 0})
        per_class[groups[i]]["test"] += 1

    summary = {
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "num_classes": len(label_map),
        "total_images": len(items),
        "split_counts": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "per_class": per_class,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train", "val", "test"):
            with open(out_dir / f"{name}.json", "w", encoding="utf-8") as fh:
                json.dump(splits[name], fh, ensure_ascii=False)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

    return {"train": splits["train"], "val": splits["val"], "test": splits["test"], "summary": summary}
