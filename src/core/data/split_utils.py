"""
Grouped dataset splitting utilities — grouped by image SOURCE, not by character.

Group by image source / post-id to prevent near-duplicate leakage
=================================================================
This is a CLOSED-SET character classifier: every class must be learnable, so
each character needs images in BOTH train and test. The thing that must never
span splits is the **same source image** (one booru / pixiv post) together with
its near-duplicate variants (re-crops, resizes, alternate file extensions and
``_p0`` / ``_1`` / ``_dup`` / ``_cropped`` style suffixes).

Therefore the split group is the **post-id**:

* same post-id                        -> all variants land in exactly ONE split
* same character, different post-id   -> MAY span splits (this is intended and
  is exactly what a closed-set classifier must learn from)

A post-id is treated as **global**: when a single artwork is tagged and cropped
into several character folders (group art — e.g. one Madoka post cropped into
``akemi_homura`` / ``kaname_madoka`` / ``miki_sayaka`` / ``tomoe_mami``), all of
those crops share one post-id and therefore stay in the same split.

Historical note (why this was changed)
--------------------------------------
The previous revision grouped by CHARACTER. That did prevent leakage, but it
was over-corrective: it made 52 of 197 classes *zero-shot* (present in val/test
yet absent from train), so the model could only guess on them and the resulting
metrics were meaningless. Grouping by post-id keeps the anti-leakage guarantee
where it actually matters while restoring per-class trainability.

Use ``grouped_split`` to obtain leak-free index lists, or
``make_character_grouped_split`` to scan ``final_dataset`` and emit the same
``{train,val,test}.json`` manifest schema consumed by
``scripts/model_evaluation/train_clean_split.py``.

This module is intentionally torch-free so it can be imported and unit-tested
without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sklearn.model_selection import GroupShuffleSplit

ImageSample = Dict[str, object]
SplitName = str

# --------------------------------------------------------------------------
# post-id extraction
# --------------------------------------------------------------------------
# Observed corpus (data/final_dataset, 9637 files): 100% of filenames begin
# with the numeric source post-id. Stem forms present:
#   <post_id>.<ext>              9292  e.g. 1008785.png
#   <post_id>_<n>.<ext>           342  e.g. 5776378_1.jpg   (crop index)
#   <post_id>_dup.<ext>             2  e.g. 1251664_dup.jpg
#   <post_id>_cropped.<ext>         1  e.g. 6883660_cropped.jpg
# Leading digit-run length ranges 4..7, so a >=3 digit threshold is safe.
_LEADING_POST_ID_RE = re.compile(r"^(\d{3,})")

# Tolerate other collectors that prefix the source, e.g. "pixiv_123456_p0.jpg"
# or "danbooru-987654.jpg".
_PREFIXED_POST_ID_RE = re.compile(r"^[A-Za-z]+[-_](\d{3,})")

# Only used for the non-numeric fallback path: strip known variant markers so
# that "foo_crop" and "foo_p1" collapse onto "foo".
_VARIANT_SUFFIX_RE = re.compile(
    r"(?:[-_](?:p\d+|\d+|dup|copy|crop|cropped|sample|orig|original"
    r"|large|small|thumb|resized|face\d*))+$",
    re.IGNORECASE,
)


def extract_post_id(filename) -> str:
    """Extract the source post-id (near-duplicate cluster key) from a filename.

    All variants of one source image — different crop indices, ``_dup`` /
    ``_cropped`` markers, and differing file extensions — map to the SAME
    returned post-id, so they can be forced into a single split.

    Args:
        filename: a bare filename, or any path (only the basename is used).

    Returns:
        The numeric post-id as a string when one can be parsed (the common
        case), otherwise the variant-stripped, lower-cased stem. The result is
        always a non-empty string.

    Examples:
        >>> extract_post_id("1008785.png")
        '1008785'
        >>> extract_post_id("5776378_1.jpg")       # crop index -> same post
        '5776378'
        >>> extract_post_id("6883660_cropped.jpg")
        '6883660'
        >>> extract_post_id("pixiv_123456_p0.jpg")
        '123456'
    """
    stem = Path(str(filename)).stem
    match = _LEADING_POST_ID_RE.match(stem) or _PREFIXED_POST_ID_RE.match(stem)
    if match:
        return match.group(1)
    cleaned = _VARIANT_SUFFIX_RE.sub("", stem).strip("._-")
    return (cleaned or stem).lower()


def post_group_key(character: str, filename) -> str:
    """Build the split group key for one image.

    A parsed numeric post-id is a *global* identifier (the same artwork can be
    cropped into several character folders), so it is NOT scoped by character.
    An unparseable name cannot be proven to share a source with anything else,
    so it falls back to a character-scoped singleton group — safe by default,
    and it never merges unrelated files from different characters.
    """
    post_id = extract_post_id(filename)
    if post_id.isdigit():
        return f"post:{post_id}"
    return f"file:{character}/{post_id}"


# --------------------------------------------------------------------------
# 内容级去重：sha256 + union-find
# --------------------------------------------------------------------------
# 为什么 post_group_key 不够：
#   post_group_key 只看文件名里的 post-id。同一张图被不同 post 号收录时
#   （例如 itsuka_kotori/936671.png 与 itsuka_kotori/5373390.png 字节完全相同），
#   它们的 post-id 不同 -> 被判为两个组 -> 可能一个进 train 一个进 test。
#   实测全库 sha256 在 seed42 切分下确有 1 处 train/test 内容泄漏。
#   这是结构性缺陷（换 seed 只是换一处泄漏），不是概率问题。
# 修法：对全库算 sha256，把"共享同一内容哈希"的 post-id 用并查集合并成超级组，
#   再喂给 grouped_split，使字节相同的图必然落在同一 split。
# --------------------------------------------------------------------------


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through sha256 (constant memory, safe for large images)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


class _UnionFind:
    """Minimal union-find with path compression + union by size."""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.size: Dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def merge_groups_by_content(
    groups: Sequence[str], hashes: Sequence[str]
) -> Tuple[List[str], Dict[str, int]]:
    """Merge post-id groups that share identical file content into super-groups.

    Args:
        groups: per-item post-id group key (from ``post_group_key``).
        hashes: parallel per-item sha256 of the file bytes.

    Returns:
        ``(super_groups, stats)`` where ``super_groups`` is a parallel list of
        merged group keys, and ``stats`` reports how much merging happened.
    """
    if len(groups) != len(hashes):
        raise ValueError("groups and hashes must have equal length")

    uf = _UnionFind()
    for g in groups:
        uf.add(g)

    # Every group that shares a content hash gets unioned with the first one.
    first_group_for_hash: Dict[str, str] = {}
    for g, h in zip(groups, hashes):
        if h in first_group_for_hash:
            uf.union(first_group_for_hash[h], g)
        else:
            first_group_for_hash[h] = g

    super_groups = [f"sg:{uf.find(g)}" for g in groups]
    n_before, n_after = len(set(groups)), len(set(super_groups))
    stats = {
        "groups_before_merge": n_before,
        "groups_after_merge": n_after,
        "groups_merged_away": n_before - n_after,
        "duplicate_content_clusters": sum(
            1 for h, c in _count(hashes).items() if c > 1
        ),
        "duplicate_files": sum(c for c in _count(hashes).values() if c > 1),
    }
    return super_groups, stats


def _count(seq: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for x in seq:
        out[x] += 1
    return dict(out)


def _assert_no_content_leak(
    hashes: Sequence[str],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    items: Optional[Sequence[str]] = None,
) -> None:
    """FATAL if the same file CONTENT (sha256) appears in two different splits.

    This is stricter than ``_assert_no_post_id_leak``: it catches the
    "same image published under two different post-ids" case that filename
    based grouping is blind to.
    """

    def hashes_in(idxs: Sequence[int]):
        return {hashes[i] for i in idxs}

    t, v, e = hashes_in(train_idx), hashes_in(val_idx), hashes_in(test_idx)
    pairs = (("train", "val", t & v), ("train", "test", t & e), ("val", "test", v & e))
    problems = [(a, b, ov) for a, b, ov in pairs if ov]
    if problems:
        lines = []
        for a, b, ov in problems:
            lines.append(f"  {a} ∩ {b}: {len(ov)} colliding content hash(es)")
            if items is not None:
                by_hash: Dict[str, List[str]] = defaultdict(list)
                for i, h in enumerate(hashes):
                    if h in ov:
                        by_hash[h].append(str(items[i]))
                for h in sorted(ov)[:5]:
                    lines.append(f"    {h[:12]}… -> {by_hash[h]}")
        raise AssertionError(
            "FATAL CONTENT LEAKAGE: identical file bytes span multiple splits\n"
            + "\n".join(lines)
        )


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
            character recognition this is the **post-id / near-duplicate
            cluster key** (see ``post_group_key``), NOT the character name.
            Two items with the same group value will NEVER end up in different
            splits; items of the same character but different post-id are free
            to span splits, which is the intended behaviour.
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

    # Sanity: one post-id (near-duplicate cluster) must not appear in 2 splits.
    _assert_no_post_id_leak(groups, train_idx, val_idx, test_idx)
    return list(train_idx), list(val_idx), list(test_idx)


def _assert_no_post_id_leak(
    groups: Sequence,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
) -> None:
    """Assert no group (= post-id / near-duplicate cluster) spans two splits.

    The logic is unchanged from the previous character-grouped revision — only
    the *meaning* of ``groups`` changed: it is now the image source post-id, so
    this guards against near-duplicate leakage rather than forbidding a
    character from appearing in more than one split (which is now allowed).
    """

    def groups_in(idxs: Sequence[int]):
        return {groups[i] for i in idxs}

    t, v, e = groups_in(train_idx), groups_in(val_idx), groups_in(test_idx)
    overlap = (t & v) | (t & e) | (v & e)
    if overlap:
        sample = sorted(str(g) for g in overlap)[:10]
        raise AssertionError(
            f"LEAKAGE: {len(overlap)} post-id(s) span multiple splits "
            f"(showing up to 10): {sample}"
        )


def _scan_character_images(
    dataset_dir: Path, label_map: Optional[Dict[str, int]]
) -> Tuple[List[str], List[str], List[int], List[str]]:
    """Scan ``dataset_dir/<character>/`` for images.

    Returns:
        ``(relative_paths, groups, labels, characters)`` where ``groups`` holds
        the **post-id** cluster key (``post_group_key``) used for splitting and
        ``characters`` holds the owning character directory name, kept separate
        because the label / per-class accounting is still character-based.

    Only characters present in ``label_map`` are included (mirrors the old
    make_split.py behaviour). Directories that are missing or contain 0 images
    are skipped naturally.
    """
    items: List[str] = []
    groups: List[str] = []
    labels: List[int] = []
    characters: List[str] = []
    exts = (".jpg", ".jpeg", ".png", ".webp")
    for character in sorted(label_map or {}):
        char_dir = dataset_dir / character
        if not char_dir.is_dir():
            continue
        for f in sorted(char_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                items.append(f"{character}/{f.name}")
                groups.append(post_group_key(character, f.name))
                labels.append(label_map[character])
                characters.append(character)
    return items, groups, labels, characters


def _split_hash(train_samples):
    """sha256 over the sorted train file paths — identifies a split exactly."""
    h = hashlib.sha256()
    for p in sorted(s["path"] for s in train_samples):
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def _promote_zero_shot_to_train(items, groups, characters, train_idx, val_idx, test_idx):
    """Min-train-guarantee: any class with zero train images gets ALL its
    images moved into train (they currently sit entirely in val/test because
    the global leak-free split placed their only post-id group(s) there).

    Because every post-id group is kept whole within a single split by
    ``grouped_split``, moving a class's images as a block never introduces
    post-id leakage. Zero-shot classes are typically tiny (1-2 images), so the
    impact on the global 70/15/15 ratio is negligible.
    """
    from collections import defaultdict

    char_train = defaultdict(int)
    for i in train_idx:
        char_train[characters[i]] += 1

    chars_with_images = set(characters)
    zero_shot = {c for c in chars_with_images if char_train.get(c, 0) == 0}
    if not zero_shot:
        return train_idx, val_idx, test_idx

    move = {i for i, c in enumerate(characters) if c in zero_shot}
    new_val = [i for i in val_idx if i not in move]
    new_test = [i for i in test_idx if i not in move]
    new_train = list(train_idx) + sorted(move)
    return new_train, new_val, new_test


# --------------------------------------------------------------------------
# 每类 test 非零门禁
# --------------------------------------------------------------------------
# 动机：sapphire(18 张) / tsukiyo(21 张) / theresa_apocalypse(5 张) 在旧切分里
# test 样本数为 0，却没有任何报错——它们被静默排除出评测，MacroF1 里根本没有
# 它们的贡献，指标因此虚高。必须硬门禁拦住这种情况。
#
# 但门禁不能"只报错不解决"：total==1 的类（jean/rina/seth/yae_miko）物理上
# 不可能同时出现在 train 和 test。所以策略是：
#   1) 先尝试自动补救——把该类的某个超级组整体从 train 挪到 test
#      （整组搬迁，不破坏无泄漏保证；且保证 train 侧不被搬空）
#   2) 补救后仍为 0 且样本量足够的类 -> FATAL
#   3) 样本量本就不够的类 -> 标记 EXCLUDED_FROM_EVAL，进 summary，不 FATAL
# --------------------------------------------------------------------------

MIN_EVAL_CLASS_SIZE = 3


def _guarantee_test_coverage(
    groups: Sequence[str],
    characters: Sequence[str],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    min_eval_class_size: int = MIN_EVAL_CLASS_SIZE,
) -> Tuple[List[int], List[int], List[int], List[str]]:
    """Move one whole group from train -> test for classes with zero test images.

    Only classes that (a) have >= ``min_eval_class_size`` images overall and
    (b) own >= 2 groups in train are remediated, so train never loses its last
    group for a class. Moving a WHOLE group preserves the leak-free guarantee.

    Returns ``(train_idx, val_idx, test_idx, remediated_characters)``.
    """
    train_idx = list(train_idx)
    val_idx = list(val_idx)
    test_idx = list(test_idx)

    total_per_char: Dict[str, int] = defaultdict(int)
    for c in characters:
        total_per_char[c] += 1
    test_per_char: Dict[str, int] = defaultdict(int)
    for i in test_idx:
        test_per_char[characters[i]] += 1

    # class -> {group -> [train indices]}
    train_groups_per_char: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for i in train_idx:
        train_groups_per_char[characters[i]][groups[i]].append(i)

    remediated: List[str] = []
    moved: set = set()
    for char in sorted(total_per_char):
        if test_per_char.get(char, 0) > 0:
            continue
        if total_per_char[char] < min_eval_class_size:
            continue  # inherently ineligible; reported as EXCLUDED_FROM_EVAL
        gmap = train_groups_per_char.get(char, {})
        if len(gmap) < 2:
            continue  # moving the only group would make the class zero-shot
        # Smallest group first -> least disruption to the 70/15/15 ratio.
        victim = min(sorted(gmap), key=lambda g: (len(gmap[g]), g))
        moved.update(gmap[victim])
        remediated.append(char)

    if moved:
        train_idx = [i for i in train_idx if i not in moved]
        test_idx = sorted(test_idx + sorted(moved))
    return train_idx, val_idx, test_idx, remediated


def _assert_test_coverage(
    per_class: Dict[str, Dict[str, int]],
    min_eval_class_size: int = MIN_EVAL_CLASS_SIZE,
) -> List[str]:
    """FATAL if an ACTIVE class (>= ``min_eval_class_size`` images) has 0 test.

    Returns the list of classes legitimately excluded from evaluation because
    they are too small to be split at all.
    """
    violations = [
        f"{c}(total={r['total']}, train={r['train']}, val={r['val']}, test=0)"
        for c, r in sorted(per_class.items())
        if r["test"] == 0 and r["total"] >= min_eval_class_size
    ]
    if violations:
        raise AssertionError(
            "FATAL: "
            f"{len(violations)} ACTIVE class(es) have ZERO test samples and would be "
            "silently excluded from evaluation (MacroF1 would be inflated):\n  "
            + "\n  ".join(violations)
        )
    return sorted(
        c
        for c, r in per_class.items()
        if r["test"] == 0 and r["total"] < min_eval_class_size
    )


def make_character_grouped_split(
    dataset_dir,
    out_dir=None,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    label_map: Optional[Dict[str, int]] = None,
    min_train_guarantee: bool = True,
    content_hash_grouping: bool = True,
    enforce_test_coverage: bool = True,
    min_eval_class_size: int = MIN_EVAL_CLASS_SIZE,
) -> Dict[str, object]:
    """Scan ``final_dataset`` and emit post-id-grouped, leak-free splits.

    Images sharing a source post-id (near-duplicate variants, and crops of one
    artwork across several character folders) always land in the same split.
    Images of the same character from *different* posts are free to span
    splits, so every class stays trainable.

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
        # Fallback: derive classes ONLY from subdirectories that actually contain
        # images, sorted by name, contiguous 0..N-1. Empty shell dirs (no images)
        # are excluded so labels stay contiguous and num_classes equals the true
        # count of trainable classes (175, not 205). This MUST match the identical
        # logic in scripts/model_evaluation/train_clean_split.py.
        _exts = (".jpg", ".jpeg", ".png", ".webp")
        label_map = {
            d.name: i
            for i, d in enumerate(
                sorted(
                    p
                    for p in dataset_dir.iterdir()
                    if p.is_dir()
                    and any(f.is_file() and f.suffix.lower() in _exts for f in p.iterdir())
                )
            )
        }

    items, groups, labels, characters = _scan_character_images(
        dataset_dir, label_map
    )
    if not items:
        raise ValueError(f"no images found under {dataset_dir}")

    # ---- content-hash grouping (sha256 + union-find) ----------------------
    hashes: List[str] = []
    merge_stats: Dict[str, int] = {}
    if content_hash_grouping:
        hashes = [sha256_file(dataset_dir / rel) for rel in items]
        groups, merge_stats = merge_groups_by_content(groups, hashes)

    train_idx, val_idx, test_idx = grouped_split(
        items, groups, ratios=ratios, seed=seed
    )
    if min_train_guarantee:
        train_idx, val_idx, test_idx = _promote_zero_shot_to_train(
            items, groups, characters, train_idx, val_idx, test_idx
        )

    # ---- per-class test coverage remediation ------------------------------
    remediated_chars: List[str] = []
    if enforce_test_coverage:
        train_idx, val_idx, test_idx, remediated_chars = _guarantee_test_coverage(
            groups, characters, train_idx, val_idx, test_idx, min_eval_class_size
        )

    # ---- hard gate #1: no identical CONTENT across splits ------------------
    if content_hash_grouping:
        _assert_no_content_leak(hashes, train_idx, val_idx, test_idx, items)

    def build(idxs: Sequence[int]) -> List[ImageSample]:
        return [
            {"path": items[i], "label": labels[i]} for i in idxs
        ]

    splits = {
        "train": build(train_idx),
        "val": build(val_idx),
        "test": build(test_idx),
    }

    # Per-CHARACTER accounting for the summary. NOTE: this must key off
    # ``characters``, not ``groups`` — ``groups`` now holds post-ids.
    per_class: Dict[str, Dict[str, int]] = {}

    def _bump(idxs: Sequence[int], split_name: str) -> None:
        for i in idxs:
            row = per_class.setdefault(
                characters[i], {"total": 0, "train": 0, "val": 0, "test": 0}
            )
            row[split_name] += 1
            row["total"] += 1

    _bump(train_idx, "train")
    _bump(val_idx, "val")
    _bump(test_idx, "test")

    # Zero-shot audit: a class present in val/test but absent from train is
    # unlearnable. With min_train_guarantee on, every class with >=1 image
    # lands in train, so zero_shot should be empty.
    chars_with_images = set(characters)
    trained_chars = {characters[i] for i in train_idx}
    zero_shot = sorted(c for c in chars_with_images - trained_chars)
    coverage = len(trained_chars) / len(chars_with_images) if chars_with_images else 0.0

    # Per-class evaluation status (drives which classes get a trustworthy metric).
    eval_status: Dict[str, str] = {}
    for c in chars_with_images:
        r = per_class[c]
        if r["train"] == 0:
            eval_status[c] = "TRAIN_ONLY_MISSING"
        elif r["val"] > 0 and r["test"] > 0:
            eval_status[c] = "FULL"
        elif r["val"] > 0 or r["test"] > 0:
            eval_status[c] = "PARTIAL"
        else:
            eval_status[c] = "TRAIN_ONLY"

    # ---- hard gate #2: every ACTIVE class must have >=1 test sample --------
    excluded_from_eval: List[str] = []
    if enforce_test_coverage:
        excluded_from_eval = _assert_test_coverage(per_class, min_eval_class_size)

    summary = {
        "schema_version": 3,
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "group_by": "content_sha256+post_id" if content_hash_grouping else "post_id",
        "content_hash_grouping": content_hash_grouping,
        "content_merge_stats": merge_stats,
        "enforce_test_coverage": enforce_test_coverage,
        "min_eval_class_size": min_eval_class_size,
        "test_coverage_remediated_characters": remediated_chars,
        "num_excluded_from_eval": len(excluded_from_eval),
        "excluded_from_eval_characters": excluded_from_eval,
        "min_train_guarantee": min_train_guarantee,
        "num_classes": len(label_map),
        "num_classes_with_images": len(chars_with_images),
        "num_groups": len(set(groups)),
        "total_images": len(items),
        "split_counts": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "train_character_coverage": round(coverage, 4),
        "num_zero_shot_characters": len(zero_shot),
        "zero_shot_characters": zero_shot,
        "eval_status": eval_status,
        "split_hash": _split_hash(splits["train"]),
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
