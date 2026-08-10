#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单测：验证 grouped_split 按 **post-id（图片来源 / 近重复簇）** 分组防泄漏。

语义（P0 二次修正后）：
  * 同一 post-id 的所有变体（不同裁剪 / _1 / _dup / _cropped / 不同扩展名）
    必须整体落在同一个 split —— 这才是真正的"零泄漏"。
  * 同一角色的**不同 post** 的图 **允许**跨 train/val/test —— 这是闭集分类器
    的正常训练目标，不是泄漏。

历史：上一版按"角色"分组，虽然不泄漏但矫枉过正，导致 52/197 个类在 train
零样本（只出现在 val/test），模型对这些类只能盲猜，指标无意义。

不依赖 GPU / torch，仅用 sklearn（已在 .venv 中）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.data.split_utils import (  # noqa: E402
    extract_post_id,
    grouped_split,
    make_character_grouped_split,
    post_group_key,
)

# 训练集角色覆盖率下限。实测 seed=42 全量数据集为 99.42%
# （唯一未覆盖的 'rina' 只有 1 张图，数学上不可能同时进 train 和 val/test）。
MIN_TRAIN_COVERAGE = 0.95


def test_extract_post_id_variants():
    """post-id 提取：同一 post 的各种变体必须归一到同一个 id。"""
    assert extract_post_id("1008785.png") == "1008785"
    assert extract_post_id("5776378_1.jpg") == "5776378"  # 裁剪序号
    assert extract_post_id("6883660_cropped.jpg") == "6883660"
    assert extract_post_id("1251664_dup.jpg") == "1251664"
    assert extract_post_id("pixiv_123456_p0.jpg") == "123456"  # 带来源前缀
    assert extract_post_id("danbooru-987654.jpg") == "987654"
    # 同一 post 的不同扩展名 / 不同裁剪 -> 同一簇
    assert extract_post_id("5154551.jpg") == extract_post_id("5154551.png")
    assert extract_post_id("5154551.jpg") == extract_post_id("5154551_2.webp")
    # 解析不出数字 post-id 时退化为按文件名（后续会按角色加作用域）
    assert extract_post_id("img_0.jpg") == "img"
    # 数字 post-id 是全局的；非数字回退按角色隔离，避免跨角色误合并
    assert post_group_key("char_a", "5154551.jpg") == "post:5154551"
    assert post_group_key("char_a", "img_0.jpg") != post_group_key(
        "char_b", "img_0.jpg"
    )
    print("[PASS] extract_post_id: 所有变体正确归一")


def test_same_post_never_splits_same_char_may_span():
    """构造假数据集：2 角色 × 3 post × 每 post 2 张近重复图。

    断言：① 同 post 绝不跨集；② 同角色允许跨集。
    """
    items, groups, chars = [], [], []
    post_of = {}
    pid = 6000000
    for c in ("char_a", "char_b"):
        for _ in range(3):
            pid += 1
            for variant in range(2):  # 每 post 2 张近重复图
                name = f"{pid}.jpg" if variant == 0 else f"{pid}_{variant}.jpg"
                items.append(f"{c}/{name}")
                groups.append(post_group_key(c, name))
                chars.append(c)
                post_of[len(items) - 1] = str(pid)

    assert len(items) == 12 and len(set(groups)) == 6, "应为 6 个 post 组 / 12 张图"

    char_spanned_at_least_once = False
    for seed in range(10):
        train, val, test = grouped_split(items, groups, seed=seed)

        # ① 同一 post-id 绝不跨集（硬不变量，所有 seed 都必须成立）
        tp = {post_of[i] for i in train}
        vp = {post_of[i] for i in val}
        ep = {post_of[i] for i in test}
        assert not (tp & vp) and not (tp & ep) and not (vp & ep), (
            f"LEAKAGE(seed={seed}): post-id 跨集 "
            f"t&v={tp & vp} t&e={tp & ep} v&e={vp & ep}"
        )

        # 每张图恰好被分配一次
        assert set(train) | set(val) | set(test) == set(range(len(items)))
        assert len(train) + len(val) + len(test) == len(items)

        # ② 同角色跨集是被允许的
        tc = {chars[i] for i in train}
        rest_c = {chars[i] for i in val} | {chars[i] for i in test}
        if tc & rest_c:
            char_spanned_at_least_once = True

    assert char_spanned_at_least_once, (
        "同角色应当被允许跨集（闭集分类器的正常目标），但 10 个 seed 均未出现"
    )
    print("[PASS] 假数据集：同 post 零跨集；同角色可跨集")


def test_real_dataset_no_post_id_leak():
    """真实 final_dataset：post-id 零跨集 + 训练集角色覆盖率达标。"""
    data_dir = ROOT / "data" / "final_dataset"
    if not data_dir.is_dir():
        print("[SKIP] data/final_dataset not present")
        return

    res = make_character_grouped_split(data_dir, seed=42)
    summary = res["summary"]

    def pids_of(split):
        return {extract_post_id(s["path"].split("/", 1)[1]) for s in split}

    def chars_of(split):
        return {s["path"].split("/", 1)[0] for s in split}

    tp, vp, ep = pids_of(res["train"]), pids_of(res["val"]), pids_of(res["test"])

    # ① 核心零泄漏：同 post-id 不跨集
    assert not (tp & vp), f"LEAKAGE: post-id 跨 train/val: {sorted(tp & vp)[:10]}"
    assert not (tp & ep), f"LEAKAGE: post-id 跨 train/test: {sorted(tp & ep)[:10]}"
    assert not (vp & ep), f"LEAKAGE: post-id 跨 val/test: {sorted(vp & ep)[:10]}"

    # ② 训练集角色覆盖率（修正前按角色分组时为 69.6%，52 个类零样本）
    coverage = summary["train_character_coverage"]
    assert coverage >= MIN_TRAIN_COVERAGE, (
        f"训练集角色覆盖率过低: {coverage:.2%} < {MIN_TRAIN_COVERAGE:.0%}；"
        f"零样本类={summary['zero_shot_characters']}"
    )

    # ③ 角色允许跨集 —— 不再断言"角色不跨集"，反过来要求它确实发生了
    tc, vc, ec = chars_of(res["train"]), chars_of(res["val"]), chars_of(res["test"])
    spanning = (tc & vc) | (tc & ec) | (vc & ec)
    assert spanning, "闭集分类器应当有角色跨集，否则说明又退回按角色分组了"

    assert summary["group_by"] == "post_id"
    print(
        f"[PASS] real final_dataset: {summary['total_images']} imgs / "
        f"{summary['num_groups']} post-ids / "
        f"{summary['num_classes_with_images']} classes-with-images -> "
        f"train={len(res['train'])} val={len(res['val'])} test={len(res['test'])}; "
        f"0 post-id 跨集; 训练覆盖率={coverage:.2%}; "
        f"零样本类={summary['num_zero_shot_characters']}; 跨集角色={len(spanning)}"
    )


def test_duplicate_class_merged():
    """回归：kotori_itsuka 已并入 itsuka_kotori，重复类不应再出现。"""
    data_dir = ROOT / "data" / "final_dataset"
    if not data_dir.is_dir():
        print("[SKIP] data/final_dataset not present")
        return
    assert not (data_dir / "kotori_itsuka").exists(), (
        "kotori_itsuka 应已合并进 itsuka_kotori 并删除"
    )
    print("[PASS] 重复类 kotori_itsuka 已合并/移除")


def test_make_character_grouped_split_schema():
    """make_character_grouped_split 产出与 train_clean_split.py 兼容的 schema。"""
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    ds = tmp / "final_dataset"
    label_map = {}
    pid = 7000000
    for i, c in enumerate(["char_a", "char_b", "char_c"]):
        (ds / c).mkdir(parents=True)
        label_map[c] = i
        for _ in range(5):
            pid += 1
            for variant in range(2):
                name = f"{pid}.jpg" if variant == 0 else f"{pid}_{variant}.jpg"
                (ds / c / name).write_text("x")
    out = tmp / "splits"
    res = make_character_grouped_split(ds, out_dir=out, seed=42, label_map=label_map)

    for name in ("train", "val", "test"):
        assert (out / f"{name}.json").is_file(), f"missing {name}.json"
        assert all(set(s.keys()) >= {"path", "label"} for s in res[name])
    assert (out / "summary.json").is_file()

    # per_class 必须按角色统计（不能被 post-id 污染）
    per_class = res["summary"]["per_class"]
    assert set(per_class) <= set(label_map), f"per_class 应按角色为 key: {list(per_class)[:5]}"
    for c, row in per_class.items():
        assert row["total"] == row["train"] + row["val"] + row["test"], f"{c} 计数不一致"
        assert row["total"] == 10, f"{c} 应有 10 张图, got {row['total']}"
    print("[PASS] schema 兼容 + per_class 按角色正确计数")


def test_no_zero_shot_class():
    """回归（P0 解锁）：min_train_guarantee 必须保证每个有图类都进 train。"""
    data_dir = ROOT / "data" / "final_dataset"
    if not data_dir.is_dir():
        print("[SKIP] data/final_dataset not present")
        return
    res = make_character_grouped_split(data_dir, seed=42)
    summary = res["summary"]
    assert summary["num_zero_shot_characters"] == 0, (
        f"仍存在零样本类: {summary['zero_shot_characters']}"
    )
    for c, row in summary["per_class"].items():
        assert row["train"] >= 1, f"类 {c} 在 train 中 0 张图"
    assert summary["train_character_coverage"] == 1.0
    print(f"[PASS] 零样本清零: {summary['num_classes_with_images']} 类全部进 train")


def test_no_postid_leakage():
    """独立封装：三集合 post-id 两两交集为空（不变量）。"""
    data_dir = ROOT / "data" / "final_dataset"
    if not data_dir.is_dir():
        print("[SKIP] data/final_dataset not present")
        return
    res = make_character_grouped_split(data_dir, seed=42)

    def pids(split):
        return {extract_post_id(s["path"].split("/", 1)[1]) for s in split}

    tp, vp, ep = pids(res["train"]), pids(res["val"]), pids(res["test"])
    assert not (tp & vp) and not (tp & ep) and not (vp & ep), (
        f"post-id 跨集 t&v={sorted(tp&vp)[:5]} t&e={sorted(tp&ep)[:5]} v&e={sorted(vp&ep)[:5]}"
    )
    print(f"[PASS] post-id 零跨集: train={len(tp)} val={len(vp)} test={len(ep)} 组")


def test_split_deterministic():
    """同 seed 两次运行，split_hash 必须一致（可复现）。"""
    data_dir = ROOT / "data" / "final_dataset"
    if not data_dir.is_dir():
        print("[SKIP] data/final_dataset not present")
        return
    r1 = make_character_grouped_split(data_dir, seed=42)
    r2 = make_character_grouped_split(data_dir, seed=42)
    assert r1["summary"]["split_hash"] == r2["summary"]["split_hash"], "split_hash 不稳定"
    assert [s["path"] for s in r1["train"]] == [s["path"] for s in r2["train"]]
    print(f"[PASS] 切分可复现: split_hash={r1['summary']['split_hash'][:12]}...")


if __name__ == "__main__":
    test_extract_post_id_variants()
    test_same_post_never_splits_same_char_may_span()
    test_real_dataset_no_post_id_leak()
    test_duplicate_class_merged()
    test_make_character_grouped_split_schema()
    test_no_zero_shot_class()
    test_no_postid_leakage()
    test_split_deterministic()
    print("ALL TESTS PASSED")
