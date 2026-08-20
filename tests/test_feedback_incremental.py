"""feedback_incremental_training.load_feedback_samples 纯逻辑测试（不依赖 torch）。

覆盖：① 重复 recognition_id 取最新；② 缺图 / 非法 label 被滤除；③ 已在 manifest 的被排除。
通过 monkeypatch 模块级 project_root 到 tmp_path，使 image_ref 在临时目录内解析。
"""
import os
import sys
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "model_training"))

import feedback_incremental_training as fit  # noqa: E402


CLASS_TO_IDX = {
    "Aru": 0,
    "Clara": 1,
    "Firefly": 2,
    "Klee": 3,
    "nahida": 4,
}
VALID_LABELS = set(CLASS_TO_IDX.keys())


def _img_dir(tmp_path):
    d = tmp_path / "data" / "feedback_images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_img(d, rid):
    p = d / f"{rid}.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0fakejpg")
    return f"data/feedback_images/{rid}.jpg"


def _write_jsonl(tmp_path, rows, fname="feedback_2026-08-02.jsonl"):
    log_dir = tmp_path / "logs" / "feedback"
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / fname
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(log_dir)


@pytest.fixture(autouse=True)
def _patch_root(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, "project_root", str(tmp_path))


def _sample_row(rid, label, image_ref):
    return {
        "recognition_id": rid,
        "corrected_label": label,
        "image_ref": image_ref,
        "server_timestamp": "2026-08-02T00:00:00",
    }


def test_dedup_keeps_latest(tmp_path):
    """同一 recognition_id 出现两次，取最后一行（最新 label）。"""
    img_dir = _img_dir(tmp_path)
    ref = _write_img(img_dir, "rid1")
    rows = [
        _sample_row("rid1", "Aru", ref),       # 旧（label=Aru）
        _sample_row("rid1", "Clara", ref),     # 新（label=Clara，应胜出）
    ]
    log_dir = _write_jsonl(tmp_path, rows)
    out = fit.load_feedback_samples(log_dir, CLASS_TO_IDX, {}, min_samples=0)
    assert len(out) == 1
    assert out[0]["recognition_id"] == "rid1"
    assert out[0]["corrected_label"] == "Clara"  # 最后一行胜出


def test_filter_missing_image_and_invalid_label(tmp_path):
    """缺图或非法 label 的行被滤除，仅合法行保留。"""
    img_dir = _img_dir(tmp_path)
    ref_ok = _write_img(img_dir, "ok")
    # 缺图：image_ref 指向不存在文件
    rows = [
        _sample_row("ok", "Klee", ref_ok),
        _sample_row("missing_img", "Klee", "data/feedback_images/nope.jpg"),
        _sample_row("bad_label", "NotARealRole", ref_ok),  # 非法 label（不在 51 类）
    ]
    log_dir = _write_jsonl(tmp_path, rows)
    out = fit.load_feedback_samples(log_dir, CLASS_TO_IDX, {}, min_samples=0)
    rids = {s["recognition_id"] for s in out}
    assert rids == {"ok"}
    assert "missing_img" not in rids
    assert "bad_label" not in rids


def test_exclude_consumed(tmp_path):
    """已在 consumed_manifest 的 recognition_id 被排除。"""
    img_dir = _img_dir(tmp_path)
    ref_a = _write_img(img_dir, "consumed")
    ref_b = _write_img(img_dir, "fresh")
    rows = [
        _sample_row("consumed", "Aru", ref_a),
        _sample_row("fresh", "Clara", ref_b),
    ]
    log_dir = _write_jsonl(tmp_path, rows)
    consumed = {"consumed": {"consumed_at": "x", "consumed_by": "v4_old"}}
    out = fit.load_feedback_samples(log_dir, CLASS_TO_IDX, consumed, min_samples=0)
    rids = {s["recognition_id"] for s in out}
    assert rids == {"fresh"}
    assert "consumed" not in rids


def test_min_samples_threshold(tmp_path):
    """min_samples 仅用于日志统计；返回列表本身不受其裁剪（由调用方依此判退出码）。"""
    img_dir = _img_dir(tmp_path)
    ref = _write_img(img_dir, "only1")
    rows = [_sample_row("only1", "Firefly", ref)]
    log_dir = _write_jsonl(tmp_path, rows)
    out = fit.load_feedback_samples(log_dir, CLASS_TO_IDX, {}, min_samples=10)
    assert len(out) == 1  # 仍返回 1 条，由 main 据 min_samples 决定退出码 2
