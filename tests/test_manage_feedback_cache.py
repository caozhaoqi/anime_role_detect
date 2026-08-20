"""manage_feedback_cache.select_for_eviction 单元测试（纯函数 + tmp_path，不碰真实目录）。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "model_training"))

from manage_feedback_cache import select_for_eviction  # noqa: E402


def _mk(tmp_path, name, content=b"x", mtime=None):
    p = tmp_path / name
    p.write_bytes(content)
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    st = os.stat(p)
    return {"rid": name[:-4], "path": str(p), "size": st.st_size, "mtime": st.st_mtime}


def test_unconsumed_never_selected(tmp_path):
    """未消费（rid 不在 manifest）的图片永不入选。"""
    a = _mk(tmp_path, "rid_A.jpg", b"a", mtime=100)
    b = _mk(tmp_path, "rid_B.jpg", b"b", mtime=200)
    c = _mk(tmp_path, "rid_C.jpg", b"c", mtime=300)
    d = _mk(tmp_path, "rid_D.jpg", b"d", mtime=400)  # 未消费
    images = [a, b, c, d]
    consumed = {"rid_A": {}, "rid_B": {}, "rid_C": {}}
    # 收紧 max_files 强制触发淘汰，验证未消费的 D 绝不入选
    out = select_for_eviction(images, consumed, max_bytes=10**9, max_files=2)
    rids = {i["rid"] for i in out}
    assert "rid_D" not in rids  # 安全铁律：未消费永不入选
    assert rids <= {"rid_A", "rid_B", "rid_C"}  # 仅已消费图可入选


def test_oldest_first(tmp_path):
    """最旧优先：3 张已消费图，max_files=2 -> 删最旧的两张 A、B。"""
    a = _mk(tmp_path, "rid_A.jpg", b"a", mtime=100)
    b = _mk(tmp_path, "rid_B.jpg", b"b", mtime=200)
    c = _mk(tmp_path, "rid_C.jpg", b"c", mtime=300)
    images = [a, b, c]
    consumed = {"rid_A": {}, "rid_B": {}, "rid_C": {}}
    out = select_for_eviction(images, consumed, max_bytes=10**9, max_files=2)
    rids = [i["rid"] for i in out]
    assert rids == ["rid_A", "rid_B"]  # 最旧两个
    assert "rid_C" not in rids  # 最新的保留


def test_under_limit_no_delete(tmp_path):
    """低于上限：待删列表为空。"""
    a = _mk(tmp_path, "rid_A.jpg", b"a", mtime=100)
    b = _mk(tmp_path, "rid_B.jpg", b"b", mtime=200)
    images = [a, b]
    consumed = {"rid_A": {}, "rid_B": {}}
    # 文件数上限 5，体积上限充足 -> 不删
    assert select_for_eviction(images, consumed, max_bytes=10**9, max_files=5) == []
    # 单文件也低于上限
    assert select_for_eviction([a], consumed, max_bytes=10**9, max_files=2) == []


def test_dedup_keeps_newest(tmp_path):
    """dedup：同内容多余副本进候选，每组最新一份（keeper）绝不删。"""
    # X 与 Y 内容相同（同 hash），X 较旧；Z 内容不同且最新
    x = _mk(tmp_path, "rid_X.jpg", b"same-content", mtime=100)
    y = _mk(tmp_path, "rid_Y.jpg", b"same-content", mtime=200)
    z = _mk(tmp_path, "rid_Z.jpg", b"different", mtime=300)
    images = [x, y, z]
    consumed = {"rid_X": {}, "rid_Y": {}, "rid_Z": {}}

    # 不开启 dedup：max_files=1 时三张全部进待删（含 keeper Y）
    no_dedup = select_for_eviction(images, consumed, max_bytes=10**9, max_files=1)
    assert {i["rid"] for i in no_dedup} == {"rid_X", "rid_Y", "rid_Z"}

    # 开启 dedup：keeper Y 保留，多余副本 X 与另一份 Z 进待删
    with_dedup = select_for_eviction(images, consumed, max_bytes=10**9, max_files=1, dedup=True)
    out_rids = {i["rid"] for i in with_dedup}
    assert "rid_Y" not in out_rids  # keeper 绝不删
    assert "rid_X" in out_rids      # 多余副本被淘汰
    assert "rid_Z" in out_rids
