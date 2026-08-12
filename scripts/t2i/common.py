"""T2I Phase 0 共享工具：加载角色身份标签映射、遍历 final_dataset。

设计说明
--------
CHARACTER_TAG_MAP（scripts/data_collection/single_collectors/collect_final_dataset.py）
是「角色目录名 -> (danbooru 搜索标签, 备选标签)」的身份 token 映射（约 76 条），
本身不含发色/服饰等属性。data/.tag_cache*.json 是「角色名 -> 标签」的归一化缓存，
同样不是逐图属性标签。因此 Phase 0 的 caption 以身份 token 为触发词 + 通用画质词，
符合 Character LoRA 标准范式；富属性 caption 需后续对全量图跑 WD ViT 打标（增强项）。
"""

import ast
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DATASET = PROJECT_ROOT / "data" / "final_dataset"
COLLECTOR = PROJECT_ROOT / "scripts" / "data_collection" / "single_collectors" / "collect_final_dataset.py"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_identity_map():
    """返回 {角色目录名: danbooru 身份标签}。

    优先 CHARACTER_TAG_MAP 主标签；缺失时回退到 .tag_cache_p0 / .tag_cache_curl
    （均为英文名 key）。不执行 collector 模块（避免其重依赖），仅静态解析源码中的 dict。
    """
    identity = {}
    # 1) CHARACTER_TAG_MAP（主来源，约 76 条）
    if COLLECTOR.exists():
        src = COLLECTOR.read_text(encoding="utf-8")
        m = re.search(r"CHARACTER_TAG_MAP\s*=\s*\{(.*?)\n\}", src, re.S)
        if m:
            try:
                block = "{" + m.group(1) + "}"
                raw = ast.literal_eval(block)
                for k, v in raw.items():
                    primary = v[0] if isinstance(v, (list, tuple)) and v else v
                    if primary:
                        identity[str(k)] = str(primary)
            except Exception:
                pass
    # 2) 回退：.tag_cache_p0.json / .tag_cache_curl.json（英文名 key）
    for cache in ["data/.tag_cache_p0.json", "data/.tag_cache_curl.json"]:
        p = PROJECT_ROOT / cache
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if str(k) not in identity and v:
                            identity[str(k)] = str(v)
            except Exception:
                pass
    return identity


def iter_role_images(min_side=None):
    """遍历 final_dataset，产出 (role, [图片绝对路径], 身份标签或 None)。"""
    identity = load_identity_map()
    if not FINAL_DATASET.exists():
        return
    for role_dir in sorted(p for p in FINAL_DATASET.iterdir() if p.is_dir()):
        role = role_dir.name
        imgs = [
            p for p in role_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if min_side:
            imgs = [p for p in imgs if p.stat().st_size >= min_side]
        token = identity.get(role)
        yield role, imgs, token


def count_images(role_dir):
    return sum(
        1 for p in role_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
