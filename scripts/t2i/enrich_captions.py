#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1: caption 升级——用 WD ViT v3 自动打标生成描述性 caption（替代纯模板）。

背景（T2I 改进方案 S1，2026-08-20）
-----------------------------------
当前 build_metadata / prepare_captions 的 caption 是模板化
（"role, solo character, anime style, high quality, detailed"），
每个角色每张图同一句话，LoRA 学不到发色/瞳色/服装等判别特征。

本脚本：逐图调用 WDViTV3Tagger.get_tags() 打标 →
  caption = "{role}, {tags}, solo character, anime style, high quality"
输出与 build_metadata 兼容的四列 CSV（image_path, caption, role, has_identity_token）。

健壮性
-------
- WD ViT 权重重（~2GB）、懒加载；模型加载/推理失败时回退模板 caption 并计数
- 本机 HF 不通：模型需先在本地就位（models_cache / huggingface_cache），否则自动降级
- 标签过滤：去重、剔除与角色名重复的 token、控制条数（token 预算 77）

用法
----
  .venv/bin/python scripts/t2i/enrich_captions.py --role Bailu          # 单角色
  .venv/bin/python scripts/t2i/enrich_captions.py                       # 全部角色
  .venv/bin/python scripts/t2i/enrich_captions.py --role Bailu --dry-run # 不加载模型，只验证数据通路
  .venv/bin/python scripts/t2i/enrich_captions.py --out outputs/t2i_lora/Bailu_v1/metadata.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.services.t2i_service import config  # noqa: E402

# 模板兜底（模型不可用时的降级 caption，与现状一致）
TEMPLATE_CAPTION = "{role}, solo character, anime style, high quality, detailed"
MAX_TAGS = 10          # 每图最多保留的判别性标签数（token 预算）
MIN_TAG_LEN = 3        # 过短标签过滤（如 "1"）
# 与角色名冲突/无信息量的标签黑名单（示例，可按需扩充）
SKIP_TAGS = {"solo", "1girl", "1boy", "no humans", "animal focus", "simple background"}


def build_caption(role: str, tags: list[str]) -> str:
    """组装描述性 caption，保证 token 预算（SD1.5 上限 77）。"""
    filtered = []
    role_lower = role.lower().replace("_", " ")
    for t in tags:
        t = t.strip().lower()
        if not t or len(t) < MIN_TAG_LEN:
            continue
        if t in SKIP_TAGS or t == role_lower or t in role_lower or role_lower in t:
            continue
        if t not in filtered:
            filtered.append(t)
    # 粗算 token 预算：英文 token ≈ 词数 × 1.3 + 标点；超过则截断
    budget = 60  # 给 role + 固定后缀留 ~17 token
    kept = []
    for t in filtered[:MAX_TAGS]:
        est = len(t.split()) + 2
        if sum(len(x.split()) + 2 for x in kept) + est > budget:
            break
        kept.append(t)
    tail = ", ".join(kept)
    if tail:
        return f"{role}, {tail}, solo character, anime style, high quality"
    return TEMPLATE_CAPTION.format(role=role)


def iter_role_images(data_root: Path, role: str | None = None):
    """遍历 final_dataset/<role>/*.{jpg,png,jpeg,webp}。"""
    exts = (".jpg", ".jpeg", ".png", ".webp")
    dirs = [data_root / role] if role else sorted(d for d in data_root.iterdir() if d.is_dir())
    for d in dirs:
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
        if imgs:
            yield d.name, imgs


def main():
    ap = argparse.ArgumentParser(description="WD ViT 打标生成描述性 caption（S1）")
    ap.add_argument("--role", default=None, help="单角色名；缺省处理全部角色")
    ap.add_argument("--data-dir", default=str(config.DATASET_ROOT))
    ap.add_argument("--out", default=str(config.LORA_DIR / "captions_enriched.csv"),
                    help="输出 CSV（四列：image_path, caption, role, has_identity_token）")
    ap.add_argument("--limit", type=int, default=None, help="每角色最多打标 N 张（调试）")
    ap.add_argument("--dry-run", action="store_true", help="不加载模型，只验证数据通路与 caption 组装")
    args = ap.parse_args()

    data_root = Path(args.data_dir)
    if not data_root.exists():
        print(f"[ERROR] 数据目录不存在: {data_root}")
        sys.exit(1)

    # 惰性加载真 WD ViT v3（S1 方案 A，2026-08-20）。
    # 原 WDViTV3Tagger 是"假打标器"（CLIP 零样本 + 模拟标签静默造假），已弃用。
    # 真权重需先下载（本地缺失时明确报错，绝不产出假数据）。
    tagger = None
    if not args.dry_run:
        sys.path.insert(0, str(Path(__file__).parent))  # 供 import wd_vit_captioner
        try:
            from wd_vit_captioner import WDV3Captioner
            tagger = WDV3Captioner()   # 找不到真实权重会抛 FileNotFoundError
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            print("        请先运行: .venv/bin/python scripts/t2i/download_wd_vit_modelscope.py")
            sys.exit(2)
        except Exception as e:
            import traceback
            print(f"[ERROR] 真 WD ViT 初始化失败: {e}")
            traceback.print_exc()
            sys.exit(2)
        print("[tagger] 真 WD ViT v3 就绪")

    rows = []
    stats = {"roles": 0, "images": 0, "tagged_ok": 0, "fallback": 0, "tag_fail": 0}
    for role, imgs in iter_role_images(data_root, args.role):
        stats["roles"] += 1
        if args.limit:
            imgs = imgs[: args.limit]
        for p in imgs:
            stats["images"] += 1
            caption = None
            if not args.dry_run and tagger is not None:
                caption = tagger.caption(role, str(p))  # 失败返回 None -> 回退模板
                if caption is not None:
                    stats["tagged_ok"] += 1
                else:
                    stats["tag_fail"] += 1
            if caption is None:
                caption = TEMPLATE_CAPTION.format(role=role)
                if not args.dry_run:
                    stats["fallback"] += 1
            rows.append({
                "image_path": str(p),
                "caption": caption,
                "role": role,
                "has_identity_token": "True",
            })
            if args.dry_run:
                print(f"  [{role}] {p.name} -> {caption[:80]}...")

    # 落盘
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "caption", "role", "has_identity_token"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[out] {len(rows)} 条 caption -> {out}")
    print(f"[stat] 角色 {stats['roles']} | 图 {stats['images']} | 打标成功 {stats['tagged_ok']} "
          f"| 打标失败 {stats['tag_fail']} | 模板回退 {stats['fallback']}")
    if stats["fallback"] > 0 or stats["tag_fail"] > 0:
        print("[提示] 存在回退/失败——若模型权重未就位（本机 HF 不通），需先经 ModelScope 下载 WD ViT 权重")


if __name__ == "__main__":
    main()
