"""Phase 0 - 构造 T2I 训练 caption。

策略：以角色身份 token（CHARACTER_TAG_MAP 中的 danbooru 标签）作为触发词，
拼接通用画质词，生成 (image_path, caption) 对。这是 Character LoRA 的标准范式——
让模型从图像本身学习角色外观，过度描述属性反而损害身份一致性学习。

富属性 caption（发色/服饰等）需后续对全量图跑 WD ViT 打标，本阶段不启用。

用法：
    .venv/bin/python scripts/t2i/prepare_captions.py \
        --out outputs/t2i_phase0/metadata.csv \
        --template "{token}, solo, anime style, high quality" \
        --min-images 20
"""
import argparse
import csv
import json
from pathlib import Path

from common import iter_role_images

DEFAULT_TEMPLATE = "{token}, solo, anime style, high quality, masterpiece"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/t2i_phase0/metadata.csv")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE,
                    help="caption 模板，{token} 替换为角色身份标签")
    ap.add_argument("--min-images", type=int, default=1,
                    help="仅对图像数 >= 此值的角色生成 caption")
    ap.add_argument("--require-token", action="store_true",
                    help="仅对拥有身份标签（非目录名回退）的角色生成")
    args = ap.parse_args()

    rows = []
    stats = {"total_pairs": 0, "roles_used": 0, "roles_skipped_no_token": 0, "roles_skipped_few": 0}
    for role, imgs, token in iter_role_images():
        if len(imgs) < args.min_images:
            stats["roles_skipped_few"] += 1
            continue
        effective_token = token if token else role
        if args.require_token and not token:
            stats["roles_skipped_no_token"] += 1
            continue
        caption = args.template.format(token=effective_token)
        for p in imgs:
            rows.append({
                "image_path": str(p),
                "caption": caption,
                "role": role,
                "has_identity_token": bool(token),
            })
        stats["roles_used"] += 1

    stats["total_pairs"] = len(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "caption", "role", "has_identity_token"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = out.with_suffix(".json")
    summary.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"生成 caption 对: {stats['total_pairs']}")
    print(f"使用角色: {stats['roles_used']} | 跳过(图少): {stats['roles_skipped_few']} | 跳过(无标签): {stats['roles_skipped_no_token']}")
    print(f"CSV: {out}")
    print(f"统计: {summary}")
    if rows:
        print("样例 caption:", rows[0]["caption"])


if __name__ == "__main__":
    main()
