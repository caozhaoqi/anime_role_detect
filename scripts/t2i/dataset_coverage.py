"""Phase 0 - 数据集覆盖度统计。

统计 data/final_dataset 各角色的图像数量与身份标签（CHARACTER_TAG_MAP）覆盖情况，
输出可训练角色清单与 Phase 0 闸门所需真实数字。

用法：
    .venv/bin/python scripts/t2i/dataset_coverage.py \
        --out outputs/t2i_phase0/coverage_report.json \
        --min-images 20
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from common import FINAL_DATASET, iter_role_images

BUCKETS = [(1, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 10**9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/t2i_phase0/coverage_report.json")
    ap.add_argument("--min-images", type=int, default=20,
                    help="Phase 0 闸门：单角色最少图像数")
    args = ap.parse_args()

    per_role = {}
    total_images = 0
    with_token = 0
    without_token_roles = []
    bucket_counter = Counter()

    for role, imgs, token in iter_role_images():
        n = len(imgs)
        total_images += n
        has = bool(token)
        if has:
            with_token += 1
        else:
            without_token_roles.append(role)
        per_role[role] = {"images": n, "has_identity_token": has, "token": token}
        for lo, hi in BUCKETS:
            if lo <= n <= hi:
                bucket_counter[(lo, hi)] += 1
                break

    gate_roles = [
        r for r, info in per_role.items()
        if info["images"] >= args.min_images and info["has_identity_token"]
    ]

    report = {
        "total_roles": len(per_role),
        "total_images": total_images,
        "roles_with_identity_token": with_token,
        "roles_without_identity_token": len(without_token_roles),
        "without_identity_token_sample": without_token_roles[:20],
        "image_distribution_buckets": {
            f"{lo}-{hi if hi < 10**9 else '+'}": bucket_counter[(lo, hi)]
            for lo, hi in BUCKETS
        },
        "phase0_gate": {
            "min_images": args.min_images,
            "roles_meeting_gate": len(gate_roles),
            "gate_sample": sorted(gate_roles)[:20],
        },
        "per_role": per_role,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"角色总数: {report['total_roles']}")
    print(f"图像总数: {report['total_images']}")
    print(f"有身份标签的角色: {with_token} / 无: {len(without_token_roles)}")
    print("图像分布:", report["image_distribution_buckets"])
    print(f"Phase0 闸门(≥{args.min_images}图且有标签) 达标角色: {len(gate_roles)}")
    print(f"报告已写出: {out}")


if __name__ == "__main__":
    main()
