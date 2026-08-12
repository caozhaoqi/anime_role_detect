#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_t2i_role.py — T2I 角色一致性闭环校验器 (closed-loop verifier)

把文生图（SD1.5 + LoRA）生成的图片，丢回项目已有的 EfficientNet 识别系统
（默认 v9 诚实基线），判定其是否被识别为目标角色，从而量化「角色一致性」。

闭环链路：
    训练 LoRA → 生成图(smoke_test.png)
              → EfficientNet v9 推理 → 预测索引
              → 由 training_results.json 的 class_names[索引] 还原角色名
              → 预测角色名 == 目标角色 ? 一致 : 不一致

====================================================================
⚠️ 两个易踩的坑（本项目识别链路的特殊性，已在本脚本中处理）：
--------------------------------------------------------------------
1) 模型选择必须在 import classify_image【之前】用环境变量锁定：
   EfficientNetClassifier.MODEL_DIR_NAME 是「类属性」，在模块导入瞬间
   就读取 EFFICIENTNET_MODEL_DIR 并固化，默认是 efficientnet_b3_v4（174 类）。
   本脚本用两阶段参数解析，先抽出 --model-name 再设置该环境变量，确保加载的是
   目标模型（默认 efficientnet_b3_v9）。直接把 --model-name 透传给
   classify_image 是【无效】的（use_model 分支会忽略该参数）。

2) classify_image(use_model=True) 返回的角色是「数值索引字符串」（如 "21"），
   不是角色名。因为 checkpoint 的 class_to_idx 用的是数值字符串键。
   真实「索引 → 角色名」映射在 models/<model>/training_results.json 的
   class_names 列表里（class_names[索引] = 角色名）。
   本脚本读取该映射把数值索引还原为角色名后再比较，避免永远比对失败。

若目标角色不在 v9 的 167 类契约内（TRAIN_ONLY 单图类），v9 无法判定，
脚本会显式告警并建议改用 CLIP+FAISS 检索（需联网下载 CLIP 权重）。
====================================================================

用法：
    # 1) 校验单张生成图
    .venv/bin/python scripts/t2i/verify_t2i_role.py \
        --image outputs/t2i_lora/amber_v1/smoke_test.png --target-role amber

    # 2) 自动定位 LoRA 输出目录里的 smoke_test.png 并校验
    .venv/bin/python scripts/t2i/verify_t2i_role.py \
        --lora-dir outputs/t2i_lora/amber_v1 --target-role amber

    # 3) 批量校验整个目录
    .venv/bin/python scripts/t2i/verify_t2i_role.py \
        --image-dir outputs/t2i_lora/amber_v1 --target-role amber

    # 4) 导出 JSON 报告（便于 CI / 流水线消费）
    ... --json outputs/t2i_lora/amber_v1/verify_report.json

依赖：复用项目 .venv（含 v9 模型与识别链路），不引入新依赖。
注意：只用 EfficientNet 分类分支（use_model=True），不触发 CLIP/FAISS 下载。
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# 日志静音：识别链路日志很吵，这里只保留关键信息
os.environ.setdefault("LOGURU_LEVEL", "WARNING")


# ---------------------------------------------------------------------------
# 两阶段解析：在 import classify_image 之前，先抽出 --model-name 并锁定环境变量
# ---------------------------------------------------------------------------
def _preparse_model_name(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--model-name", default="efficientnet_b3_v9")
    ns, _ = ap.parse_known_args(argv)
    return ns.model_name


_MODEL_NAME = _preparse_model_name(sys.argv[1:])
# 关键：必须在 import classify_image 之前设置，导入即固化模型目录
os.environ["EFFICIENTNET_MODEL_DIR"] = _MODEL_NAME


def load_classifier():
    """延迟导入识别链路（首次调用才加载 v9 权重）。"""
    try:
        from src.services.model.classification_service import classify_image
        return classify_image
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] 无法导入识别链路：{e}")
        sys.exit(2)


def load_class_names(model_dir_name: str):
    """读取 models/<dir>/training_results.json 的 class_names（索引→角色名）。"""
    path = ROOT / "models" / model_dir_name / "training_results.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            tr = json.load(f)
        return tr.get("class_names")
    except Exception:
        return None


def resolve_role_name(role, class_names):
    """把 classify_image 返回的角色（可能是数值索引字符串）还原为角色名。

    - v9 返回数值索引字符串（如 "21"）→ 用 class_names[索引] 还原。
    - 其他模型若直接返回角色名 → 原样返回。
    """
    role = str(role).strip()
    if role.isdigit() and class_names is not None:
        idx = int(role)
        if 0 <= idx < len(class_names):
            return class_names[idx]
    return role  # 已经就是角色名


def verify_one(classify_image, image_path, target_role, class_names, use_attributes):
    """校验单张图，返回 (pred_name, similarity, consistent:bool, mode)。"""
    role, similarity, boxes, mode, attrs, texts = classify_image(
        image_path,
        use_model=True,
        use_attributes=use_attributes,
    )
    pred_name = resolve_role_name(role, class_names)
    consistent = (pred_name.strip().lower() == target_role.strip().lower())
    return pred_name, float(similarity), consistent, mode


def collect_images(args):
    if args.image:
        return [args.image]
    if args.lora_dir:
        smoke = Path(args.lora_dir) / "smoke_test.png"
        if smoke.exists():
            return [str(smoke)]
        # 未找到 smoke_test.png，退化为扫描整目录
    if args.image_dir or args.lora_dir:
        base = args.image_dir or args.lora_dir
        exts = tuple("." + e.strip().lower() for e in args.ext.split(","))
        return [str(p) for p in sorted(Path(base).rglob("*")) if p.suffix.lower() in exts]
    print("[ERROR] 必须提供 --image / --image-dir / --lora-dir 之一")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="T2I 角色一致性闭环校验器")
    ap.add_argument("--image", help="单张生成图路径")
    ap.add_argument("--image-dir", help="批量校验：目录内所有图片")
    ap.add_argument("--lora-dir", help="自动定位 LoRA 输出目录的 smoke_test.png")
    ap.add_argument("--target-role", required=True,
                    help="目标角色名（与 CHARACTER_TAG_MAP 角色目录名一致，如 amber）")
    ap.add_argument("--model-name", default="efficientnet_b3_v9",
                    help="识别模型目录名（默认 efficientnet_b3_v9；写进 EFFICIENTNET_MODEL_DIR）")
    ap.add_argument("--use-attributes", action="store_true",
                    help="启用属性预测分支（会触发 WD ViT 权重下载，需联网）")
    ap.add_argument("--ext", default="jpg,jpeg,png,webp",
                    help="批量模式下的图片扩展名，逗号分隔")
    ap.add_argument("--threshold", type=float, default=50.0,
                    help="一致率达标阈值（%%），默认 50")
    ap.add_argument("--json", help="导出 JSON 报告路径")
    args = ap.parse_args()

    # 校验目标角色是否在模型契约内
    class_names = load_class_names(args.model_name)
    if class_names is not None:
        if args.target_role not in class_names:
            print(f"[WARN] 目标角色 '{args.target_role}' 不在 {args.model_name} 的 "
                  f"{len(class_names)} 类契约内，EfficientNet 无法判定该角色。")
            print("       建议改用 CLIP+FAISS 检索校验（--use-clip-faiss，但需联网下载权重）。")
    else:
        print(f"[WARN] 未能读取 {args.model_name}/training_results.json，角色名校验可能不准。")

    classify_image = load_classifier()
    targets = collect_images(args)
    if not targets:
        print("[ERROR] 未找到待校验图片")
        sys.exit(1)

    print("=== T2I 一致性校验 ===")
    print(f"目标角色: {args.target_role}")
    print(f"识别模型: {args.model_name} (EFFICIENTNET_MODEL_DIR={os.environ.get('EFFICIENTNET_MODEL_DIR')})")
    print(f"样本数  : {len(targets)}")
    print("-" * 48)

    ok = 0
    rows = []
    for img in targets:
        try:
            pred, sim, consistent, mode = verify_one(
                classify_image, img, args.target_role, class_names, args.use_attributes)
        except Exception as e:
            print(f"  [FAIL] {os.path.basename(img)}: {e}")
            rows.append({"image": os.path.basename(img), "error": str(e),
                         "predicted_role": None, "similarity": 0.0, "consistent": False})
            continue
        ok += 1 if consistent else 0
        rows.append({"image": os.path.basename(img), "predicted_role": str(pred),
                     "similarity": round(sim, 4), "consistent": consistent})
        flag = "✅ 一致" if consistent else "❌ 不一致"
        print(f"  {os.path.basename(img)[:34]:34} -> 预测[{pred}] sim={sim:.3f} {flag}")

    print("-" * 48)
    rate = (ok / len(targets)) * 100.0 if targets else 0.0
    print(f"一致性命中: {ok}/{len(targets)}  ({rate:.1f}%)")
    if rate >= args.threshold:
        print(f"结论: 角色一致性达标（≥{args.threshold:.0f}%），LoRA 初步可用。")
        verdict = "PASS"
    else:
        print(f"结论: 角色一致性偏低（<{args.threshold:.0f}%），建议增样本/调 LoRA 超参或排查 caption。")
        verdict = "FAIL"

    if args.json:
        report = {
            "target_role": args.target_role,
            "model_name": args.model_name,
            "threshold": args.threshold,
            "samples": len(targets),
            "consistent": ok,
            "consistency_rate": round(rate, 2),
            "verdict": verdict,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "details": rows,
        }
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[json] 报告已导出: {args.json}")


if __name__ == "__main__":
    main()
