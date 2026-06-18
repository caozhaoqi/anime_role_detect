#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型测试报告生成器 —— 全面评估模型并输出报告
"""
import os, sys, json, torch
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断图片
from torchvision import transforms, datasets, models
from tqdm import tqdm

try:
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ sklearn/matplotlib 未安装，跳过混淆矩阵和分类报告")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

DATA_DIR = Path("data/final_dataset")
MODEL_DIR = Path("models/efficientnet_b3_anime_20260616_132028")
OUTPUT_DIR = MODEL_DIR / "eval_report"
BATCH_SIZE = 64
NUM_WORKERS = 0
MAX_SAMPLES = None  # 限制样本数用于快速测试


def load_model_and_classes():
    with open(MODEL_DIR / "training_results.json") as f:
        results = json.load(f)
    class_names = results["class_names"]
    num_classes = results["num_classes"]

    model = models.efficientnet_b3(num_classes=num_classes)
    model.load_state_dict(
        torch.load(MODEL_DIR / "model_best.pth", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, class_names, num_classes


class EvalDataset(torch.utils.data.Dataset):
    """仅加载 model 支持的类且目录非空的图片"""
    def __init__(self, data_dir: Path, class_names: list):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        self.samples = []
        self.valid_classes = {}

        for cls in class_names:
            cls_dir = data_dir / cls
            if not cls_dir.exists():
                continue
            files = sorted(cls_dir.iterdir())
            if not files:
                continue
            self.valid_classes[cls] = []
            for f in files:
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
                    self.valid_classes[cls].append(str(f))
                    self.samples.append((str(f), self.class_to_idx[cls]))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), label
        except Exception as e:
            # 损坏图片，用下一张替代
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return len(self.samples)

    def print_summary(self):
        print(f"  📂 有效样本: {len(self.samples)} 张, 覆盖 {len(self.valid_classes)}/{len(self.class_to_idx)} 个类别")
        for cls in sorted(self.valid_classes):
            print(f"    ✅ {cls}: {len(self.valid_classes[cls])} 张")


def get_dataloader(class_names):
    dataset = EvalDataset(DATA_DIR, class_names)
    if len(dataset) == 0:
        print("⚠️ 本地没有可用于评估的图片")
        sys.exit(1)
    dataset.print_summary()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    return loader, class_names


def evaluate(model, loader, device, class_names):
    all_preds = []
    all_labels = []
    all_probs = []
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    model = model.to(device)
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="📊 评估中"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

            for p, l in zip(preds.cpu().numpy(), labels.numpy()):
                per_class_total[l] += 1
                if p == l:
                    per_class_correct[l] += 1

    return all_labels, all_preds, np.array(all_probs), per_class_correct, per_class_total


def generate_report(all_labels, all_preds, all_probs, per_class_correct,
                    per_class_total, class_names, loader):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_classes = len(class_names)

    # 1. 总体指标
    total = len(all_labels)
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    acc = correct / total
    print(f"\n📊 总样本: {total}, 正确: {correct}, 准确率: {acc:.4f}")

    # 2. Top-K 准确率
    topk_acc = {}
    for k in [1, 3, 5]:
        topk = 0
        for i in range(len(all_labels)):
            topk_preds = np.argsort(all_probs[i])[-k:]
            if all_labels[i] in topk_preds:
                topk += 1
        topk_acc[k] = topk / total
        print(f"  Top-{k} 准确率: {topk_acc[k]:.4f} ({topk}/{total})")

    # 3. 每个样本的平均置信度（正确 vs 错误）
    conf_correct = []
    conf_wrong = []
    for i in range(len(all_labels)):
        p = all_probs[i][all_preds[i]]
        if all_preds[i] == all_labels[i]:
            conf_correct.append(p)
        else:
            conf_wrong.append(p)
    avg_conf_correct = np.mean(conf_correct) if conf_correct else 0
    avg_conf_wrong = np.mean(conf_wrong) if conf_wrong else 0
    print(f"\n  ✅ 预测正确平均置信度: {avg_conf_correct:.4f}")
    print(f"  ❌ 预测错误平均置信度: {avg_conf_wrong:.4f}")

    # 4. 每类指标
    class_metrics = {}
    all_classes_sorted = sorted(per_class_total.keys())
    for c in all_classes_sorted:
        total_c = per_class_total[c]
        correct_c = per_class_correct.get(c, 0)
        acc_c = correct_c / total_c if total_c > 0 else 0
        class_metrics[class_names[c]] = {
            "idx": c,
            "total": total_c,
            "correct": correct_c,
            "accuracy": round(acc_c, 4),
        }

    # 5. 混淆矩阵 & sklearn 报告（仅含实际有标签的类）
    if HAS_SKLEARN:
        present_labels = sorted(set(all_labels))
        present_class_names = [class_names[i] for i in present_labels]
        cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
        report = classification_report(
            all_labels, all_preds,
            labels=present_labels,
            target_names=present_class_names,
            output_dict=True,
            zero_division=0,
        )

        # 混淆矩阵图
        plt.figure(figsize=(min(50, len(present_labels) * 1.2), min(50, len(present_labels) * 1.2)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=present_class_names)
        disp.plot(cmap="Blues", xticks_rotation=90, values_format="d")
        plt.title(f"EfficientNet-B3 混淆矩阵 (val set)\nOverall Acc: {acc:.4f}")
        plt.tight_layout()
        cm_path = OUTPUT_DIR / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  📊 混淆矩阵已保存: {cm_path}")
    else:
        report = None

    # 6. 找出最差表现的类别
    sorted_by_acc = sorted(class_metrics.items(), key=lambda x: x[1]["accuracy"])
    worst_10 = sorted_by_acc[:10]
    best_10 = sorted_by_acc[-10:][::-1]

    # 7. 找出最常被误分的 pairs
    misclass_pairs = Counter()
    for p, l in zip(all_preds, all_labels):
        if p != l:
            misclass_pairs[(class_names[l], class_names[p])] += 1
    top_confusions = misclass_pairs.most_common(15)

    # 8. 生成报告文件
    report_path = OUTPUT_DIR / "evaluation_report.md"
    lines = []
    def w(text=""):
        lines.append(text)

    w(f"# 模型测试报告")
    w(f"**模型**: EfficientNet-B3 | **类别数**: {num_classes} | **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w()
    w("---")
    w()

    w("## 总体表现")
    w(f"| 指标 | 值 |")
    w(f"|------|-----|")
    w(f"| 测试样本数 | {total} |")
    w(f"| 总体准确率 (Top-1) | **{acc:.4f}** ({correct}/{total}) |")
    for k in [1, 3, 5]:
        w(f"| Top-{k} 准确率 | {topk_acc[k]:.4f} |")
    w(f"| 预测正确平均置信度 | {avg_conf_correct:.4f} |")
    w(f"| 预测错误平均置信度 | {avg_conf_wrong:.4f} |")
    w()

    w("## 各类别准确率")
    w(f"| 排名 | 类别 | 准确率 | 正确/总数 |")
    w(f"|------|------|--------|-----------|")
    for i, (cls, m) in enumerate(sorted_by_acc, 1):
        w(f"| {i} | {cls} | {m['accuracy']:.4f} | {m['correct']}/{m['total']} |")
    w()

    if HAS_SKLEARN and report:
        w("## 各类别详细指标 (Precision/Recall/F1)")
        w(f"| 类别 | Precision | Recall | F1-score | Support |")
        w(f"|------|-----------|--------|----------|---------|")
        for cls in present_class_names:
            if cls in report:
                r = report[cls]
                w(f"| {cls} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {r['support']} |")
        macro = report.get("macro avg", {})
        weighted = report.get("weighted avg", {})
        w(f"| **macro avg** | {macro.get('precision', 0):.4f} | {macro.get('recall', 0):.4f} | {macro.get('f1-score', 0):.4f} | {int(macro.get('support', 0))} |")
        w(f"| **weighted avg** | {weighted.get('precision', 0):.4f} | {weighted.get('recall', 0):.4f} | {weighted.get('f1-score', 0):.4f} | {int(weighted.get('support', 0))} |")
        w()

    w("## 表现最差的 10 个类别")
    w(f"| 类别 | 准确率 | 样本数 |")
    w(f"|------|--------|--------|")
    for cls, m in worst_10:
        w(f"| {cls} | {m['accuracy']:.4f} | {m['total']} |")
    w()

    w("## 表现最好的 10 个类别")
    w(f"| 类别 | 准确率 | 样本数 |")
    w(f"|------|--------|--------|")
    for cls, m in best_10:
        w(f"| {cls} | {m['accuracy']:.4f} | {m['total']} |")
    w()

    w("## 最常被误判的组合 (真实→预测)")
    w(f"| 次数 | 真实类别 | 误判为 |")
    w(f"|------|----------|--------|")
    for (true_cls, pred_cls), cnt in top_confusions:
        w(f"| {cnt} | {true_cls} | {pred_cls} |")
    w()

    w("## 混淆矩阵")
    w(f"![confusion_matrix](confusion_matrix.png)")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  📄 报告已保存: {report_path}")

    # 9. 同时保存 JSON 格式供程序消费
    json_report = {
        "model": "efficientnet_b3",
        "timestamp": datetime.now().isoformat(),
        "num_classes": num_classes,
        "total_samples": total,
        "accuracy": round(acc, 4),
        "top3_accuracy": round(topk_acc.get(3, 0), 4),
        "top5_accuracy": round(topk_acc.get(5, 0), 4),
        "avg_conf_correct": round(float(avg_conf_correct), 4),
        "avg_conf_wrong": round(float(avg_conf_wrong), 4),
        "per_class": class_metrics,
        "worst_classes": [{"class": cls, **m} for cls, m in worst_10],
        "best_classes": [{"class": cls, **m} for cls, m in best_10],
        "top_confusions": [
            {"true": t, "predicted": p, "count": c}
            for (t, p), c in top_confusions
        ],
    }
    json_path = OUTPUT_DIR / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    print(f"  📄 JSON 报告已保存: {json_path}")


def main():
    print("=" * 60)
    print("  🧪 模型测试报告生成器")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"📱 设备: {device}")

    model, class_names, num_classes = load_model_and_classes()
    loader, dataset_classes = get_dataloader(class_names)

    print(f"📂 数据集: {len(loader.dataset.samples)} 张图片, {num_classes} 个类别")

    all_labels, all_preds, all_probs, per_class_correct, per_class_total = evaluate(
        model, loader, device, class_names
    )

    generate_report(all_labels, all_preds, all_probs, per_class_correct,
                    per_class_total, class_names, loader)

    print("\n✅ 报告生成完毕！")


if __name__ == "__main__":
    main()