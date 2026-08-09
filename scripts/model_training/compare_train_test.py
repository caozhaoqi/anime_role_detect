#!/usr/bin/env python3
"""
对比训练集/验证集/测试集上的模型表现，验证数据分布差异
采用逐类加载方式避免 OOM
"""
import json, torch, numpy as np, warnings, sys, time
from pathlib import Path
from PIL import Image
from torchvision import transforms, models

# 解码策略统一收口到 src/common/preprocess 唯一真源（导入即继承）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import src.common.preprocess  # noqa: E402,F401

warnings.filterwarnings("ignore")

MODEL_DIR = Path("models/efficientnet_b3_anime_20260616_132028")
CKPT = MODEL_DIR / "model_best.pth"
TRAIN_DIR = Path("data/training_dataset")
FINAL_DIR = Path("data/final_dataset")
out_path = MODEL_DIR / "compare_report.txt"

# 输出到终端 + 文件
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
        sys.__stdout__.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_fh = open(out_path, "w")
sys.stdout = Tee(sys.__stdout__, log_fh)

with open(MODEL_DIR / "training_results.json") as f:
    r = json.load(f)
class_names = r["class_names"]
num_classes = len(class_names)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.efficientnet_b3(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3), torch.nn.Linear(model.classifier[1].in_features, num_classes)
)
ckpt = torch.load(CKPT, map_location=device)
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict, strict=False)
model.to(device).eval()
print(f"类别数: {num_classes} | 模型: efficientnet_b3 | device: {device}")

transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def evaluate_dir(data_dir, name="", max_per_class=9999, seed=42):
    """逐类加载并评估，返回指标"""
    all_preds, all_labels, all_confs = [], [], []
    per_class = {}

    for cls in class_names:
        d = data_dir / cls
        if not d.exists():
            per_class[cls] = (0, 0)
            continue
        files = sorted(d.glob("*"))[:max_per_class]
        valid = [f for f in files if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
        if not valid:
            per_class[cls] = (0, 0)
            continue
        cls_idx = class_names.index(cls)

        # 逐张推理
        correct = 0
        for f in valid:
            try:
                img = Image.open(f).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                logits = model(x).cpu()
                pred = logits.argmax().item()
                conf = logits.softmax(dim=1).max().item()
                all_preds.append(pred)
                all_labels.append(cls_idx)
                all_confs.append(conf)
                if pred == cls_idx:
                    correct += 1
            except:
                pass
        per_class[cls] = (correct, len(valid))

    if not all_preds:
        print(f"\n  {name}: ❌ 无数据")
        return

    total = len(all_labels)
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    all_confs = torch.tensor(all_confs)

    # Top-K
    print(f"\n  {'='*55}")
    print(f"  {name}")
    print(f"  {'='*55}")
    print(f"    样本数:              {total}")
    print(f"    Top-1:               {correct/total*100:.2f}% ({correct}/{total})")

    # 各类别准确率
    zero_shot = [(cls, *per_class[cls]) for cls in class_names
                  if per_class[cls][1] >= 10 and per_class[cls][0] == 0]
    print(f"    0% 类别 (样本≥10):   {len(zero_shot)} / {sum(1 for c in class_names if per_class[c][1]>=10)}")
    if zero_shot:
        print(f"      {', '.join(c for c, _, _ in zero_shot[:8])}{'...' if len(zero_shot)>8 else ''}")

    # 最常误判
    from collections import Counter
    wrong_pairs = Counter()
    for p, l in zip(all_preds.tolist(), all_labels.tolist()):
        if p != l:
            wrong_pairs[(class_names[l], class_names[p])] += 1
    print(f"    最常误判组合 (真实→预测):")
    for (true, pred), cnt in wrong_pairs.most_common(5):
        print(f"      {true:20s} → {pred:20s} ({cnt}次)")

    # 置信度
    print(f"    平均置信度:          {all_confs.mean():.4f}")
    print(f"    中位数置信度:        {all_confs.median():.4f}")

    # 预测分布
    pred_dist = Counter(all_preds.tolist())
    top_preds = pred_dist.most_common(5)
    print(f"    模型最常预测的类别:")
    for idx, cnt in top_preds:
        print(f"      [{idx:3d}] {class_names[idx]:20s} {cnt:4d}次 ({cnt/total*100:.1f}%)")

print("\n" + "="*70)
print("对比验证：训练集 vs 验证集 vs 独立测试集")
print("="*70)

# 方案：直接评估全部 training_dataset（不划分），然后评估 final_dataset
# 因为 random_split 的划分是随机的，而且验证集和训练集同分布
# 如果 training_dataset 整体表现显著优于 final_dataset → 分布漂移
print("\n  ⏳ 评估 training_dataset (全部 99 类, 逐步加载)...")
t0 = time.time()
evaluate_dir(TRAIN_DIR, "训练集 (training_dataset 全部)")
print(f"  耗时: {time.time()-t0:.0f}s")

print("\n  ⏳ 评估 final_dataset (独立测试集)...")
t0 = time.time()
evaluate_dir(FINAL_DIR, "测试集 (final_dataset 全部)")
print(f"  耗时: {time.time()-t0:.0f}s")

# 再按 80/20 划分一次看
print("\n\n" + "="*70)
print("额外验证：从 training_dataset 随机取 80% / 20% 对比")
print("="*70)

# 逐类随机分流
rng = np.random.RandomState(42)
train_preds, train_labels, train_confs = [], [], []
val_preds, val_labels, val_confs = [], [], []

for cls in class_names:
    d = TRAIN_DIR / cls
    if not d.exists():
        continue
    files = sorted(d.glob("*"))
    valid = [f for f in files if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
    if not valid:
        continue
    cls_idx = class_names.index(cls)
    rng.shuffle(valid)
    split = max(1, int(len(valid) * 0.8))
    train_files, val_files = valid[:split], valid[split:]

    for f in train_files:
        try:
            img = Image.open(f).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            logits = model(x).cpu()
            train_preds.append(logits.argmax().item())
            train_labels.append(cls_idx)
            train_confs.append(logits.softmax(dim=1).max().item())
        except:
            pass
    for f in val_files:
        try:
            img = Image.open(f).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            logits = model(x).cpu()
            val_preds.append(logits.argmax().item())
            val_labels.append(cls_idx)
            val_confs.append(logits.softmax(dim=1).max().item())
        except:
            pass

def print_split(preds, labels, confs, name):
    total = len(labels)
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    avg_conf = sum(confs) / len(confs) if confs else 0
    print(f"  {name:40s}: {total:4d} 张, Top-1: {correct/total*100:.2f}% ({correct}/{total}), 平均置信: {avg_conf:.4f}")

print()
print_split(train_preds, train_labels, train_confs, "训练数据 (training_dataset 80%)")
print_split(val_preds, val_labels, val_confs, "验证数据 (training_dataset 20%)")
print(f"\n  差值 (训练 - 验证): {(sum(1 for p,l in zip(train_preds,train_labels) if p==l)/len(train_labels) - sum(1 for p,l in zip(val_preds,val_labels) if p==l)/len(val_labels))*100:.2f}%")

log_fh.close()
print(f"\n报告已保存: {out_path}")