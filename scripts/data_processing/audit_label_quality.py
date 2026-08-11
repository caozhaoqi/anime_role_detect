#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
弱类标签质量审计 (Label Quality Audit)
======================================

针对「训练图充足却认不准」的弱类做小批量标签质量审计，定位两类根因：

  ① 标签噪声 —— 文件夹里混了别的角色 / 多人图
  ② 域偏移   —— 人脸(角色)过小 / 漫画风格(黑白、分镜、对话框)

四路检测
--------
  A. 错标嫌疑  CLIP kNN 近邻投票 + v7 模型预测 + WD 角色标签  (三路证据三角验证)
  B. 多人图    YOLOv8 person 计数 (+ WD 多角色标签作为动漫域补充信号)
  C. 小角色    YOLOv8 最大框面积占比
  D. 漫画风    WD ViT v3 风格标签 (sigmoid 多标签)

铁律
----
  * 只读 + 只标记。本脚本 **绝不删除 / 移动 / 修改** 任何图片或数据目录。
    唯一的写操作发生在 --out-dir 之内 (JSON 清单)。
  * 任何一路检测不可用时降级并在报告中标注 status，不阻塞其余路径。

用法
----
  ./.venv/bin/python scripts/data_processing/audit_label_quality.py \
      --classes "sorasaki_hina,ram_(re:zero),..." \
      --out-dir outputs/label_audit

  # 快速自验 (每类 15 张)
  ./.venv/bin/python scripts/data_processing/audit_label_quality.py \
      --classes "paimon,lisa" --sample 15
"""

# ---------------------------------------------------------------------------
# 环境变量必须在导入 torch / 项目模块之前设置。
# 项目内 clip_embedder.py / wd_vit_v3_tagger.py 在 import 期用 os.environ.setdefault
# 强制 CPU 并关闭 MPS fallback；我们提前占位以保留 MPS 与算子回退能力。
# ---------------------------------------------------------------------------
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 权重全部已在本地缓存，强制离线避免 HuggingFace 读超时拖死流程
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import csv
import glob
import json
import logging
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")

# --- 资产路径 (均已核实存在) ------------------------------------------------
YOLO_WEIGHTS = "models/yolov8n.pt"
# 注意: models/efficientnet_b3_v7/model_best.pth 已被 epoch-3 重训产物覆盖
# (epoch=3, best_acc=0.4524)。真正的 v7 基线是 model_full.pth (epoch=37,
# best_acc=0.6183, 与 training_results.json 完全一致)。默认用后者。
V7_FULL = "models/efficientnet_b3_v7/model_full.pth"
V7_BEST = "models/efficientnet_b3_v7/model_best.pth"
V7_NAME2IDX = "outputs/v7_audit/name2idx.json"
CLIP_DOWNLOAD_ROOT = os.path.expanduser("~/.cache/clip")

# --- D 路: 规格指定的漫画风标签 --------------------------------------------
COMIC_TAGS_PRIMARY = ("comic", "monochrome", "speech_bubble", "multiple_girls", "sketch")
# multiple_girls 语义上属于「多角色」而非「漫画风」。若混在一起，comic_style 会
# 被多人图刷满(实测 paimon 5/5 全部只因 multiple_girls 命中)，导致无法区分
# 猜测①(标签噪声/多人) 与 猜测②(域偏移/漫画风)。因此额外给出 strict 口径。
COMIC_TAGS_STRICT = ("comic", "monochrome", "speech_bubble", "sketch")
# 补充记录 (不参与 comic_style 判定，单独存档供人工参考)
COMIC_TAGS_EXTRA = ("greyscale", "4koma", "traditional_media", "lineart", "borrowed_character")
# B 路补充: 动漫域多角色标签 (YOLO person 在动漫图上实测召回仅 ~38%)
MULTI_CHAR_TAGS = (
    "multiple_girls", "2girls", "3girls", "4girls", "5girls", "6+girls",
    "multiple_boys", "2boys", "3boys", "multiple_others", "group",
)

# --- A3 路: 文件夹类名 -> WD v3 角色标签别名 --------------------------------
# WD 标签表用后缀区分作品，与本项目类名不同名。仅收录人工确认过的映射；
# 未收录者该路记为 unmapped (不产生误报)。
WD_CHARACTER_ALIAS: Dict[str, str] = {
    "sorasaki_hina": "hina_(blue_archive)",
    "silver_wolf": "silver_wolf_(honkai:_star_rail)",
    "lisa": "lisa_(genshin_impact)",
    "yuuki_asuna": "asuna_(sao)",
    "asagi_mutsuki": "mutsuki_(blue_archive)",
    "paimon": "paimon_(genshin_impact)",
    "ram_(re:zero)": "ram_(re:zero)",
    "hiiragi_kagami": "hiiragi_kagami",
    "kaname_madoka": "kaname_madoka",
    "tsushima_yoshiko": "tsushima_yoshiko",
    # lynx / kagura_(onmyouji): WD v3 词表中无对应角色标签 -> unmapped
    # --- 强类对照组 (test F1 >= 0.75), 仅用于判别力标定, 非审计目标 ---
    "collei": "collei_(genshin_impact)",
    "hu_tao": "hu_tao_(genshin_impact)",
    "sayu": "sayu_(genshin_impact)",
    "ningguang": "ningguang_(genshin_impact)",
    "hachikuji_mayoi": "hachikuji_mayoi",
    # sigewinne / ilulu_(maid_dragon) / corin_wickes: WD v3 词表中无对应标签 -> unmapped
}

logger = logging.getLogger("label_audit")


# ===========================================================================
# 工具函数
# ===========================================================================
def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # 项目内 wd tagger 使用 loguru/自定义 logger，噪声很大，压掉
    for noisy in ("src.core.tagging.wd_vit_v3_tagger", "ultralytics", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def pick_device(requested: Optional[str]) -> str:
    """选择推理设备: 显式指定 > mps > cpu"""
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def list_images(class_dir: str) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(class_dir)):
        if name.startswith("."):
            continue
        if name.lower().endswith(IMAGE_EXTS):
            full = os.path.join(class_dir, name)
            if os.path.isfile(full):
                out.append(full)
    return out


def load_rgb(path: str):
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


def rel(path: str) -> str:
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return path


# ===========================================================================
# A1: CLIP 近邻检索
# ===========================================================================
class ClipKnn:
    """
    用 CLIP 图像 embedding 做 kNN 近邻投票。

    为什么不用 src/data/search_index/faiss_index.index:
      该索引 ntotal=3793 / dim=1048，条目全部来自 data/merged_english_dataset
      的 57 个类，与本次 12 个目标类**类名空间完全不相交**(覆盖 0/12)，
      且维度与 CLIP ViT-B/32 (512) 不匹配。用它算出的 foreign_ratio 对每张图
      恒等于 1.0，无判别力。因此改为就地对 final_dataset 现建索引。
    """

    def __init__(self, device: str, batch_size: int = 64, workers: int = 8):
        self.device = device
        self.batch_size = batch_size
        self.workers = workers
        self.model = None
        self.preprocess = None
        self.status = "pending"
        self.detail = ""
        self.ref_vecs: Optional[np.ndarray] = None
        self.ref_paths: List[str] = []
        self.ref_classes: List[str] = []

    def load(self) -> bool:
        try:
            import clip
            import torch

            # HF CLIP 缓存不完整 (仅 config.json, 权重为 .no_exist)，
            # 走 OpenAI CLIP 的本地 ViT-B-32.pt (354MB, 完整)
            model, preprocess = clip.load(
                "ViT-B/32", device=self.device, download_root=CLIP_DOWNLOAD_ROOT
            )
            model.eval()
            self.model = model
            self.preprocess = preprocess
            self.status = "ok"
            logger.info("[A1/CLIP] 已加载 ViT-B/32 (offline) device=%s", self.device)
            return True
        except Exception as exc:
            self.status = "unavailable"
            self.detail = f"{type(exc).__name__}: {exc}"
            logger.warning("[A1/CLIP] 不可用，该路降级跳过: %s", self.detail)
            return False

    def encode(self, paths: Sequence[str]) -> np.ndarray:
        """批量编码，返回 L2 归一化后的 (N, 512)。失败图片置零向量。"""
        import torch

        vecs = np.zeros((len(paths), 512), dtype=np.float32)
        pool = ThreadPoolExecutor(max_workers=self.workers)
        try:
            for start in range(0, len(paths), self.batch_size):
                chunk = list(paths[start : start + self.batch_size])

                def _prep(p):
                    try:
                        return self.preprocess(load_rgb(p))
                    except Exception:
                        return None

                tensors = list(pool.map(_prep, chunk))
                keep = [(i, t) for i, t in enumerate(tensors) if t is not None]
                if not keep:
                    continue
                batch = torch.stack([t for _, t in keep]).to(self.device)
                with torch.no_grad():
                    feats = self.model.encode_image(batch).float()
                    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                feats = feats.cpu().numpy().astype(np.float32)
                for slot, (i, _) in enumerate(keep):
                    vecs[start + i] = feats[slot]
                done = min(start + self.batch_size, len(paths))
                if done % (self.batch_size * 20) == 0 or done == len(paths):
                    logger.info("[A1/CLIP] 编码进度 %d/%d", done, len(paths))
        finally:
            pool.shutdown(wait=True)
        return vecs

    def build_reference(self, ref_paths: List[str], ref_classes: List[str]) -> None:
        logger.info("[A1/CLIP] 构建参考索引: %d 张图 / %d 类",
                    len(ref_paths), len(set(ref_classes)))
        t0 = time.time()
        self.ref_vecs = self.encode(ref_paths)
        self.ref_paths = ref_paths
        self.ref_classes = ref_classes
        logger.info("[A1/CLIP] 参考索引完成 (%.1fs)", time.time() - t0)

    def query(self, vec: np.ndarray, self_path: str, own_class: str, k: int) -> Dict[str, Any]:
        """返回近邻统计。vec 为该图 embedding。"""
        if self.ref_vecs is None or not np.any(vec):
            return {"status": "skipped"}
        sims = self.ref_vecs @ vec  # 均已归一化 -> 余弦相似度
        # 取 k+8 再剔除自身与零向量条目
        topn = min(len(sims), k + 8)
        cand = np.argpartition(-sims, topn - 1)[:topn]
        cand = cand[np.argsort(-sims[cand])]
        neigh: List[Tuple[str, float]] = []
        for idx in cand:
            if self.ref_paths[idx] == self_path:
                continue
            if not np.any(self.ref_vecs[idx]):
                continue
            neigh.append((self.ref_classes[idx], float(sims[idx])))
            if len(neigh) >= k:
                break
        if not neigh:
            return {"status": "skipped"}
        foreign = [c for c, _ in neigh if c != own_class]
        ratio = len(foreign) / len(neigh)
        top_foreign, top_foreign_n = None, 0
        if foreign:
            top_foreign, top_foreign_n = Counter(foreign).most_common(1)[0]
        return {
            "status": "ok",
            "k": len(neigh),
            "neighbor_foreign_ratio": round(ratio, 4),
            "top_foreign_class": top_foreign,
            "top_foreign_count": top_foreign_n,
            "mean_sim": round(float(np.mean([s for _, s in neigh])), 4),
            "neighbor_classes": [c for c, _ in neigh],
        }


# ===========================================================================
# A2: v7 EfficientNet-B3 预测
# ===========================================================================
class V7Classifier:
    def __init__(self, device: str, weights: str, name2idx_path: str, batch_size: int = 32):
        self.device = device
        self.weights = weights
        self.name2idx_path = name2idx_path
        self.batch_size = batch_size
        self.model = None
        self.idx2name: Dict[int, str] = {}
        self.status = "pending"
        self.detail = ""
        self.ckpt_meta: Dict[str, Any] = {}
        self.transform = None

    def load(self) -> bool:
        try:
            import torch
            import torchvision.transforms as T
            from torchvision.models import efficientnet_b3

            if not os.path.exists(self.name2idx_path):
                raise FileNotFoundError(self.name2idx_path)
            name2idx = json.load(open(self.name2idx_path))
            self.idx2name = {int(v): k for k, v in name2idx.items()}

            ckpt = torch.load(self.weights, map_location="cpu")
            sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
            self.ckpt_meta = {
                "weights": rel(self.weights),
                "epoch": ckpt.get("epoch"),
                "best_acc": ckpt.get("best_acc"),
            }
            head_w = sd.get("classifier.5.weight")
            if head_w is None:
                raise RuntimeError("checkpoint 缺少 classifier.5.weight")
            num_classes = head_w.shape[0]
            if num_classes != len(self.idx2name):
                raise RuntimeError(
                    f"类数不匹配: ckpt={num_classes} vs name2idx={len(self.idx2name)}"
                )

            model = efficientnet_b3(weights=None)
            in_feat = model.classifier[1].in_features
            # v7 训练时用的自定义分类头: Linear->BN->...->Linear(classifier.5)
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.4, inplace=True),
                torch.nn.Linear(in_feat, 768),
                torch.nn.SiLU(inplace=True),
                torch.nn.BatchNorm1d(768),
                torch.nn.Dropout(p=0.3, inplace=True),
                torch.nn.Linear(768, num_classes),
            )
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                logger.debug("[A2/v7] missing=%s unexpected=%s",
                             list(missing)[:5], list(unexpected)[:5])
                # 分类头权重必须命中，否则预测无意义
                if any(m.startswith("classifier.5") for m in missing):
                    raise RuntimeError("分类头权重未命中")
            model.eval().to(self.device)
            self.model = model

            # 严格复刻 training_results.json 的 eval_pipeline:
            # ensure_rgb -> Resize((288,288)) -> CenterCrop(256) -> ToTensor -> Normalize
            self.transform = T.Compose([
                T.Resize((288, 288), interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(256),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.status = "ok"
            logger.info("[A2/v7] 已加载 %s (epoch=%s best_acc=%s) %d 类",
                        rel(self.weights), self.ckpt_meta.get("epoch"),
                        self.ckpt_meta.get("best_acc"), num_classes)
            return True
        except Exception as exc:
            self.status = "unavailable"
            self.detail = f"{type(exc).__name__}: {exc}"
            logger.warning("[A2/v7] 不可用，该路降级跳过: %s", self.detail)
            return False

    def predict(self, paths: Sequence[str]) -> List[Optional[Dict[str, Any]]]:
        import torch

        results: List[Optional[Dict[str, Any]]] = [None] * len(paths)
        pool = ThreadPoolExecutor(max_workers=8)
        try:
            for start in range(0, len(paths), self.batch_size):
                chunk = list(paths[start : start + self.batch_size])

                def _prep(p):
                    try:
                        return self.transform(load_rgb(p))
                    except Exception:
                        return None

                tensors = list(pool.map(_prep, chunk))
                keep = [(i, t) for i, t in enumerate(tensors) if t is not None]
                if not keep:
                    continue
                batch = torch.stack([t for _, t in keep]).to(self.device)
                with torch.no_grad():
                    probs = torch.softmax(self.model(batch).float(), dim=1)
                    conf, pred = probs.max(dim=1)
                conf = conf.cpu().numpy()
                pred = pred.cpu().numpy()
                for slot, (i, _) in enumerate(keep):
                    results[start + i] = {
                        "pred_class": self.idx2name.get(int(pred[slot]), f"IDX_{pred[slot]}"),
                        "confidence": round(float(conf[slot]), 4),
                    }
        finally:
            pool.shutdown(wait=True)
        return results


# ===========================================================================
# B/C: YOLOv8 检测
# ===========================================================================
class YoloDetector:
    def __init__(self, device: str, weights: str, conf: float = 0.3):
        self.device = device
        self.weights = weights
        self.conf = conf
        self.model = None
        self.status = "pending"
        self.detail = ""

    def load(self) -> bool:
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.weights)
            self.status = "ok"
            logger.info("[B,C/YOLO] 已加载 %s", rel(self.weights))
            return True
        except Exception as exc:
            self.status = "unavailable"
            self.detail = f"{type(exc).__name__}: {exc}"
            logger.warning("[B,C/YOLO] 不可用，该路降级跳过: %s", self.detail)
            return False

    def detect_batch(self, paths: Sequence[str], batch_size: int = 16
                     ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for start in range(0, len(paths), batch_size):
            chunk = list(paths[start : start + batch_size])
            try:
                results = self.model.predict(
                    chunk, verbose=False, device=self.device, conf=self.conf
                )
                out.extend(self._parse(r) for r in results)
            except Exception:
                # 整批失败时退化为逐图，避免一张坏图拖垮整批
                out.extend(self.detect(p) for p in chunk)
        return out

    def _parse(self, res) -> Dict[str, Any]:
        try:
            names = res.names
            h, w = res.orig_shape
            area = float(h * w) or 1.0
            persons, allboxes = [], []
            for b in res.boxes:
                cls_id = int(b.cls)
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                ratio = ((x2 - x1) * (y2 - y1)) / area
                item = {
                    "cls": names.get(cls_id, str(cls_id)),
                    "conf": round(float(b.conf), 4),
                    "bbox_rel": [round(x1 / w, 4), round(y1 / h, 4),
                                 round(x2 / w, 4), round(y2 / h, 4)],
                    "area_ratio": round(ratio, 4),
                }
                allboxes.append(item)
                if cls_id == 0:
                    persons.append(item)
            max_person = max((p["area_ratio"] for p in persons), default=None)
            max_any = max((b["area_ratio"] for b in allboxes), default=None)
            return {
                "status": "ok",
                "person_count": len(persons),
                "persons": persons[:8],
                "box_count": len(allboxes),
                "max_person_area_ratio": max_person,
                "max_any_area_ratio": max_any,
            }
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    def detect(self, path: str) -> Dict[str, Any]:
        try:
            res = self.model.predict(
                path, verbose=False, device=self.device, conf=self.conf
            )[0]
            return self._parse(res)
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


# ===========================================================================
# D: WD ViT v3 风格 / 角色标签
# ===========================================================================
class WdTagger:
    """
    直接复用项目 WDViTV3Tagger 已加载的 timm 模型与标签映射，但**绕过**其
    generate_tags():

      1. wd_vit_v3_tagger.py:953 对多标签模型误用 softmax(dim=1)。WD ViT v3
         是 multi-label 模型，正确激活是 sigmoid。实测同一张图
         softmax>=0.3 只剩 0~1 个标签，sigmoid>=0.3 有 17~38 个标签。
      2. _filter_tags() 是白名单过滤 (仅保留发色/瞳色/人数等属性)，
         comic / monochrome / speech_bubble / sketch 全部会被丢弃，
         D 路会永远为空。

    因此本类自行做 sigmoid + 全量标签阈值化。
    """

    def __init__(self, device: str, threshold: float = 0.35):
        self.device = device
        self.threshold = threshold
        self.tagger = None
        self.model = None
        self.idx2label: Dict[int, str] = {}
        self.char_tags: set = set()
        self.transform = None
        self.status = "pending"
        self.detail = ""

    def load(self) -> bool:
        try:
            import torch
            import torchvision.transforms as T

            from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger

            tg = WDViTV3Tagger(device=self.device)
            tg.load_model()
            if getattr(tg, "wd_model", None) is None:
                raise RuntimeError("wd_model 未加载 (可能回退到简单标签列表)")
            self.tagger = tg
            self.model = tg.wd_model
            self.idx2label = dict(tg.num_id2label)
            size = int(getattr(tg, "img_size", 448))
            self.transform = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            self.char_tags = self._load_character_vocab()
            self.status = "ok"
            logger.info("[D/WD] 已加载 wd-vit-tagger-v3 (%d 标签, %d 角色标签) size=%d",
                        len(self.idx2label), len(self.char_tags), size)
            return True
        except Exception as exc:
            self.status = "unavailable"
            self.detail = f"{type(exc).__name__}: {exc}"
            logger.warning("[D/WD] 不可用，该路降级跳过: %s", self.detail)
            return False

    @staticmethod
    def _load_character_vocab() -> set:
        """从 selected_tags.csv 读 category==4 的角色标签集合"""
        pattern = os.path.expanduser(
            "~/.cache/huggingface/hub/models--SmilingWolf--wd-vit-tagger-v3/"
            "snapshots/*/selected_tags.csv"
        )
        hits = glob.glob(pattern) + glob.glob(
            os.path.join(PROJECT_ROOT, "huggingface_cache/hub/"
                         "models--SmilingWolf--wd-vit-tagger-v3/snapshots/*/selected_tags.csv")
        )
        for path in hits:
            try:
                with open(path, newline="", encoding="utf-8") as fh:
                    return {r["name"] for r in csv.DictReader(fh) if r.get("category") == "4"}
            except Exception:
                continue
        return set()

    def _probs_to_tags(self, probs: np.ndarray) -> Dict[str, Any]:
        hits = np.nonzero(probs >= self.threshold)[0]
        tags: Dict[str, float] = {}
        for i in hits:
            name = self.idx2label.get(int(i))
            if name:
                tags[name] = round(float(probs[i]), 4)
        chars = {t: c for t, c in tags.items() if t in self.char_tags}
        return {
            "status": "ok",
            "n_tags": len(tags),
            "style_tags": {t: tags[t] for t in COMIC_TAGS_PRIMARY if t in tags},
            "style_tags_strict": {t: tags[t] for t in COMIC_TAGS_STRICT if t in tags},
            "style_tags_extra": {t: tags[t] for t in COMIC_TAGS_EXTRA if t in tags},
            "multi_char_tags": {t: tags[t] for t in MULTI_CHAR_TAGS if t in tags},
            "character_tags": dict(sorted(chars.items(), key=lambda kv: -kv[1])[:5]),
            "top_tags": dict(sorted(tags.items(), key=lambda kv: -kv[1])[:12]),
        }

    def tag_batch(self, paths: Sequence[str], batch_size: int = 16, workers: int = 8
                  ) -> List[Dict[str, Any]]:
        import torch

        out: List[Dict[str, Any]] = [
            {"status": "error", "error": "not_processed"} for _ in paths
        ]
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            for start in range(0, len(paths), batch_size):
                chunk = list(paths[start : start + batch_size])

                def _prep(p):
                    try:
                        return self.transform(load_rgb(p))
                    except Exception:
                        return None

                tensors = list(pool.map(_prep, chunk))
                keep = [(i, t) for i, t in enumerate(tensors) if t is not None]
                for i, t in enumerate(tensors):
                    if t is None:
                        out[start + i] = {"status": "error", "error": "decode_failed"}
                if not keep:
                    continue
                batch = torch.stack([t for _, t in keep]).to(self.device)
                with torch.no_grad():
                    probs = torch.sigmoid(self.model(batch).float()).cpu().numpy()
                for slot, (i, _) in enumerate(keep):
                    out[start + i] = self._probs_to_tags(probs[slot])
        finally:
            pool.shutdown(wait=True)
        return out

    def tag(self, path: str) -> Dict[str, Any]:
        return self.tag_batch([path])[0]


# ===========================================================================
# 主审计流程
# ===========================================================================
def collect_targets(data_dir: str, classes: List[str], sample: Optional[int]
                    ) -> Tuple[Dict[str, List[str]], List[str]]:
    per_class: Dict[str, List[str]] = {}
    missing: List[str] = []
    for cls in classes:
        cdir = os.path.join(data_dir, cls)
        if not os.path.isdir(cdir):
            missing.append(cls)
            continue
        imgs = list_images(cdir)
        if sample is not None and sample > 0:
            imgs = imgs[:sample]
        per_class[cls] = imgs
    return per_class, missing


def build_reference_set(data_dir: str, target_classes: List[str], scope: str,
                        per_class_cap: int) -> Tuple[List[str], List[str]]:
    """参考底库: scope=all 用 final_dataset 全部 171 类 (foreign_ratio 才有意义)"""
    if scope == "targets":
        pool = target_classes
    else:
        pool = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
        )
    paths, labels = [], []
    for cls in pool:
        imgs = list_images(os.path.join(data_dir, cls))
        if per_class_cap > 0:
            imgs = imgs[:per_class_cap]
        paths.extend(imgs)
        labels.extend([cls] * len(imgs))
    return paths, labels


def main() -> int:
    ap = argparse.ArgumentParser(
        description="弱类标签质量审计 (只标记, 绝不删除/移动任何图片)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--classes", type=str, default=None,
                    help="逗号分隔类名; 与 --all 二选一")
    ap.add_argument("--all", action="store_true", help="审计 --data-dir 下全部类")
    ap.add_argument("--data-dir", type=str, default="data/final_dataset")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="默认 outputs/label_audit/<时间戳>; 若传入目录则在其下建时间戳子目录")
    ap.add_argument("--device", type=str, default="auto", help="auto|mps|cpu")
    ap.add_argument("--sample", type=int, default=None, help="每类抽样上限")
    # 阈值
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--foreign-ratio-thr", type=float, default=0.6)
    ap.add_argument("--model-conf-thr", type=float, default=0.7)
    ap.add_argument("--person-conf", type=float, default=0.3)
    ap.add_argument("--small-area-thr", type=float, default=0.15)
    ap.add_argument("--wd-threshold", type=float, default=0.35)
    # 参考底库
    ap.add_argument("--ref-scope", type=str, default="all", choices=("all", "targets"),
                    help="CLIP 近邻底库范围")
    ap.add_argument("--ref-per-class", type=int, default=0,
                    help="底库每类图片上限, 0=不限")
    # 开关
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-yolo", action="store_true")
    ap.add_argument("--skip-wd", action="store_true")
    ap.add_argument("--v7-weights", type=str, default=V7_FULL)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    os.chdir(PROJECT_ROOT)

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        logger.error("数据目录不存在: %s", data_dir)
        return 2

    if args.all:
        classes = sorted(d for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("."))
    elif args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        logger.error("必须指定 --classes 或 --all")
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("outputs", "label_audit")
    # 若传入的是父目录 (或默认)，统一在其下建时间戳子目录
    if not os.path.basename(out_dir).replace("_", "").isdigit():
        out_dir = os.path.join(out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    logger.info("输出目录: %s", out_dir)

    device = pick_device(args.device)
    logger.info("设备: %s | 目标类: %d | 抽样: %s", device, len(classes), args.sample or "全扫")

    per_class, missing = collect_targets(data_dir, classes, args.sample)
    if missing:
        logger.warning("以下类目录不存在, 已跳过: %s", missing)
    total_imgs = sum(len(v) for v in per_class.values())
    if total_imgs == 0:
        logger.error("没有可审计的图片")
        return 2
    logger.info("待审计图片: %d 张", total_imgs)

    t_start = time.time()
    degradations: List[str] = []

    # ---------------- 加载各路检测器 ----------------
    clip_knn = None
    if not args.skip_clip:
        clip_knn = ClipKnn(device=device)
        if clip_knn.load():
            ref_paths, ref_labels = build_reference_set(
                data_dir, list(per_class.keys()), args.ref_scope, args.ref_per_class
            )
            clip_knn.build_reference(ref_paths, ref_labels)
        else:
            degradations.append(f"A1(CLIP kNN) 不可用: {clip_knn.detail}")
    else:
        degradations.append("A1(CLIP kNN) 被 --skip-clip 关闭")

    v7 = None
    if not args.skip_model:
        v7 = V7Classifier(device=device, weights=args.v7_weights, name2idx_path=V7_NAME2IDX)
        if not v7.load():
            degradations.append(f"A2(v7 预测) 不可用: {v7.detail}")
    else:
        degradations.append("A2(v7 预测) 被 --skip-model 关闭")

    yolo = None
    if not args.skip_yolo:
        # YOLO 固定 cpu: ultralytics 在 mps 上对部分动漫图会触发算子回退，反而更慢
        yolo = YoloDetector(device="cpu", weights=YOLO_WEIGHTS, conf=args.person_conf)
        if not yolo.load():
            degradations.append(f"B,C(YOLO) 不可用: {yolo.detail}")
    else:
        degradations.append("B,C(YOLO) 被 --skip-yolo 关闭")

    wd = None
    if not args.skip_wd:
        wd = WdTagger(device=device, threshold=args.wd_threshold)
        if not wd.load():
            degradations.append(f"D(WD 风格标签) 不可用: {wd.detail}")
    else:
        degradations.append("D(WD 风格标签) 被 --skip-wd 关闭")

    # ---------------- 逐类审计 ----------------
    records: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, Any]] = {}

    for cls, paths in per_class.items():
        if not paths:
            continue
        logger.info("── 审计 %s (%d 张)", cls, len(paths))
        t_cls = time.time()

        clip_vecs = clip_knn.encode(paths) if (clip_knn and clip_knn.status == "ok") else None
        v7_preds = v7.predict(paths) if (v7 and v7.status == "ok") else [None] * len(paths)
        yolo_dets = (yolo.detect_batch(paths) if (yolo and yolo.status == "ok")
                     else [{"status": "skipped"}] * len(paths))
        wd_dets = (wd.tag_batch(paths) if (wd and wd.status == "ok")
                   else [{"status": "skipped"}] * len(paths))

        wd_alias = WD_CHARACTER_ALIAS.get(cls)

        for i, path in enumerate(paths):
            det: Dict[str, Any] = {}

            # --- A1 CLIP kNN ---
            knn = {"status": "skipped"}
            if clip_vecs is not None:
                knn = clip_knn.query(clip_vecs[i], path, cls, args.knn_k)
            det["clip_knn"] = knn

            # --- A2 v7 ---
            pred = v7_preds[i] if i < len(v7_preds) else None
            if pred is None:
                det["model_pred"] = {"status": "skipped"}
            else:
                det["model_pred"] = {
                    "status": "ok",
                    "pred_class": pred["pred_class"],
                    "confidence": pred["confidence"],
                    "matches_folder": pred["pred_class"] == cls,
                }

            # --- B/C YOLO ---
            ydet = yolo_dets[i] if i < len(yolo_dets) else {"status": "skipped"}
            det["yolo"] = ydet

            # --- D WD ---
            wdet = wd_dets[i] if i < len(wd_dets) else {"status": "skipped"}
            det["wd"] = wdet

            # ================= flags =================
            flags: Dict[str, Any] = {}

            # A: 错标嫌疑 (主判据: CLIP 近邻异类占比)
            mislabel = False
            suspect_class = None
            foreign_ratio = None
            if knn.get("status") == "ok":
                foreign_ratio = knn["neighbor_foreign_ratio"]
                if foreign_ratio > args.foreign_ratio_thr and knn.get("top_foreign_class"):
                    mislabel = True
                    suspect_class = knn["top_foreign_class"]
            flags["mislabel_suspect"] = mislabel

            # A2 旁证
            mm = det["model_pred"]
            flags["model_mislabel_suspect"] = bool(
                mm.get("status") == "ok"
                and not mm.get("matches_folder")
                and mm.get("confidence", 0) > args.model_conf_thr
            )

            # A3 旁证: WD 角色标签与文件夹类名不符
            wd_char_status = "unmapped" if not wd_alias else "ok"
            wd_char_mismatch = False
            wd_top_char = None
            if wdet.get("status") == "ok" and wd_alias:
                chars = wdet.get("character_tags") or {}
                if chars:
                    wd_top_char = max(chars, key=chars.get)
                    wd_char_mismatch = wd_alias not in chars
                else:
                    wd_char_status = "no_character_tag"
            elif not wd_alias:
                wd_char_status = "unmapped"
            det["wd_character_check"] = {
                "status": wd_char_status,
                "expected_tag": wd_alias,
                "top_character_tag": wd_top_char,
            }
            flags["wd_char_mismatch"] = bool(wd_char_mismatch)

            # B: 多人图
            person_count = ydet.get("person_count") if ydet.get("status") == "ok" else None
            flags["multi_person"] = bool(person_count is not None and person_count >= 2)
            # B 补充: 动漫域多角色标签
            mct = wdet.get("multi_char_tags") if wdet.get("status") == "ok" else None
            flags["multi_character_tag"] = bool(mct)

            # C: 小角色
            small = False
            area_ratio = None
            area_src = None
            if ydet.get("status") == "ok":
                if ydet.get("max_person_area_ratio") is not None:
                    area_ratio = ydet["max_person_area_ratio"]
                    area_src = "person"
                elif ydet.get("max_any_area_ratio") is not None:
                    area_ratio = ydet["max_any_area_ratio"]
                    area_src = "any_box"
                if area_ratio is not None and area_ratio < args.small_area_thr:
                    small = True
            det["body_box"] = {
                "body_box_area_ratio": area_ratio,
                "source": area_src or ("none" if ydet.get("status") == "ok" else "skipped"),
            }
            flags["small_character"] = small

            # D: 漫画风 (comic_style 为规格口径, 含 multiple_girls;
            #    comic_style_strict 剔除 multiple_girls, 才是真正的域偏移信号)
            stags = wdet.get("style_tags") if wdet.get("status") == "ok" else None
            flags["comic_style"] = bool(stags)
            flags["comic_style_strict"] = bool(
                wdet.get("style_tags_strict") if wdet.get("status") == "ok" else None
            )

            core = ("mislabel_suspect", "multi_person", "small_character", "comic_style")
            severity = sum(1 for k in core if flags.get(k))

            records.append({
                "path": rel(path),
                "class": cls,
                "detections": det,
                "flags": flags,
                "severity": severity,
            })

        # ---- 每类汇总 ----
        sub = [r for r in records if r["class"] == cls]
        n = len(sub)

        def cnt(key: str) -> int:
            return sum(1 for r in sub if r["flags"].get(key))

        summary[cls] = {
            "n_images": n,
            "mislabel_suspect": cnt("mislabel_suspect"),
            "model_mislabel_suspect": cnt("model_mislabel_suspect"),
            "wd_char_mismatch": cnt("wd_char_mismatch"),
            "multi_person": cnt("multi_person"),
            "multi_character_tag": cnt("multi_character_tag"),
            "small_character": cnt("small_character"),
            "comic_style": cnt("comic_style"),
            "comic_style_strict": cnt("comic_style_strict"),
            "clean": sum(1 for r in sub if r["severity"] == 0),
            "severity_ge2": sum(1 for r in sub if r["severity"] >= 2),
            "ratios": {
                k: round(cnt(k) / n, 4) if n else 0.0
                for k in ("mislabel_suspect", "model_mislabel_suspect", "wd_char_mismatch",
                          "multi_person", "multi_character_tag", "small_character",
                          "comic_style", "comic_style_strict")
            },
            "elapsed_sec": round(time.time() - t_cls, 1),
        }
        s = summary[cls]
        logger.info(
            "   %s: 错标%d 多人%d(标签%d) 小角色%d 漫画风%d | 干净%d/%d (%.1fs)",
            cls, s["mislabel_suspect"], s["multi_person"], s["multi_character_tag"],
            s["small_character"], s["comic_style"], s["clean"], n, s["elapsed_sec"],
        )

    elapsed = time.time() - t_start

    # ---------------- 汇总 & 落盘 ----------------
    def total(key: str) -> int:
        return sum(v[key] for v in summary.values())

    overall = {
        "n_images": len(records),
        "n_classes": len(summary),
        **{k: total(k) for k in ("mislabel_suspect", "model_mislabel_suspect",
                                 "wd_char_mismatch", "multi_person", "multi_character_tag",
                                 "small_character", "comic_style", "comic_style_strict",
                                 "clean", "severity_ge2")},
    }

    report = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_dir": data_dir,
            "device": device,
            "sample_per_class": args.sample,
            "full_scan": args.sample is None,
            "elapsed_sec": round(elapsed, 1),
            "classes_requested": classes,
            "classes_missing": missing,
            "thresholds": {
                "knn_k": args.knn_k,
                "foreign_ratio_thr": args.foreign_ratio_thr,
                "model_conf_thr": args.model_conf_thr,
                "person_conf": args.person_conf,
                "small_area_thr": args.small_area_thr,
                "wd_threshold": args.wd_threshold,
            },
            "detector_status": {
                "A1_clip_knn": clip_knn.status if clip_knn else "disabled",
                "A2_v7_model": v7.status if v7 else "disabled",
                "B_C_yolo": yolo.status if yolo else "disabled",
                "D_wd_tagger": wd.status if wd else "disabled",
            },
            "v7_checkpoint": v7.ckpt_meta if (v7 and v7.status == "ok") else None,
            "clip_reference": {
                "scope": args.ref_scope,
                "n_images": len(clip_knn.ref_paths) if (clip_knn and clip_knn.ref_vecs is not None) else 0,
                "n_classes": len(set(clip_knn.ref_classes)) if (clip_knn and clip_knn.ref_vecs is not None) else 0,
            },
            "degradations": degradations,
            "guarantee": "read-only: 未删除/移动/修改任何图片",
        },
        "overall": overall,
        "per_class": summary,
        "images": records,
    }

    report_path = os.path.join(out_dir, "audit_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # ---- 可疑清单 (给人工裁决) ----
    manifest: Dict[str, Any] = {
        "meta": {
            "generated_at": report["meta"]["generated_at"],
            "source_report": os.path.basename(report_path),
            "note": "仅可疑清单, 需人工裁决。脚本不做任何删除/移动。",
        },
        "mislabel_candidates": [],
        "multi_person": [],
        "small_character": [],
        "comic_style": [],
    }
    for r in records:
        f, d = r["flags"], r["detections"]
        if f.get("mislabel_suspect"):
            k = d["clip_knn"]
            manifest["mislabel_candidates"].append({
                "path": r["path"], "class": r["class"],
                "suspect_class": k.get("top_foreign_class"),
                "neighbor_foreign_ratio": k.get("neighbor_foreign_ratio"),
                "model_pred": d["model_pred"].get("pred_class"),
                "model_conf": d["model_pred"].get("confidence"),
                "wd_top_character": d["wd_character_check"].get("top_character_tag"),
                "severity": r["severity"],
            })
        if f.get("multi_person"):
            manifest["multi_person"].append({
                "path": r["path"], "class": r["class"],
                "person_count": d["yolo"].get("person_count"),
                "boxes": [p["bbox_rel"] for p in d["yolo"].get("persons", [])],
                "wd_multi_char_tags": list((d["wd"].get("multi_char_tags") or {}).keys()),
                "severity": r["severity"],
            })
        if f.get("small_character"):
            manifest["small_character"].append({
                "path": r["path"], "class": r["class"],
                "body_box_area_ratio": d["body_box"]["body_box_area_ratio"],
                "source": d["body_box"]["source"],
                "severity": r["severity"],
            })
        if f.get("comic_style"):
            manifest["comic_style"].append({
                "path": r["path"], "class": r["class"],
                "style_tags": list((d["wd"].get("style_tags") or {}).keys()),
                "style_tags_strict": list((d["wd"].get("style_tags_strict") or {}).keys()),
                "style_tags_extra": list((d["wd"].get("style_tags_extra") or {}).keys()),
                "is_strict_comic": bool(f.get("comic_style_strict")),
                "severity": r["severity"],
            })
    manifest["counts"] = {k: len(v) for k, v in manifest.items() if isinstance(v, list)}
    manifest["counts"]["suspect_images_total"] = sum(1 for r in records if r["severity"] > 0)

    manifest_path = os.path.join(out_dir, "suspect_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    # ---------------- 控制台摘要 ----------------
    print("\n" + "=" * 94)
    print("弱类标签质量审计摘要  (只标记, 未删除/移动任何图片)")
    print("=" * 94)
    print(f"{'类名':<22}{'图数':>5}{'错标':>6}{'多人':>6}{'多角色':>7}"
          f"{'小角色':>7}{'漫画风':>7}{'纯漫画':>7}{'干净':>6}")
    print("-" * 94)
    for cls, s in summary.items():
        print(f"{cls:<22}{s['n_images']:>5}{s['mislabel_suspect']:>6}{s['multi_person']:>6}"
              f"{s['multi_character_tag']:>7}{s['small_character']:>7}"
              f"{s['comic_style']:>7}{s['comic_style_strict']:>7}{s['clean']:>6}")
    print("-" * 94)
    print(f"{'合计':<22}{overall['n_images']:>5}{overall['mislabel_suspect']:>6}"
          f"{overall['multi_person']:>6}{overall['multi_character_tag']:>7}"
          f"{overall['small_character']:>7}{overall['comic_style']:>7}"
          f"{overall['comic_style_strict']:>7}{overall['clean']:>6}")
    print("=" * 94)
    print(f"可疑图片总数 (severity>=1): {manifest['counts']['suspect_images_total']} / {overall['n_images']}")
    print(f"耗时 {elapsed:.1f}s | 设备 {device}")
    if degradations:
        print("\n降级/跳过:")
        for d in degradations:
            print(f"  - {d}")
    print(f"\n报告:   {report_path}")
    print(f"可疑清单: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
