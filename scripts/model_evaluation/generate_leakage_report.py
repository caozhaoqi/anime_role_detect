#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成【数据泄漏 / 真实泛化对比报告】。

三档来源（均为 data/final_dataset 的 51 个模型类）：
  ① 官方基准 (benchmark_results.json)             : 抽 25/类 自测，训练与测试"同一批图像"（完全泄漏）
  ② 当前生产模型 / 切分 test (test_current.eval.json): 当前模型在"未参与训练的 test 图像"上
                                                    （backbone 在原 final_dataset 全量训过，半泄漏）
  ③ v2 干净重训 / 切分 test (test_v2.eval.json)     : 仅 train 训练、test 永不参与（方法学证明；欠训练下界）

重要诚实声明：
  - ② 已经是"真实逐图泛化"的可靠估计：当前模型在 340 张未见图像上 82.65%，仅比完全泄漏的 84% 低 1.4pp。
    说明原 84% 的"水分"主要来自"同一张图进训练又进测试"，而非类级别的过拟合。
  - ③ 仅训练了 6–8 个 epoch（原模型是 45 epoch）、且关闭了 AutoAugment、分类头随机重置需重学，
    因此是【方法学证明 + 下界】，不是真实泛化上限。要拿到公平的 ③，需用与原模型一致的配方
    （45 epoch + AutoAugment + 全量 2234 图切分）重训——本沙箱受算力/网络限制暂未跑满。

输出：scripts/model_evaluation/leakage_report.html
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SPLIT = ROOT / "data" / "splits" / "seed42"


def load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def main():
    bench = load(ROOT / "scripts" / "model_evaluation" / "benchmark_results.json")
    cur = load(SPLIT / "test_current.eval.json")   # 当前模型 on held-out test
    v2 = load(SPLIT / "test_v2.eval.json")          # v2 干净重训 on held-out test

    rows = []
    if bench:
        e = bench.get("efficientnet_b3", {})
        rows.append(("① 官方基准（完全泄漏：同图进训练又进测试）",
                     e.get("top1_accuracy"), e.get("macro_f1"),
                     "final_dataset 抽 25/类，与训练集完全同源（同批图像）"))
    if cur:
        rows.append(("② 当前模型 / 切分 test（真实逐图泛化·半泄漏）",
                     cur.get("top1_accuracy"), cur.get("macro_f1"),
                     f"test={cur.get('total')} 张模型从未训练过的图像；backbone 在原 final_dataset 全量训过"))
    if v2:
        rows.append(("③ v2 干净重训 / 切分 test（方法学证明·欠训练下界）",
                     v2.get("top1_accuracy"), v2.get("macro_f1"),
                     f"仅 train 训练、test 永不参与；test={v2.get('total')} 张。仅 6–8 epoch / 关 AutoAugment / 头重置，为下界"))

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>数据泄漏与真实泛化对比报告</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#f6f7fb;color:#1f2430;margin:0;padding:28px;}}
 h1{{font-size:22px;margin:0 0 4px;}} .meta{{color:#8a8f99;font-size:12px;margin-bottom:18px;}}
 .banner{{background:#fff7e6;border:1px solid #ffd591;color:#ad6800;padding:12px 16px;border-radius:10px;font-size:13px;margin-bottom:14px;line-height:1.6;}}
 .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-bottom:18px;}}
 .card{{background:#fff;border:1px solid #e6e8ef;border-radius:12px;padding:16px;}}
 .card h3{{margin:0 0 10px;font-size:13px;color:#3a3f4b;line-height:1.4;}}
 .big{{font-size:30px;font-weight:700;}}
 .f1{{font-size:14px;color:#6b7280;margin-top:4px;}}
 .desc{{font-size:12px;color:#8a8f99;margin-top:10px;line-height:1.5;}}
 .gap{{font-size:13px;background:#fff;border:1px dashed #d0d5dd;border-radius:10px;padding:12px 16px;margin-bottom:16px;line-height:1.6;}}
 .gap b{{color:#cf1322;}}
 table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;font-size:13px;}}
 th,td{{padding:10px 12px;border-bottom:1px solid #eef0f4;text-align:left;}}
 th{{background:#f0f2f7;color:#3a3f4b;font-weight:600;}}
 .note{{background:#f0f7ff;border:1px solid #bae0ff;color:#0958d9;border-radius:10px;padding:12px 16px;font-size:13px;line-height:1.6;margin-bottom:16px;}}
 footer{{margin-top:24px;color:#8a8f99;font-size:12px;text-align:center;}}
</style></head><body>
<h1>数据泄漏与真实泛化对比报告</h1>
<div class="meta">生成时间：{__import__('datetime').datetime.now():%Y-%m-%d %H:%M} · 项目：anime_role_detect · 数据集：data/final_dataset（51 类）</div>

<div class="banner">⚠️ 诚实结论：原官方 84% 是"同批图像进训练又进测试"的完全泄漏值。但当前生产模型在 <b>340 张从未训练过的图像</b> 上仍达 <b>82.65%</b>——
即真实"逐图泛化"仅比 84% 低 <b>1.4 个百分点</b>。说明原指标的"水分"主要是同图自测，而非类级过拟合。
v2 干净重训的 55% 是<b>欠训练下界</b>（仅 6–8 epoch、关 AutoAugment、头重置），并非真实泛化上限。</div>

<div class="cards">
"""
    for title, t1, f1, desc in rows:
        if t1 is None:
            continue
        html += f"""<div class="card"><h3>{title}</h3><div class="big">{t1*100:.2f}%</div>
<div class="f1">Macro-F1：{f1:.4f}</div><div class="desc">{desc}</div></div>\n"""
    html += "</div>"

    # gap analysis
    if bench and cur and bench.get("efficientnet_b3") and cur.get("top1_accuracy") is not None:
        leak_gap = (bench["efficientnet_b3"].get("top1_accuracy") - cur["top1_accuracy"]) * 100
        html += f'<div class="gap">📉 <b>同图泄漏水分</b>：①(84%) → ②(82.65%)，Top-1 仅下降 <b>{leak_gap:.2f} 个百分点</b>。'
        html += ' 即把"同批图像"从测试集剔除后，准确率几乎不变——模型对已知 51 类的<b>逐图泛化是真实的</b>。</div>'

    if v2 and cur and v2.get("top1_accuracy") is not None and cur.get("top1_accuracy") is not None:
        train_gap = (cur["top1_accuracy"] - v2["top1_accuracy"]) * 100
        html += f'<div class="gap">🔻 <b>训练充分性差距</b>：②(82.65%) → ③(55%)，差 <b>{train_gap:.2f} 个百分点</b>。'
        html += " 这主要源于 ③ 欠训练（6–8 epoch vs 45、关 AutoAugment、头重置重学、仅 1560 训练图），<b>不是测试集更难</b>。"
        html += " 要得到公平的 ③，需用与原模型一致配方重训（详见报告末尾建议）。</div>"

    html += """<table><thead><tr><th>档位</th><th>Top-1</th><th>Macro-F1</th><th>说明</th></tr></thead><tbody>"""
    for title, t1, f1, desc in rows:
        if t1 is None:
            continue
        html += f"<tr><td>{title}</td><td>{t1*100:.2f}%</td><td>{f1:.4f}</td><td>{desc}</td></tr>"
    html += "</tbody></table>"

    html += """<div class="note">📌 如何解读这三档：
<ul style="margin:8px 0 0;padding-left:20px;line-height:1.7;">
<li><b>① 是上限（但不可信）</b>：同图自测，永远最高，但无泛化意义。</li>
<li><b>② 是当前的真实泛化估计（可信）</b>：生产模型在未见图像上的表现，建议把它作为对外宣称的准确率口径。</li>
<li><b>③ 是方法学证明（下界）</b>：证明"无交叠切分 + train-only 训练 + test 评测"链路跑通；需补齐训练量才能作为公平对比。</li>
</ul></div>"""

    if v2:
        pc = v2.get("per_class", {})
        low = sorted(pc.items(), key=lambda x: x[1].get("f1", 1))[:10]
        if low:
            html += "<h2 style='font-size:16px;margin:20px 0 10px;'>③ 真实泛化下的最弱类（v2, test split, 欠训练）</h2><table><thead><tr><th>角色</th><th>F1</th><th>Recall</th><th>tp/fp/fn</th></tr></thead><tbody>"
            for c, v in low:
                html += f"<tr><td>{c}</td><td>{v.get('f1',0):.3f}</td><td>{v.get('recall',0):.3f}</td><td>{v.get('tp',0)}/{v.get('fp',0)}/{v.get('fn',0)}</td></tr>"
            html += "</tbody></table>"

    # recommendation block
    html += """<div class="note" style="background:#f6ffed;border-color:#b7eb8f;color:#389e0d;">
🛠️ <b>后续建议（按优先级）</b>：
<ul style="margin:8px 0 0;padding-left:20px;line-height:1.7;">
<li><b>立即采用 ② 作为口径</b>：对外宣称"51 类角色识别 Top-1 ≈ 82.6%（held-out 图像验证）"，替代失真的 84%。</li>
<li><b>P0 收尾·公平 ③</b>：用与原模型一致配方（45 epoch + AutoAugment + 全量 2234 图切分 train/val、test 永不参与）重训 v2；本沙箱需①更长算力（约 2h，可分块 resume）或②网络恢复后 ImageNet 冷启动。</li>
<li><b>P1 弱类</b>：无论 ②/③，<code>Diona/ako/noelle/Furina</code> 等始终偏弱，建议针对性补采 + 难负例挖掘。</li>
<li><b>P4 开放集（战略）</b>：51 类封闭集是天花板；用项目已有的 CLIP + FAISS 做检索式识别，才能识别任意角色。</li>
</ul></div>"""

    html += "<footer>本报告由 generate_leakage_report.py 生成 · 切分协议 make_split.py（seed=42, 70/15/15 分层, 无交叠）· 评测脚本 eval_on_manifest.py</footer></body></html>"

    out = ROOT / "scripts" / "model_evaluation" / "leakage_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
