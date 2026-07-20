#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 benchmark_results.json 生成可视化 HTML 报告
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"
OUTPUT_PATH = Path(__file__).parent / "benchmark_report.html"


def fmt(v, suffix=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}{suffix}"
    return f"{v}{suffix}"


def pct(v):
    if v is None:
        return "—"
    return f"{v * 100:.2f}%"


def generate_html(data):
    eff = data.get("efficientnet_b3", {}) or {}
    yolo = data.get("yolov8n", {}) or {}

    # ---- EfficientNet 每类准确率（用于条形图） ----
    per_class = eff.get("per_class_accuracy", {}) or {}
    # 按准确率升序排，便于发现弱类
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1].get("accuracy", 0))
    # 仅展示最弱 15 + 最强 15，避免图表过密
    weak = sorted_classes[:15]
    strong = sorted_classes[-15:][::-1]
    display = weak + strong  # 30 项

    bar_labels = [c for c, _ in display]
    bar_values = [round(v.get("accuracy", 0) * 100, 2) for _, v in display]
    bar_totals = [v.get("total", 0) for _, v in display]
    bar_corrects = [v.get("correct", 0) for _, v in display]

    # ---- Top confused pairs ----
    confused = eff.get("top_confused_pairs", []) or []

    # ---- best/worst classes ----
    best = eff.get("best_5_classes", []) or []
    worst = eff.get("worst_5_classes", []) or []

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Anime Role Detect - 模型基准测试报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Helvetica Neue", Arial, sans-serif;
    margin: 0; padding: 32px; background: #f7f8fa; color: #1f2329; line-height: 1.6;
  }}
  .container {{ max-width: 1180px; margin: 0 auto; }}
  h1 {{ font-size: 28px; margin: 0 0 4px 0; }}
  h2 {{ font-size: 22px; margin: 32px 0 12px 0; border-left: 4px solid #4f7cff; padding-left: 10px; }}
  h3 {{ font-size: 16px; margin: 20px 0 8px 0; color: #4f7cff; }}
  .meta {{ color: #8a8f99; font-size: 13px; margin-bottom: 24px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 16px 0; }}
  .card {{
    background: #fff; border-radius: 10px; padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #eef0f3;
  }}
  .card .label {{ font-size: 12px; color: #8a8f99; margin-bottom: 6px; }}
  .card .value {{ font-size: 22px; font-weight: 600; color: #1f2329; }}
  .card .sub {{ font-size: 12px; color: #8a8f99; margin-top: 4px; }}
  .card.accent .value {{ color: #4f7cff; }}
  .card.good .value {{ color: #16a34a; }}
  .card.warn .value {{ color: #d97706; }}
  .card.bad .value {{ color: #dc2626; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 10px; overflow: hidden;
           box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #eef0f3; font-size: 13px; }}
  th, td {{ padding: 9px 12px; text-align: left; border-bottom: 1px solid #f0f1f3; }}
  th {{ background: #fafbfc; font-weight: 600; color: #4f5b66; }}
  tr:last-child td {{ border-bottom: none; }}
  .chart-box {{ background: #fff; border-radius: 10px; padding: 18px; border: 1px solid #eef0f3;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin: 12px 0; }}
  .chart-box canvas {{ max-height: 420px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 768px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; background: #eef2ff; color: #4f7cff; }}
  .err {{ background: #fef2f2; color: #b91c1c; padding: 10px 14px; border-radius: 8px; font-size: 13px; }}
  footer {{ margin-top: 32px; color: #8a8f99; font-size: 12px; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <h1>Anime Role Detect · 模型基准测试报告</h1>
  <div class="meta">生成时间：{data.get('benchmark_at', '—')} · 项目：{data.get('project', 'anime_role_detect')}</div>

  <!-- ============ EfficientNet-B3 ============ -->
  <h2>① EfficientNet-B3 分类模型</h2>
"""

    if "error" in eff:
        html += f'  <div class="err">测试失败: {eff["error"]}</div>\n'
    else:
        html += f"""
  <div class="cards">
    <div class="card accent"><div class="label">Top-1 准确率</div><div class="value">{pct(eff.get('top1_accuracy'))}</div><div class="sub">{eff.get('test_samples', 0)} 样本</div></div>
    <div class="card accent"><div class="label">Top-5 准确率</div><div class="value">{pct(eff.get('top5_accuracy'))}</div><div class="sub">51 类别</div></div>
    <div class="card"><div class="label">Macro-F1</div><div class="value">{fmt(eff.get('macro_f1'))}</div><div class="sub">P={fmt(eff.get('macro_precision'))} R={fmt(eff.get('macro_recall'))}</div></div>
    <div class="card"><div class="label">单张延迟</div><div class="value">{fmt(eff.get('single_latency_ms'), ' ms')}</div><div class="sub">{fmt(eff.get('single_fps'), ' FPS')}</div></div>
    <div class="card"><div class="label">Batch({eff.get('batch_size', 32)}) 吞吐</div><div class="value">{fmt(eff.get('batch_fps'), ' FPS')}</div><div class="sub">{fmt(eff.get('batch_latency_ms'), ' ms/batch')}</div></div>
    <div class="card"><div class="label">参数量</div><div class="value">{eff.get('total_parameters', 0):,}</div><div class="sub">可训练 {eff.get('trainable_parameters', 0):,}</div></div>
    <div class="card"><div class="label">模型大小</div><div class="value">{fmt(eff.get('model_size_mb'), ' MB')}</div><div class="sub">权重内存 {fmt(eff.get('param_memory_mb'), ' MB')}</div></div>
    <div class="card"><div class="label">进程内存 (RSS)</div><div class="value">{fmt(eff.get('process_rss_mb'), ' MB')}</div><div class="sub">设备: {eff.get('device', '—')}</div></div>
  </div>

  <h3>每类准确率分布（最弱 15 + 最强 15，共 30 类）</h3>
  <div class="chart-box"><canvas id="perClassChart"></canvas></div>

  <div class="grid2">
    <div>
      <h3>✅ 表现最佳的 5 个类别</h3>
      <table>
        <thead><tr><th>类别</th><th>正确/总数</th><th>准确率</th></tr></thead>
        <tbody>
"""
        for c in best:
            html += f'          <tr><td>{c.get("class", "")}</td><td>{c.get("correct", 0)}/{c.get("total", 0)}</td><td>{pct(c.get("accuracy"))}</td></tr>\n'
        html += """        </tbody>
      </table>
    </div>
    <div>
      <h3>⚠️ 表现最差的 5 个类别</h3>
      <table>
        <thead><tr><th>类别</th><th>正确/总数</th><th>准确率</th></tr></thead>
        <tbody>
"""
        for c in worst:
            html += f'          <tr><td>{c.get("class", "")}</td><td>{c.get("correct", 0)}/{c.get("total", 0)}</td><td>{pct(c.get("accuracy"))}</td></tr>\n'
        html += """        </tbody>
      </table>
    </div>
  </div>

  <h3>🔀 最易混淆的类别对（真实 → 预测）</h3>
  <table>
    <thead><tr><th>#</th><th>真实类别</th><th>被误判为</th><th>次数</th></tr></thead>
    <tbody>
"""
        for i, pair in enumerate(confused[:10], 1):
            html += f'      <tr><td>{i}</td><td>{pair.get("true", "")}</td><td>{pair.get("pred", "")}</td><td>{pair.get("count", 0)}</td></tr>\n'
        if not confused:
            html += '      <tr><td colspan="4" style="text-align:center;color:#8a8f99;">无混淆记录</td></tr>\n'
        html += """    </tbody>
  </table>
"""

    # ---- 速度对比小图 ----
    if "error" not in eff:
        html += """
  <h3>⚡ 推理性能对比</h3>
  <div class="chart-box"><canvas id="speedChart"></canvas></div>
"""

    # ============ YOLOv8n ============
    html += "\n  <h2>② YOLOv8n 检测模型</h2>\n"
    if "error" in yolo:
        html += f'  <div class="err">测试失败: {yolo["error"]}</div>\n'
    elif yolo.get("fps") is None:
        html += f'  <div class="err">未完成测试</div>\n'
    else:
        html += f"""
  <div class="cards">
    <div class="card accent"><div class="label">平均延迟</div><div class="value">{fmt(yolo.get('avg_latency_ms'), ' ms')}</div><div class="sub">单张图像</div></div>
    <div class="card accent"><div class="label">吞吐</div><div class="value">{fmt(yolo.get('fps'), ' FPS')}</div><div class="sub">{yolo.get('test_samples', 0)} 样本</div></div>
    <div class="card"><div class="label">参数量</div><div class="value">{yolo.get('total_parameters', 0):,}</div><div class="sub">YOLOv8n 预训练</div></div>
    <div class="card"><div class="label">模型大小</div><div class="value">{fmt(yolo.get('model_size_mb'), ' MB')}</div><div class="sub">yolov8n.pt</div></div>
    <div class="card"><div class="label">加载耗时</div><div class="value">{fmt(yolo.get('load_time_seconds'), ' s')}</div><div class="sub">设备: {yolo.get('device', '—')}</div></div>
    <div class="card"><div class="label">平均检测数/图</div><div class="value">{fmt(yolo.get('avg_detections_per_image'))}</div><div class="sub">平均置信度 {fmt(yolo.get('avg_confidence'))}</div></div>
  </div>
"""

    # ============ 结论 ============
    html += "\n  <h2>③ 关键发现与建议</h2>\n  <div class=\"cards\">\n"
    if "error" not in eff:
        top1 = eff.get("top1_accuracy", 0)
        if top1 < 0.5:
            html += '    <div class="card bad"><div class="label">分类准确率</div><div class="value">偏低</div><div class="sub">Top-1 &lt; 50%，存在明显过拟合（训练集 ~79% vs 验证 ~49%）。建议：解冻 backbone 微调、增加数据增强、扩大数据集。</div></div>\n'
        elif top1 < 0.7:
            html += '    <div class="card warn"><div class="label">分类准确率</div><div class="value">中等</div><div class="sub">Top-1 50-70%，仍有提升空间。</div></div>\n'
        else:
            html += '    <div class="card good"><div class="label">分类准确率</div><div class="value">良好</div><div class="sub">Top-1 ≥ 70%。</div></div>\n'

        # 单张延迟评估
        lat = eff.get("single_latency_ms", 999)
        if lat < 50:
            html += '    <div class="card good"><div class="label">推理延迟</div><div class="value">实时</div><div class="sub">单张 &lt; 50 ms，满足实时场景。</div></div>\n'
        elif lat < 150:
            html += '    <div class="card warn"><div class="label">推理延迟</div><div class="value">可接受</div><div class="sub">单张 50-150 ms。</div></div>\n'
        else:
            html += '    <div class="card bad"><div class="label">推理延迟</div><div class="value">偏慢</div><div class="sub">单张 &gt; 150 ms，建议量化或换轻量模型。</div></div>\n'

        # 混淆类别
        if confused:
            html += f'    <div class="card"><div class="label">主要混淆</div><div class="value">{confused[0].get("true", "")} → {confused[0].get("pred", "")}</div><div class="sub">最常误判对（{confused[0].get("count", 0)} 次），建议针对性增补样本。</div></div>\n'

    html += """  </div>

  <footer>由 scripts/model_evaluation/generate_benchmark_report.py 自动生成 · 数据来源：benchmark_results.json</footer>
</div>

<script>
const ctx1 = document.getElementById('perClassChart');
if (ctx1) {
  new Chart(ctx1, {
    type: 'bar',
    data: {
      labels: """ + json.dumps(bar_labels, ensure_ascii=False) + """,
      datasets: [{
        label: '准确率 (%)',
        data: """ + json.dumps(bar_values) + """,
        backgroundColor: """ + json.dumps(['#dc2626' if v < 30 else ('#d97706' if v < 60 else '#16a34a') for v in bar_values]) + """,
        borderRadius: 3,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(ctx) {
              const i = ctx.dataIndex;
              return '准确率 ' + ctx.parsed.x + '%  (' + """ + json.dumps(bar_corrects) + """[i] + '/' + """ + json.dumps(bar_totals) + """[i] + ')';
            }
          }
        }
      },
      scales: {
        x: { beginAtZero: true, max: 100, title: { display: true, text: '准确率 (%)' } },
        y: { ticks: { font: { size: 10 } } }
      }
    }
  });
}

const ctx2 = document.getElementById('speedChart');
if (ctx2) {
  new Chart(ctx2, {
    type: 'bar',
    data: {
      labels: ['单张延迟 (ms)', 'Batch(""" + str(eff.get("batch_size", 32)) + """) 延迟 (ms)', '单张吞吐 (FPS)', 'Batch 吞吐 (FPS)'],
      datasets: [{
        data: [""" + str(eff.get("single_latency_ms", 0)) + """, """ + str(eff.get("batch_latency_ms", 0)) + """, """ + str(eff.get("single_fps", 0)) + """, """ + str(eff.get("batch_fps", 0)) + """],
        backgroundColor: ['#4f7cff', '#4f7cff', '#16a34a', '#16a34a'],
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } }
    }
  });
}
</script>
</body>
</html>
"""
    return html


def main():
    if not RESULTS_PATH.exists():
        print(f"❌ 找不到结果文件: {RESULTS_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    html = generate_html(data)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ 报告已生成: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
