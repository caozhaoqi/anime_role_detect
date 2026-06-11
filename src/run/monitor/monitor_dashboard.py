#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务监控仪表板 - 统一查看所有服务状态，包含链路追踪功能
已拆分前后端代码
"""

import os
import sys

# 添加 src/ 到Python路径
_current_dir = os.path.dirname(os.path.abspath(__file__))            # .../src/run/monitor/
_src_dir = os.path.dirname(os.path.dirname(_current_dir))            # .../src/
sys.path.insert(0, _src_dir)

# 导入拆分后的模块（同在 monitor/ 目录下，Python 自动添加该目录到 sys.path）
from dashboard_backend import (
    get_all_services_status,
    get_tracing_stats,
    get_recent_traces,
    get_trace_details,
    get_topology_data,
)
from dashboard_templates import (
    generate_service_monitor_html,
    generate_tracing_html,
    generate_topology_html,
)
from cleaning_progress import generate_cleaning_progress_html

from flask import Flask, jsonify

# 监控配置
MONITOR_PORT = 9000

app = Flask(__name__)


def generate_html_dashboard() -> str:
    """生成完整的HTML仪表板"""
    services_status = get_all_services_status()
    tracing_stats = get_tracing_stats()
    recent_traces = get_recent_traces(20)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>动漫角色识别系统 - 服务监控仪表板</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #1a1a2e;
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header p {{
            color: #888;
        }}
        
        .tabs {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 30px;
        }}
        
        .tab {{
            padding: 10px 30px;
            background: #16213e;
            border: 1px solid #333;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.1em;
            transition: all 0.2s;
        }}
        
        .tab:hover {{
            background: #1a4d8c;
        }}
        
        .tab.active {{
            background: #667eea;
            border-color: #667eea;
        }}
        
        .tab-content {{
            /* display controlled by JS on parent */-tab-content divs */
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid #333;
        }}
        
        .stat-card h3 {{
            color: #888;
            font-size: 1em;
            margin-bottom: 10px;
        }}
        
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #fff;
        }}
        
        .stat-card.success .value {{ color: #4CAF50; }}
        .stat-card.error .value {{ color: #f44336; }}
        .stat-card.warning .value {{ color: #ff9800; }}
        
        .services-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .service-card {{
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #333;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .service-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        
        .service-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .service-name {{
            font-size: 1.2em;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .status-indicator {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }}
        
        .status-healthy {{ background: #4CAF50; box-shadow: 0 0 10px #4CAF50; }}
        .status-unreachable {{ background: #f44336; box-shadow: 0 0 10px #f44336; }}
        .status-timeout {{ background: #ff9800; box-shadow: 0 0 10px #ff9800; }}
        .status-disabled {{ background: #9E9E9E; }}
        .status-unknown {{ background: #9E9E9E; }}
        
        .service-badge {{
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8em;
        }}
        
        .badge-core {{ background: #667eea; color: white; }}
        .badge-aux {{ background: #444; color: #ccc; }}
        
        .service-info {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        
        .info-item {{
            background: rgba(255, 255, 255, 0.05);
            padding: 8px;
            border-radius: 5px;
        }}
        
        .info-label {{
            font-size: 0.8em;
            color: #888;
            margin-bottom: 4px;
        }}
        
        .info-value {{
            font-size: 0.9em;
        }}
        
        .api-links {{
            margin-top: 15px;
            display: flex;
            gap: 10px;
        }}
        
        .api-link {{
            padding: 6px 15px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 0.9em;
            transition: background 0.2s;
        }}
        
        .api-link:hover {{
            background: #5a6fd6;
        }}
        
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .panel {{
            background: #16213e;
            border-radius: 10px;
            border: 1px solid #333;
        }}
        
        .panel-header {{
            padding: 15px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .panel-title {{
            font-size: 1.1em;
            font-weight: bold;
        }}
        
        .panel-content {{
            padding: 20px;
        }}
        
        .detail-panel {{
            min-height: 400px;
        }}
        
        .trace-list {{
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
        }}
        
        .trace-item {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: background 0.2s;
            border-left: 4px solid transparent;
        }}
        
        .trace-item:hover {{
            background: rgba(255, 255, 255, 0.1);
        }}
        
        .trace-item.success {{ border-left-color: #4CAF50; }}
        .trace-item.error {{ border-left-color: #f44336; }}
        .trace-item.unset {{ border-left-color: #9E9E9E; }}
        
        .trace-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .trace-id {{
            font-size: 1.1em;
            font-weight: bold;
        }}
        
        .trace-duration {{
            color: #667eea;
            font-weight: bold;
        }}
        
        .trace-time {{
            color: #888;
            font-size: 0.9em;
            margin-bottom: 4px;
        }}
        
        .trace-spans {{
            color: #666;
            font-size: 0.85em;
        }}
        
        /* ========= 拓扑图样式 ========= */
        .topology-container {{
            background: #16213e;
            border-radius: 10px;
            border: 1px solid #333;
            padding: 20px;
            min-height: 500px;
        }}
        
        .topology-header h2 {{
            color: #667eea;
            margin-bottom: 15px;
        }}
        
        .topology-canvas {{
            position: relative;
            width: 100%;
            min-height: 560px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .topology-svg {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
        }}
        
        .topology-nodes {{
            position: relative;
            width: 100%;
            min-height: 560px;
            z-index: 2;
        }}
        
        .topo-node {{
            position: absolute;
            background: #1a1a2e;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 10px 16px;
            min-width: 120px;
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }}
        
        .topo-node:hover {{
            transform: scale(1.08);
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
        }}
        
        .topo-node-name {{
            font-size: 1em;
            font-weight: bold;
            color: #fff;
            margin-bottom: 2px;
        }}
        
        .topo-node-port {{
            font-size: 0.8em;
            color: #888;
            margin-bottom: 2px;
        }}
        
        .topo-node-status {{
            font-size: 0.75em;
            font-weight: bold;
        }}
        /* ========= 拓扑图样式结束 ========= */
        
        .refresh-btn {{
            padding: 8px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background 0.2s;
        }}
        
        .refresh-btn:hover {{
            background: #5a6fd6;
        }}
        
        .empty-state {{
            text-align: center;
            color: #666;
            padding: 40px;
            font-size: 1.1em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 动漫角色识别监控面板</h1>
            <p>实时监控服务状态 · API链路追踪 · 微服务拓扑</p>
            <div style="margin-top: 10px;">
                <button class="refresh-btn" onclick="location.reload()">🔄 刷新仪表板</button>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('services')">📊 服务监控</div>
            <div class="tab" onclick="switchTab('cleaning')">🧹 数据清理</div>
            <div class="tab" onclick="switchTab('tracing')">🔗 链路追踪</div>
            <div class="tab" onclick="switchTab('topology')">📊 拓扑图</div>
        </div>
        
        <div id="services-tab-content">
            {generate_service_monitor_html(services_status)}
        </div>
        
        <div id="cleaning-tab-content">
            {generate_cleaning_progress_html()}
        </div>
        
        <div id="tracing-tab-content">
            {generate_tracing_html(tracing_stats, recent_traces)}
        </div>
        
        <div id="topology-tab-content">
            {generate_topology_html(services_status, get_topology_data())}
        </div>
    </div>
    
    <script>
        function switchTab(tabName) {{
            // 隐藏所有tab
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('[id$="-tab-content"]').forEach(c => c.style.display = 'none');
            
            // 显示选中tab
            document.querySelector(`.tab[onclick*="'${{tabName}}'"]`).classList.add('active');
            document.getElementById(`${{tabName}}-tab-content`).style.display = 'block';
            
            // 如果是拓扑图tab，渲染拓扑图
            if (tabName === 'topology') {{
                setTimeout(renderTopology, 100);
            }}
        }}
        
        // 默认显示服务监控
        document.addEventListener('DOMContentLoaded', function() {{
            document.querySelectorAll('[id$="-tab-content"]').forEach(c => c.style.display = 'none');
            document.getElementById('services-tab-content').style.display = 'block';
        }});
        
        function refreshTracing() {{
            location.reload();
        }}
        
        // 追踪详情相关函数
        function loadTraceDetails(traceId) {{
            fetch('/api/trace/' + traceId)
                .then(response => response.json())
                .then(data => {{
                    if (data.error) {{
                        document.getElementById('trace-detail').innerHTML = 
                            '<div class="empty-state">加载失败: ' + data.error + '</div>';
                        return;
                    }}
                    renderTraceDetail(data);
                }});
        }}
        
        function renderTraceDetail(trace) {{
            const detailDiv = document.getElementById('trace-detail');
            detailDiv.innerHTML = '<h3 style="margin-bottom: 15px; color: #667eea;">' + 
                trace.trace_id + '</h3>' +
                '<div class="trace-tree">' + trace.tree_html + '</div>';
            
            // 渲染甘特图
            renderGanttChart(trace);
        }}
        
        function renderGanttChart(trace) {{
            const chartDiv = document.getElementById('gantt-chart');
            chartDiv.innerHTML = '<h3 style="margin-bottom: 15px; color: #667eea;">时序图</h3>' +
                '<div class="gantt-container">' + (trace.gantt_html || '暂无数据') + '</div>';
        }}
        
        function toggleSpanDetails(element) {{
            const details = element.querySelectorAll('.span-attributes, .span-events');
            details.forEach(d => {{
                d.style.display = d.style.display === 'none' ? 'block' : 'none';
            }});
        }}
        
        // 拓扑图渲染
        function renderTopology() {{
            const container = document.getElementById('topology-canvas');
            if (!container) return;
            
            const svg = document.getElementById('topology-svg');
            const nodes = container.querySelectorAll('.topo-node');
            
            if (nodes.length === 0) return;
            
            const svgWidth = container.offsetWidth;
            const svgHeight = container.offsetHeight;
            
            svg.setAttribute('width', svgWidth);
            svg.setAttribute('height', svgHeight);
            
            // 定位节点 — 7层架构布局
            // Layer 1: 前端层
            // Layer 2: 网关层
            // Layer 3: 核心业务层
            // Layer 4: 模型服务层 + 搜索工作进程
            // Layer 5: 推理工作进程
            // Layer 6: 监控观测层（右侧独立）
            const positions = {{
                'frontend': {{ x: svgWidth/2, y: 40 }},
                'api_gateway': {{ x: svgWidth/2, y: 120 }},
                'api_service': {{ x: svgWidth/3 - 30, y: 220 }},
                'multimedia_service': {{ x: svgWidth/2, y: 220 }},
                'search_service': {{ x: svgWidth*2/3 + 30, y: 220 }},
                'model_service': {{ x: svgWidth/2, y: 340 }},
                'search_worker': {{ x: svgWidth*2/3 + 30, y: 340 }},
                'inference_worker': {{ x: svgWidth/2, y: 440 }},
                'monitor_dashboard': {{ x: svgWidth - 100, y: svgHeight/2 }},
            }};
            
            nodes.forEach(node => {{
                const id = node.dataset.id;
                const pos = positions[id];
                if (pos) {{
                    const width = node.offsetWidth;
                    const height = node.offsetHeight;
                    node.style.position = 'absolute';
                    node.style.left = (pos.x - width/2) + 'px';
                    node.style.top = (pos.y - height/2) + 'px';
                }}
            }});
            
            // 绘制连线
            let svgContent = '';
            edgesData.forEach(edge => {{
                const sourcePos = positions[edge.source];
                const targetPos = positions[edge.target];
                if (sourcePos && targetPos) {{
                    svgContent += `<line x1="${{sourcePos.x}}" y1="${{sourcePos.y}}" 
                        x2="${{targetPos.x}}" y2="${{targetPos.y}}"
                        stroke="#667eea" stroke-width="2" opacity="0.5"/>`;
                    
                    // 添加箭头
                    const angle = Math.atan2(targetPos.y - sourcePos.y, targetPos.x - sourcePos.x);
                    const arrowLen = 10;
                    const endX = targetPos.x - 60 * Math.cos(angle);
                    const endY = targetPos.y - 60 * Math.sin(angle);
                    
                    svgContent += `<polygon points="
                        ${{endX}}, ${{endY}}
                        ${{endX - arrowLen * Math.cos(angle - 0.5)}}, ${{endY - arrowLen * Math.sin(angle - 0.5)}}
                        ${{endX - arrowLen * Math.cos(angle + 0.5)}}, ${{endY - arrowLen * Math.sin(angle + 0.5)}}
                    " fill="#667eea" opacity="0.5"/>`;
                }}
            }});
            
            svg.innerHTML = svgContent;
        }}
    </script>
</body>
</html>"""
    return html


@app.route("/")
def dashboard():
    """返回监控仪表板"""
    return generate_html_dashboard()


@app.route("/api/health")
def health():
    """健康检查接口"""
    return jsonify({"status": "healthy", "service": "monitor_dashboard"})


@app.route("/api/services/status")
def services_status():
    """获取服务状态"""
    return jsonify(get_all_services_status())


@app.route("/api/tracing/stats")
def tracing_stats():
    """获取追踪统计"""
    return jsonify(get_tracing_stats())


@app.route("/api/tracing/recent")
def recent_traces():
    """获取最近的追踪记录"""
    limit = int(20)
    traces = get_recent_traces(limit)
    return jsonify(traces)


@app.route("/api/trace/<trace_id>")
def trace_detail(trace_id):
    """获取追踪详情"""
    trace = get_trace_details(trace_id)
    if trace:
        # 生成HTML
        from dashboard_templates import generate_trace_tree_html, generate_gantt_html
        trace["tree_html"] = generate_trace_tree_html(trace)
        trace["gantt_html"] = generate_gantt_html(trace)
        return jsonify(trace)
    return jsonify({"error": "Trace not found"})


@app.route("/api/topology")
def topology():
    """获取拓扑图数据"""
    return jsonify(get_topology_data())


@app.route("/api/cleaning/progress")
def cleaning_progress():
    """获取数据清理进度"""
    from cleaning_progress import get_cleaning_progress
    return jsonify(get_cleaning_progress())


@app.route("/api/cleaning/reset")
def cleaning_reset():
    """重置数据清理进度"""
    from cleaning_progress import CleaningProgressTracker
    tracker = CleaningProgressTracker()
    tracker.reset_progress()
    return jsonify({"status": "success", "message": "进度已重置"})


if __name__ == "__main__":
    print(f"🚀 监控仪表板启动在 http://localhost:{MONITOR_PORT}")
    print(f"   📊 服务监控: http://localhost:{MONITOR_PORT}/")
    print(f"   🔗 API: http://localhost:{MONITOR_PORT}/api/health")
    app.run(
        host="0.0.0.0",
        port=MONITOR_PORT,
        debug=False,
    )