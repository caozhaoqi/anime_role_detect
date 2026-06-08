#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务监控仪表板 - 统一查看所有服务状态，包含链路追踪功能
已拆分前后端代码
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入拆分后的模块
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
    generate_trace_items_html,
    generate_tracing_stats_html,
    generate_topology_html,
)

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
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
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
        
        .trace-item.OK {{ border-left-color: #4CAF50; }}
        .trace-item.ERROR {{ border-left-color: #f44336; }}
        
        .trace-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }}
        
        .trace-id {{
            font-family: monospace;
            font-size: 0.9em;
            color: #667eea;
        }}
        
        .trace-duration {{
            font-size: 0.9em;
            color: #ff9800;
        }}
        
        .trace-time {{
            font-size: 0.8em;
            color: #888;
            margin-bottom: 5px;
        }}
        
        .trace-spans {{
            font-size: 0.8em;
            color: #666;
        }}
        
        .empty-state {{
            text-align: center;
            color: #666;
            padding: 40px;
        }}
        
        .endpoint-bar {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
            margin-bottom: 5px;
        }}
        
        .endpoint-name {{
            font-size: 0.9em;
            width: 200px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        .endpoint-bar-fill {{
            flex: 1;
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .endpoint-bar-inner {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 10px;
            transition: width 0.3s;
        }}
        
        .span-tree {{
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .span-item {{
            margin-bottom: 5px;
        }}
        
        .span-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
            cursor: pointer;
        }}
        
        .span-header:hover {{
            background: rgba(255, 255, 255, 0.1);
        }}
        
        .span-kind {{
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.7em;
            color: white;
            font-weight: bold;
        }}
        
        .span-name {{
            flex: 1;
            font-size: 0.9em;
        }}
        
        .span-duration {{
            font-size: 0.8em;
            color: #ff9800;
        }}
        
        .span-status {{
            font-size: 0.8em;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        
        .span-attributes, .span-events {{
            padding: 10px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 5px;
            margin-top: 5px;
            font-size: 0.8em;
        }}
        
        .attr-title {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #888;
        }}
        
        .status-ok {{ color: #4CAF50; }}
        .status-error {{ color: #f44336; }}
        .status-unset {{ color: #9E9E9E; }}
        
        .trace-summary {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        
        .summary-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .summary-row:last-child {{
            border-bottom: none;
        }}
        
        .summary-label {{
            color: #888;
        }}
        
        .summary-value {{
            font-weight: bold;
        }}
        
        .trace-id-full {{
            font-family: monospace;
            font-size: 0.8em;
            color: #667eea;
            word-break: break-all;
        }}
        
        .trace-spans-header {{
            margin-bottom: 10px;
        }}
        
        .refresh-info {{
            text-align: center;
            color: #888;
            margin-bottom: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
        }}
        
        .refresh-btn {{
            padding: 8px 20px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: bold;
            transition: all 0.2s;
        }}
        
        .refresh-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        
        .topology-container {{
            position: relative;
            height: 600px;
            background: #16213e;
            border-radius: 10px;
            border: 1px solid #333;
            overflow: hidden;
        }}
        
        .topology-svg {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        }}
        
        .edge-line {{
            transition: stroke 0.3s;
        }}
        
        .edge-line:hover {{
            stroke: #764ba2;
            stroke-width: 3;
        }}
        
        .topology-nodes {{
            position: relative;
            z-index: 2;
            width: 100%;
            height: 100%;
        }}
        
        .topology-node {{
            position: absolute;
            width: 240px;
            background: linear-gradient(135deg, #1e3a5f, #16213e);
            border-radius: 10px;
            padding: 15px;
            border: 2px solid #333;
            transition: all 0.3s;
            cursor: pointer;
        }}
        
        .topology-node:hover {{
            transform: scale(1.05);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }}
        
        .topology-node.border-core {{
            border-color: #667eea;
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.3);
        }}
        
        .node-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}
        
        .node-status {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
        
        .node-name {{
            font-size: 1.1em;
            font-weight: bold;
            color: #fff;
        }}
        
        .node-info {{
            font-size: 0.9em;
            color: #888;
        }}
        
        .status-text {{
            font-weight: bold;
        }}
        
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            padding: 15px;
            background: #16213e;
            border-radius: 10px;
            border: 1px solid #333;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: #888;
            font-size: 0.9em;
        }}
        
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        
        .legend-dot.healthy {{ background: #4CAF50; }}
        .legend-dot.unreachable {{ background: #f44336; }}
        .legend-dot.timeout {{ background: #ff9800; }}
        .legend-dot.disabled {{ background: #9E9E9E; }}
        
        .legend-core {{
            width: 20px;
            height: 20px;
            border: 2px solid #667eea;
            border-radius: 5px;
            background: rgba(102, 126, 234, 0.2);
        }}
        
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 动漫角色识别系统</h1>
            <p>服务监控仪表板</p>
        </div>
        
        <div class="refresh-info">
            <button class="refresh-btn" onclick="refreshDashboard()">🔄 刷新仪表板</button>
            <span id="last-update">最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('services')">📊 服务监控</div>
            <div class="tab" onclick="switchTab('tracing')">🔗 链路追踪</div>
            <div class="tab" onclick="switchTab('topology')">🌐 服务拓扑</div>
        </div>
        
        {generate_service_monitor_html(services_status)}
        
        {generate_tracing_html(tracing_stats, recent_traces)}
        
        {generate_topology_html(get_topology_data())}
        
        <div class="footer">
            <p>监控系统 v1.0 | 数据仅供参考</p>
        </div>
    </div>
    
    <script src="/static/dashboard.js?v={datetime.now().strftime('%Y%m%d%H%M%S')}"></script>
</body>
</html>"""
    return html


@app.route("/")
def index():
    """主页 - 返回监控仪表板"""
    return generate_html_dashboard()


@app.route("/api/reload")
def reload_dashboard():
    """重新加载仪表板数据"""
    return generate_html_dashboard()


@app.route("/api/services")
def get_services_api():
    """获取服务状态API"""
    services = get_all_services_status()
    return jsonify({"success": True, "data": services})


@app.route("/api/tracing/stats")
def get_tracing_stats_api():
    """获取追踪统计API"""
    stats = get_tracing_stats()
    return jsonify({"success": True, "data": stats})


@app.route("/api/tracing/traces")
def get_traces_api():
    """获取追踪列表API"""
    traces = get_recent_traces(20)
    return jsonify({"success": True, "data": traces})


@app.route("/api/tracing/trace/<trace_id>")
def get_trace_api(trace_id):
    """获取单个追踪详情API"""
    trace = get_trace_details(trace_id)
    if trace:
        return jsonify({"success": True, "data": trace})
    return jsonify({"success": False, "message": "Trace not found"}), 404


@app.route("/api/tracing/reload")
def reload_tracing():
    """刷新追踪数据API"""
    stats = get_tracing_stats()
    recent_traces = get_recent_traces(20)
    
    stats_html = generate_tracing_stats_html(stats)
    traces_html = generate_trace_items_html(recent_traces)
    if not recent_traces:
        traces_html = "<div class='empty-state'>暂无追踪记录</div>"
    
    return jsonify({
        "success": True,
        "stats_html": stats_html,
        "traces_html": traces_html
    })


@app.route("/api/topology/reload")
def reload_topology():
    """刷新拓扑图数据API"""
    topology_data = get_topology_data()
    html = generate_topology_html(topology_data)
    
    return jsonify({
        "success": True,
        "html": html
    })


@app.route("/static/dashboard.js")
def serve_js():
    """提供JavaScript文件"""
    js_path = os.path.join(project_root, "static", "dashboard.js")
    if os.path.exists(js_path):
        with open(js_path, "r") as f:
            return f.read(), {"Content-Type": "application/javascript"}
    return "// Dashboard JS not found", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=MONITOR_PORT, debug=False)