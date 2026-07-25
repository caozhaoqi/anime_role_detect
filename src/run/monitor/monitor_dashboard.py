#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务监控仪表板 - 统一查看所有服务状态，包含链路追踪功能
已拆分前后端代码
"""

import os

from src.run.monitor.dashboard_backend import (
    get_all_services_status,
    get_tracing_stats,
    get_recent_traces,
    get_trace_details,
    get_topology_data,
    get_trace_span_tree,
    get_trace_root_span,
    get_span_details,
    get_child_spans,
    search_traces_api,
)
from src.run.monitor.dashboard_templates import (
    generate_service_monitor_html,
    generate_tracing_html,
    generate_topology_html,
    generate_trace_tree_html,
    generate_gantt_html,
)
from src.run.monitor.cleaning_progress import (
    generate_cleaning_progress_html,
    get_cleaning_progress,
    CleaningProgressTracker,
)

from flask import Flask, jsonify, request

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
    <title>动漫角色识别系统 - 监控中心</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        :root {{
            --primary-color: #667eea;
            --primary-dark: #5a6fd6;
            --secondary-color: #764ba2;
            --success-color: #10b981;
            --error-color: #ef4444;
            --warning-color: #f59e0b;
            --info-color: #3b82f6;
            --bg-dark: #0f0f1a;
            --bg-card: #1a1a2e;
            --bg-card-hover: #252542;
            --border-color: #2a2a4a;
            --text-primary: #ffffff;
            --text-secondary: #a0a0c0;
            --text-muted: #6c6c8c;
        }}
        
        /* Light theme overrides */
        [data-theme="light"] {{
            --primary-color: #5b6cdb;
            --primary-dark: #4a5bc8;
            --secondary-color: #7c3aed;
            --success-color: #059669;
            --error-color: #dc2626;
            --warning-color: #d97706;
            --info-color: #2563eb;
            --bg-dark: #f0f2f8;
            --bg-card: #ffffff;
            --bg-card-hover: #f5f7fa;
            --border-color: #e2e5ee;
            --text-primary: #1a1a2e;
            --text-secondary: #555570;
            --text-muted: #9999b0;
        }}

        [data-theme="light"] body {{
            background: linear-gradient(135deg, #f0f2f8 0%, #e8eaf6 50%, #f5f7fa 100%);
        }}

        [data-theme="light"] body::before {{
            background: 
                radial-gradient(circle at 20% 80%, rgba(91, 108, 219, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(124, 58, 237, 0.08) 0%, transparent 50%);
        }}

        [data-theme="light"] .panel-header {{
            background: rgba(0, 0, 0, 0.02);
        }}

        [data-theme="light"] .info-item,
        [data-theme="light"] .trace-item,
        [data-theme="light"] .endpoint-bar,
        [data-theme="light"] .cleaning-task,
        [data-theme="light"] .span-tree-node {{
            background: rgba(0, 0, 0, 0.02);
            border-color: rgba(0, 0, 0, 0.06);
        }}

        [data-theme="light"] .search-bar {{
            background: rgba(0, 0, 0, 0.02);
        }}

        [data-theme="light"] .search-bar input,
        [data-theme="light"] .search-bar select {{
            background: rgba(255, 255, 255, 0.8);
        }}

        [data-theme="light"] .topology-canvas {{
            background: rgba(0, 0, 0, 0.03);
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1a2e 50%, #16213e 100%);
            min-height: 100vh;
            color: var(--text-primary);
            padding: 24px;
            overflow-x: hidden;
            transition: background 0.4s ease, color 0.4s ease;
        }}
        
        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 80%, rgba(102, 126, 234, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(118, 75, 162, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(16, 185, 129, 0.05) 0%, transparent 70%);
            pointer-events: none;
            z-index: -1;
            transition: opacity 0.4s ease;
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            animation: fadeInDown 0.6s ease-out;
        }}
        
        @keyframes fadeInDown {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .header h1 {{
            font-size: 3em;
            margin-bottom: 12px;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        
        .header p {{
            color: var(--text-secondary);
            font-size: 1.1em;
            letter-spacing: 1px;
        }}
        
        .header-actions {{
            margin-top: 20px;
            display: flex;
            justify-content: center;
            gap: 12px;
        }}
        
        .tabs {{
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-bottom: 32px;
            background: rgba(255, 255, 255, 0.03);
            padding: 8px;
            border-radius: 16px;
            backdrop-filter: blur(10px);
            border: 1px solid var(--border-color);
            animation: fadeInUp 0.6s ease-out 0.2s both;
        }}
        
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .tab {{
            padding: 12px 28px;
            background: transparent;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            color: var(--text-secondary);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            gap: 8px;
            position: relative;
            overflow: hidden;
        }}
        
        .tab::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            opacity: 0;
            transition: opacity 0.3s;
        }}
        
        .tab:hover {{
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
            transform: translateY(-1px);
        }}
        
        .tab.active {{
            color: white;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }}
        
        .tab.active::before {{
            opacity: 1;
        }}
        
        .tab span {{
            position: relative;
            z-index: 1;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }}
        
        .stat-card {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}
        
        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            opacity: 0;
            transition: opacity 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
            border-color: rgba(102, 126, 234, 0.3);
        }}
        
        .stat-card:hover::before {{
            opacity: 1;
        }}
        
        .stat-card h3 {{
            color: var(--text-secondary);
            font-size: 0.95em;
            font-weight: 500;
            margin-bottom: 12px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        
        .stat-card .value {{
            font-size: 2.8em;
            font-weight: 700;
            background: linear-gradient(135deg, var(--text-primary) 0%, var(--text-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
        }}
        
        .stat-card.success .value {{ 
            background: linear-gradient(135deg, var(--success-color) 0%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .stat-card.error .value {{ 
            background: linear-gradient(135deg, var(--error-color) 0%, #f87171 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .stat-card.warning .value {{ 
            background: linear-gradient(135deg, var(--warning-color) 0%, #fbbf24 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .stat-card.info .value {{ 
            background: linear-gradient(135deg, var(--info-color) 0%, #60a5fa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .services-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }}
        
        .service-card {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}
        
        .service-card::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--border-color);
            transition: background 0.3s;
        }}
        
        .service-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
            border-color: rgba(102, 126, 234, 0.4);
        }}
        
        .service-card.status-healthy::after {{ background: var(--success-color); }}
        .service-card.status-unreachable::after {{ background: var(--error-color); }}
        .service-card.status-timeout::after {{ background: var(--warning-color); }}
        
        .service-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .service-name-row {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .service-icon {{
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3em;
            background: rgba(102, 126, 234, 0.15);
        }}
        
        .service-name {{
            font-size: 1.25em;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .status-indicator {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            display: inline-block;
            position: relative;
        }}
        
        .status-indicator::after {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 100%;
            height: 100%;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ transform: translate(-50%, -50%) scale(1); opacity: 0.8; }}
            100% {{ transform: translate(-50%, -50%) scale(2.5); opacity: 0; }}
        }}
        
        .status-healthy {{ 
            background: var(--success-color); 
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
        }}
        .status-healthy::after {{ background: var(--success-color); }}
        
        .status-unreachable {{ 
            background: var(--error-color); 
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
        }}
        .status-unreachable::after {{ background: var(--error-color); }}
        
        .status-timeout {{ 
            background: var(--warning-color); 
            box-shadow: 0 0 15px rgba(245, 158, 11, 0.4);
        }}
        .status-timeout::after {{ background: var(--warning-color); }}
        
        .status-disabled, .status-unknown {{ background: var(--text-muted); }}
        
        .service-badge {{
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .badge-core {{ 
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white; 
        }}
        .badge-aux {{ background: rgba(255, 255, 255, 0.08); color: var(--text-secondary); }}
        
        .service-info {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}
        
        .info-item {{
            background: rgba(255, 255, 255, 0.03);
            padding: 12px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .info-label {{
            font-size: 0.75em;
            color: var(--text-muted);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .info-value {{
            font-size: 1em;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .api-links {{
            margin-top: 18px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .api-link {{
            padding: 8px 18px;
            background: rgba(102, 126, 234, 0.15);
            color: var(--primary-color);
            text-decoration: none;
            border-radius: 8px;
            font-size: 0.85em;
            font-weight: 500;
            transition: all 0.3s;
            border: 1px solid rgba(102, 126, 234, 0.2);
        }}
        
        .api-link:hover {{
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
            transform: translateY(-1px);
        }}
        
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        
        .panel {{
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            transition: all 0.3s;
        }}
        
        .panel:hover {{
            border-color: rgba(102, 126, 234, 0.3);
        }}
        
        .panel-header {{
            padding: 20px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(0, 0, 0, 0.1);
        }}
        
        .panel-title {{
            font-size: 1.15em;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .panel-actions {{
            display: flex;
            gap: 8px;
        }}
        
        .panel-content {{
            padding: 24px;
        }}
        
        .detail-panel {{
            min-height: 420px;
        }}
        
        .search-bar {{
            display: flex;
            gap: 12px;
            align-items: center;
            margin-bottom: 20px;
            padding: 16px 20px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        
        .search-bar input, .search-bar select {{
            padding: 10px 16px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 0.9em;
            outline: none;
            transition: border-color 0.3s;
        }}
        
        .search-bar input:focus, .search-bar select:focus {{
            border-color: var(--primary-color);
        }}
        
        .search-bar input::placeholder {{
            color: var(--text-muted);
        }}
        
        .trace-list {{
            max-height: 420px;
            overflow-y: auto;
            padding: 8px;
        }}
        
        .trace-list::-webkit-scrollbar {{
            width: 6px;
        }}
        
        .trace-list::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
        }}
        
        .trace-list::-webkit-scrollbar-thumb {{
            background: var(--border-color);
            border-radius: 3px;
        }}
        
        .trace-list::-webkit-scrollbar-thumb:hover {{
            background: var(--text-muted);
        }}
        
        .trace-item {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 12px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border-left: 4px solid transparent;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .trace-item:hover {{
            background: var(--bg-card-hover);
            transform: translateX(4px);
            border-color: rgba(102, 126, 234, 0.3);
        }}
        
        .trace-item.success {{ border-left-color: var(--success-color); }}
        .trace-item.error {{ border-left-color: var(--error-color); }}
        .trace-item.unset {{ border-left-color: var(--text-muted); }}
        
        .trace-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .trace-id {{
            font-size: 1em;
            font-weight: 600;
            color: var(--text-primary);
            font-family: 'Monaco', 'Consolas', monospace;
            letter-spacing: 0.5px;
        }}
        
        .trace-duration {{
            font-size: 1.1em;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .trace-time {{
            color: var(--text-muted);
            font-size: 0.85em;
            margin-bottom: 6px;
        }}
        
        .trace-spans {{
            color: var(--text-secondary);
            font-size: 0.85em;
            display: flex;
            gap: 16px;
        }}
        
        .span-count {{
            padding: 2px 10px;
            background: rgba(102, 126, 234, 0.15);
            border-radius: 10px;
            color: var(--primary-color);
            font-weight: 500;
        }}
        
        .trace-status {{
            padding: 2px 10px;
            border-radius: 10px;
            font-weight: 500;
        }}
        
        .trace-status.ok {{ 
            background: rgba(16, 185, 129, 0.15);
            color: var(--success-color);
        }}
        .trace-status.error {{ 
            background: rgba(239, 68, 68, 0.15);
            color: var(--error-color);
        }}
        
        .topology-container {{
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border-color);
            padding: 24px;
            min-height: 520px;
        }}
        
        .topology-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .topology-header h2 {{
            color: var(--text-primary);
            font-size: 1.3em;
            font-weight: 600;
        }}
        
        .topology-canvas {{
            position: relative;
            width: 100%;
            min-height: 560px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
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
            background: var(--bg-card);
            border: 2px solid var(--border-color);
            border-radius: 14px;
            padding: 14px 20px;
            min-width: 140px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        
        .topo-node:hover {{
            transform: scale(1.1) translateY(-2px);
            box-shadow: 0 8px 30px rgba(102, 126, 234, 0.35);
            border-color: var(--primary-color);
        }}
        
        .topo-node.healthy {{ border-color: var(--success-color); }}
        .topo-node.unreachable {{ border-color: var(--error-color); }}
        .topo-node.timeout {{ border-color: var(--warning-color); }}
        
        .topo-node-name {{
            font-size: 1.05em;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 4px;
        }}
        
        .topo-node-port {{
            font-size: 0.75em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        
        .topo-node-status {{
            font-size: 0.7em;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 8px;
            display: inline-block;
        }}
        
        .topo-node-status.healthy {{ 
            background: rgba(16, 185, 129, 0.15);
            color: var(--success-color);
        }}
        .topo-node-status.unreachable {{ 
            background: rgba(239, 68, 68, 0.15);
            color: var(--error-color);
        }}
        .topo-node-status.timeout {{ 
            background: rgba(245, 158, 11, 0.15);
            color: var(--warning-color);
        }}
        
        .refresh-btn, .action-btn {{
            padding: 10px 22px;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 500;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .refresh-btn:hover, .action-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }}
        
        .action-btn.secondary {{
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }}

        .action-btn.secondary:hover {{
            background: rgba(255, 255, 255, 0.12);
            color: var(--text-primary);
        }}

        /* === Time range selector === */
        .range-btn {{
            padding: 6px 14px;
            background: transparent;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.85em;
            font-weight: 500;
            color: var(--text-secondary);
            transition: all 0.25s;
        }}

        .range-btn:hover {{
            background: rgba(255, 255, 255, 0.06);
            color: var(--text-primary);
        }}

        .range-btn.active {{
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        }}

        /* === Response time color coding === */
        .info-value.rt-fast {{ color: var(--success-color); }}
        .info-value.rt-normal {{ color: var(--info-color); }}
        .info-value.rt-slow {{ color: var(--warning-color); }}
        .info-value.rt-critical {{ color: var(--error-color); font-weight: 700; }}

        /* === Gantt tooltip === */
        .gantt-bar {{
            position: absolute;
            cursor: pointer;
        }}

        .gantt-bar:hover {{
            opacity: 1;
            transform: scaleY(1.15);
        }}

        .gantt-tooltip {{
            position: absolute;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 0.82em;
            color: var(--text-primary);
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.4);
            pointer-events: none;
            z-index: 100;
            max-width: 280px;
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.2s;
        }}

        .gantt-tooltip.visible {{
            opacity: 1;
        }}

        .gantt-tooltip-row {{
            padding: 2px 0;
        }}

        .gantt-tooltip-label {{
            color: var(--text-muted);
            margin-right: 6px;
        }}
        
        .empty-state {{
            text-align: center;
            color: var(--text-muted);
            padding: 60px 40px;
            font-size: 1em;
        }}
        
        .empty-state-icon {{
            font-size: 3em;
            margin-bottom: 16px;
            opacity: 0.5;
        }}
        
        .endpoint-bar {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .endpoint-name {{
            flex: 1;
            font-size: 0.9em;
            color: var(--text-primary);
        }}
        
        .endpoint-count {{
            font-size: 1.1em;
            font-weight: 700;
            color: var(--primary-color);
            margin-right: 12px;
        }}
        
        .endpoint-progress {{
            flex: 2;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
        }}
        
        .endpoint-progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            border-radius: 3px;
            transition: width 0.5s ease;
        }}
        
        .cleaning-task {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.3s;
        }}
        
        .cleaning-task:hover {{
            background: var(--bg-card-hover);
            border-color: rgba(102, 126, 234, 0.2);
        }}
        
        .cleaning-task-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        
        .cleaning-task-name {{
            font-size: 1em;
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .cleaning-task-status {{
            font-size: 0.8em;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 12px;
        }}
        
        .cleaning-task-status.pending {{ 
            background: rgba(108, 108, 140, 0.2);
            color: var(--text-muted);
        }}
        .cleaning-task-status.running {{ 
            background: rgba(59, 130, 246, 0.2);
            color: var(--info-color);
        }}
        .cleaning-task-status.completed {{ 
            background: rgba(16, 185, 129, 0.2);
            color: var(--success-color);
        }}
        .cleaning-task-status.failed {{ 
            background: rgba(239, 68, 68, 0.2);
            color: var(--error-color);
        }}
        
        .cleaning-progress-bar {{
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 8px;
        }}
        
        .cleaning-progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--success-color) 0%, #34d399 100%);
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        
        .cleaning-progress-text {{
            font-size: 0.85em;
            color: var(--text-secondary);
        }}
        
        .span-tree-node {{
            margin-left: 20px;
            margin-bottom: 10px;
            padding: 14px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.3s;
        }}
        
        .span-tree-node:hover {{
            background: var(--bg-card-hover);
            border-color: rgba(102, 126, 234, 0.2);
        }}
        
        .span-tree-root {{
            margin-left: 0;
            background: rgba(102, 126, 234, 0.1);
            border-color: rgba(102, 126, 234, 0.3);
        }}
        
        .span-tree-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .span-tree-name {{
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .span-tree-duration {{
            font-size: 0.9em;
            font-weight: 600;
        }}
        
        .span-tree-duration.success {{ color: var(--success-color); }}
        .span-tree-duration.error {{ color: var(--error-color); }}
        
        .span-tree-attributes {{
            font-size: 0.8em;
            color: var(--text-muted);
            max-height: 180px;
            overflow-y: auto;
            padding-right: 8px;
        }}
        
        .span-tree-attributes::-webkit-scrollbar {{
            width: 4px;
        }}
        
        .span-tree-attributes::-webkit-scrollbar-thumb {{
            background: var(--border-color);
            border-radius: 2px;
        }}
        
        .attribute-row {{
            padding: 4px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        .attribute-key {{
            font-weight: 500;
            color: var(--text-secondary);
            margin-right: 8px;
        }}
        
        .attribute-value {{
            color: var(--text-primary);
        }}
        
        .tab-content {{
            animation: fadeIn 0.4s ease-out;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @media (max-width: 768px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}
            
            .services-grid {{
                grid-template-columns: 1fr;
            }}
            
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .tabs {{
                flex-wrap: wrap;
            }}
            
            .tab {{
                padding: 10px 16px;
                font-size: 0.9em;
            }}
        }}

        /* === Summary grid (cleaning progress) === */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .summary-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .summary-card:hover {{
            transform: translateY(-3px);
            border-color: rgba(102, 126, 234, 0.3);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }}

        .summary-card.success {{ border-top: 3px solid var(--success-color); }}
        .summary-card.warning {{ border-top: 3px solid var(--warning-color); }}
        .summary-card.info {{ border-top: 3px solid var(--info-color); }}

        .summary-icon {{
            font-size: 1.8em;
            margin-bottom: 8px;
        }}

        .summary-value {{
            font-size: 1.8em;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 4px;
        }}

        .summary-label {{
            font-size: 0.8em;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* === Loading spinner === */
        .loading-spinner {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 60px 20px;
            gap: 16px;
        }}

        .spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid var(--border-color);
            border-top-color: var(--primary-color);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        .loading-text {{
            color: var(--text-muted);
            font-size: 0.9em;
        }}

        /* === Toast notification === */
        .toast-container {{
            position: fixed;
            top: 24px;
            right: 24px;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .toast {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--error-color);
            border-radius: 12px;
            padding: 14px 20px;
            min-width: 280px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
            display: flex;
            align-items: center;
            gap: 12px;
            animation: toastIn 0.3s ease-out;
        }}

        .toast.success {{ border-left-color: var(--success-color); }}
        .toast.info {{ border-left-color: var(--info-color); }}

        .toast-icon {{
            font-size: 1.3em;
        }}

        .toast-message {{
            flex: 1;
            font-size: 0.9em;
            color: var(--text-primary);
        }}

        .toast-close {{
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 1.1em;
            padding: 4px;
        }}

        @keyframes toastIn {{
            from {{ opacity: 0; transform: translateX(40px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}

        @keyframes toastOut {{
            to {{ opacity: 0; transform: translateX(40px); }}
        }}

        /* === Theme toggle + auto-refresh controls === */
        .header-controls {{
            margin-top: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }}

        .toggle-btn {{
            padding: 8px 18px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            cursor: pointer;
            font-size: 0.85em;
            font-weight: 500;
            color: var(--text-secondary);
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .toggle-btn:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            border-color: rgba(102, 126, 234, 0.3);
        }}

        .toggle-btn.active {{
            color: var(--primary-color);
            border-color: var(--primary-color);
            background: rgba(102, 126, 234, 0.1);
        }}

        .auto-refresh-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8em;
            color: var(--text-muted);
        }}

        .refresh-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success-color);
            animation: pulse 2s infinite;
        }}

        .refresh-dot.paused {{
            background: var(--text-muted);
            animation: none;
        }}

        /* === Service card refresh animation === */
        .service-card.refreshing {{
            opacity: 0.6;
            pointer-events: none;
            transition: opacity 0.3s;
        }}

        .service-card.refreshed {{
            animation: cardFlash 0.6s ease-out;
        }}

        @keyframes cardFlash {{
            0% {{ box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.4); }}
            100% {{ box-shadow: 0 0 0 12px rgba(102, 126, 234, 0); }}
        }}

        /* === Gantt chart dark theme fix === */
        .gantt-header {{
            border-bottom-color: var(--border-color) !important;
            color: var(--text-muted) !important;
        }}

        .gantt-row {{
            border-bottom-color: rgba(255, 255, 255, 0.05) !important;
        }}

        [data-theme="light"] .gantt-row {{
            border-bottom-color: rgba(0, 0, 0, 0.05) !important;
        }}

        .gantt-label {{
            color: var(--text-secondary) !important;
        }}

        .gantt-track {{
            background: rgba(255, 255, 255, 0.05) !important;
        }}

        [data-theme="light"] .gantt-track {{
            background: rgba(0, 0, 0, 0.05) !important;
        }}

        /* === Span tree dark theme === */
        .span-tree-node-dark {{
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 10px;
            transition: all 0.3s;
        }}

        [data-theme="light"] .span-tree-node-dark {{
            background: rgba(0, 0, 0, 0.02) !important;
            border: 1px solid rgba(0, 0, 0, 0.06) !important;
        }}

        .span-tree-node-dark:hover {{
            background: var(--bg-card-hover) !important;
            border-color: rgba(102, 126, 234, 0.2) !important;
        }}

        .span-tree-attr-dark {{
            font-size: 0.8em;
            color: var(--text-muted);
            margin-top: 6px;
            max-height: 150px;
            overflow-y: auto;
        }}

        .span-tree-attr-dark div {{
            padding: 3px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }}

        [data-theme="light"] .span-tree-attr-dark div {{
            border-bottom-color: rgba(0, 0, 0, 0.04);
        }}

        .span-tree-attr-dark strong {{
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 动漫角色识别监控中心</h1>
            <p>Real-time Monitoring · API Tracing · Microservice Topology</p>
            <div class="header-controls">
                <button class="refresh-btn" onclick="refreshServiceStatus()">🔄 刷新状态</button>
                <button class="toggle-btn" id="auto-refresh-btn" onclick="toggleAutoRefresh()">
                    <span class="refresh-dot" id="refresh-dot"></span>
                    <span id="auto-refresh-label">自动刷新 30s</span>
                </button>
                <button class="toggle-btn" id="theme-toggle" onclick="toggleTheme()">
                    <span id="theme-icon">🌙</span>
                    <span id="theme-label">深色</span>
                </button>
            </div>
        </div>

        <div class="toast-container" id="toast-container"></div>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab('services')"><span>📊 服务监控</span></button>
            <button class="tab" onclick="switchTab('cleaning')"><span>🧹 数据清理</span></button>
            <button class="tab" onclick="switchTab('tracing')"><span>🔗 链路追踪</span></button>
            <button class="tab" onclick="switchTab('topology')"><span>📊 拓扑图</span></button>
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
            currentTab = tabName;

            // 如果是拓扑图tab，渲染拓扑图
            if (tabName === 'topology') {{
                setTimeout(renderTopology, 100);
            }}
        }}

        // 当前激活的 tab，供自动刷新使用
        let currentTab = 'services';
        // 追踪统计的时间窗口（小时）
        let tracingTimeRange = 24;

        function refreshTracing() {{
            // AJAX 局部刷新，避免全页 reload
            refreshTracingStats(tracingTimeRange);
            loadRecentTraces();
        }}

        function changeTimeRange(hours) {{
            tracingTimeRange = hours;
            // 切换按钮高亮
            document.querySelectorAll('.range-btn').forEach(btn => {{
                btn.classList.toggle('active', parseInt(btn.dataset.hours) === hours);
            }});
            refreshTracingStats(hours);
        }}

        function refreshTracingStats(hours) {{
            fetch('/api/tracing/stats?hours=' + hours)
                .then(response => response.json())
                .then(stats => {{
                    const container = document.getElementById('tracing-stats');
                    if (!container) return;
                    container.innerHTML =
                        '<div class="stats-grid">' +
                            '<div class="stat-card"><h3>总追踪数</h3><div class="value">' + (stats.total_traces || 0) + '</div></div>' +
                            '<div class="stat-card success"><h3>成功请求</h3><div class="value">' + (stats.success_count || 0) + '</div></div>' +
                            '<div class="stat-card error"><h3>错误请求</h3><div class="value">' + (stats.error_count || 0) + '</div></div>' +
                            '<div class="stat-card warning"><h3>错误率</h3><div class="value">' + (stats.error_rate || 0) + '%</div></div>' +
                            '<div class="stat-card info"><h3>平均耗时</h3><div class="value">' + (stats.avg_duration_ms || 0) + 'ms</div></div>' +
                            '<div class="stat-card"><h3>最大耗时</h3><div class="value">' + (stats.max_duration_ms || 0) + 'ms</div></div>' +
                        '</div>';
                }})
                .catch(err => showToast('刷新统计失败: ' + err.message, 'error'));
        }}

        function refreshCleaningProgress() {{
            fetch('/api/cleaning/progress')
                .then(response => response.json())
                .then(progress => {{
                    const wrapper = document.getElementById('cleaning-tab-content');
                    if (!wrapper) return;
                    // 仅更新更新时间文本，避免整块重渲染破坏布局
                    const timeNode = wrapper.querySelector('.panel-header span[style*="text-muted"]');
                    if (timeNode) timeNode.textContent = '更新时间: ' + (progress.last_updated || '');
                    // 更新汇总数字
                    const summary = progress.summary || {{}};
                    const values = wrapper.querySelectorAll('.summary-card .summary-value');
                    if (values.length >= 6) {{
                        values[0].textContent = summary.total_processed || 0;
                        values[1].textContent = summary.total_valid || 0;
                        values[2].textContent = summary.total_rejected || 0;
                        values[3].textContent = summary.total_duplicates || 0;
                        values[4].textContent = (summary.avg_confidence || 0).toFixed(2);
                        values[5].textContent = (summary.avg_quality_score || 0).toFixed(2);
                    }}
                    // 更新任务进度条
                    const tasks = progress.tasks || {{}};
                    Object.keys(tasks).forEach(tid => {{
                        const task = tasks[tid];
                        const card = wrapper.querySelector('.cleaning-task[data-task-id="' + tid + '"]');
                        if (!card) return;
                        const fill = card.querySelector('.cleaning-progress-fill');
                        if (fill) fill.style.width = (task.progress || 0) + '%';
                        const text = card.querySelector('.cleaning-progress-text');
                        if (text) text.innerHTML = '<span>' + task.completed + '/' + task.total + '</span><span style="float: right;">' + (task.progress || 0).toFixed(1) + '%</span>';
                    }});
                    showToast('数据清理进度已刷新', 'success');
                }})
                .catch(err => showToast('刷新清理进度失败: ' + err.message, 'error'));
        }}

        function resetCleaningProgress() {{
            if (!confirm('确定要重置数据清理进度吗？此操作不可恢复。')) return;
            fetch('/api/cleaning/reset')
                .then(response => response.json())
                .then(data => {{
                    if (data.status === 'success') {{
                        showToast('进度已重置', 'success');
                        // 整块重新渲染清理面板
                        location.reload();
                    }} else {{
                        showToast('重置失败: ' + (data.message || '未知错误'), 'error');
                    }}
                }})
                .catch(err => showToast('重置失败: ' + err.message, 'error'));
        }}
        
        // 追踪详情相关函数
        let currentTraceId = null;

        function loadTraceDetails(traceId) {{
            currentTraceId = traceId;
            showLoading('trace-detail');
            fetch('/api/trace/' + traceId)
                .then(response => response.json())
                .then(data => {{
                    if (data.error) {{
                        document.getElementById('trace-detail').innerHTML = 
                            '<div class="empty-state"><div class="empty-state-icon">⚠️</div>加载失败: ' + data.error + '</div>';
                        showToast('加载追踪详情失败: ' + data.error, 'error');
                        return;
                    }}
                    renderTraceDetail(data);
                }})
                .catch(err => {{
                    document.getElementById('trace-detail').innerHTML = 
                        '<div class="empty-state"><div class="empty-state-icon">⚠️</div>网络错误，请稍后重试</div>';
                    showToast('网络错误: ' + err.message, 'error');
                }});
        }}
        
        function renderTraceDetail(trace) {{
            const detailDiv = document.getElementById('trace-detail');
            detailDiv.innerHTML = '<h3 style="margin-bottom: 15px; color: var(--primary-color);">' + 
                trace.trace_id + '</h3>' +
                '<div class="trace-tree">' + trace.tree_html + '</div>';
            
            // 渲染甘特图
            renderGanttChart(trace);
        }}
        
        function renderGanttChart(trace) {{
            const chartDiv = document.getElementById('gantt-chart');
            chartDiv.innerHTML = '<h3 style="margin-bottom: 15px; color: var(--primary-color);">时序图</h3>' +
                '<div class="gantt-container">' + (trace.gantt_html || '暂无数据') + '</div>';
        }}
        
        function toggleSpanDetails(element) {{
            const details = element.querySelectorAll('.span-attributes, .span-events');
            details.forEach(d => {{
                d.style.display = d.style.display === 'none' ? 'block' : 'none';
            }});
        }}
        
        function searchTraces() {{
            const endpoint = document.getElementById('search-endpoint').value;
            const status = document.getElementById('search-status').value;
            const minDuration = document.getElementById('search-min-duration').value;
            const maxDuration = document.getElementById('search-max-duration').value;

            let url = '/api/tracing/search?';
            if (endpoint) url += 'endpoint=' + encodeURIComponent(endpoint) + '&';
            if (status) url += 'status=' + encodeURIComponent(status) + '&';
            if (minDuration) url += 'min_duration=' + encodeURIComponent(minDuration) + '&';
            if (maxDuration) url += 'max_duration=' + encodeURIComponent(maxDuration) + '&';

            showLoading('trace-list');
            fetch(url)
                .then(response => response.json())
                .then(data => {{
                    renderTraceList(data);
                    document.getElementById('trace-count').textContent = '共 ' + data.length + ' 条';
                }})
                .catch(err => {{
                    showToast('搜索失败: ' + err.message, 'error');
                    document.getElementById('trace-list').innerHTML = '<div class="empty-state"><div class="empty-state-icon">⚠️</div>搜索失败，请重试</div>';
                }});
        }}
        
        function loadRecentTraces() {{
            showLoading('trace-list');
            fetch('/api/tracing/recent')
                .then(response => response.json())
                .then(data => {{
                    renderTraceList(data);
                    document.getElementById('trace-count').textContent = '共 ' + data.length + ' 条';
                }})
                .catch(err => {{
                    showToast('加载失败: ' + err.message, 'error');
                    document.getElementById('trace-list').innerHTML = '<div class="empty-state"><div class="empty-state-icon">⚠️</div>加载失败，请重试</div>';
                }});
        }}
        
        function renderTraceList(traces) {{
            const listDiv = document.getElementById('trace-list');
            let html = '';
            traces.forEach(trace => {{
                const statusClass = trace.status.toLowerCase();
                html += '<div class="trace-item ' + statusClass + '" onclick="loadTraceDetails(\\'' + trace.trace_id + '\\')">' +
                    '<div class="trace-header">' +
                        '<span class="trace-id">' + trace.trace_id.substring(0, 16) + '...</span>' +
                        '<span class="trace-duration">' + trace.duration_ms + 'ms</span>' +
                    '</div>' +
                    '<div class="trace-time">' + (trace.start_time_human || '') + '</div>' +
                    '<div class="trace-spans">' +
                        trace.span_count + ' 个Span | 状态: ' + trace.status +
                    '</div>' +
                '</div>';
            }});
            listDiv.innerHTML = html || '<div class="empty-state">暂无追踪记录</div>';
        }}
        
        function viewSpanTree() {{
            if (!currentTraceId) {{
                showToast('请先选择一条追踪记录', 'info');
                return;
            }}
            showLoading('span-tree-container');
            fetch('/api/trace/' + currentTraceId + '/tree')
                .then(response => response.json())
                .then(data => {{
                    const container = document.getElementById('span-tree-container');
                    if (data.error) {{
                        container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">⚠️</div>加载失败: ' + data.error + '</div>';
                    }} else {{
                        container.innerHTML = renderSpanTreeHTML(data);
                    }}
                    document.getElementById('trace-detail').style.display = 'none';
                    container.style.display = 'block';
                }})
                .catch(err => {{
                    showToast('加载Span树失败: ' + err.message, 'error');
                }});
        }}
        
        function viewRootSpan() {{
            if (!currentTraceId) {{
                showToast('请先选择一条追踪记录', 'info');
                return;
            }}
            showLoading('span-tree-container');
            fetch('/api/trace/' + currentTraceId + '/root')
                .then(response => response.json())
                .then(data => {{
                    const container = document.getElementById('span-tree-container');
                    if (data.error) {{
                        container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">⚠️</div>加载失败: ' + data.error + '</div>';
                    }} else {{
                        container.innerHTML = renderSpanDetailHTML(data);
                    }}
                    document.getElementById('trace-detail').style.display = 'none';
                    container.style.display = 'block';
                }})
                .catch(err => {{
                    showToast('加载根Span失败: ' + err.message, 'error');
                }});
        }}
        
        function renderSpanTreeHTML(data) {{
            let html = '<h3 style="margin-bottom: 15px; color: var(--primary-color);">🌳 Span树结构</h3>';
            
            function renderNode(span, level) {{
                if (level === undefined) level = 0;
                const indent = level * 20;
                const duration = span.duration_ms || 0;
                const name = span.name || 'unknown';
                const status = span.status?.code || 'UNSET';
                const statusColor = status === 'ERROR' ? 'var(--error-color)' : 'var(--success-color)';
                
                let nodeHtml = '<div class="span-tree-node-dark" style="margin-left: ' + indent + 'px;">' +
                    '<div style="display: flex; justify-content: space-between; align-items: center;">' +
                        '<span style="font-weight: 600; color: var(--text-primary);">' + name + '</span>' +
                        '<span style="color: ' + statusColor + '; font-size: 0.9em; font-weight: 600;">' + duration + 'ms</span>' +
                    '</div>' +
                    '<div style="font-size: 0.8em; color: var(--text-muted); margin-top: 4px;">Span ID: ' + span.span_id + '</div>';
                
                if (span.attributes) {{
                    nodeHtml += '<div class="span-tree-attr-dark">';
                    Object.entries(span.attributes).forEach(function(entry) {{
                        var key = entry[0];
                        var value = entry[1];
                        nodeHtml += '<div><strong>' + key + ':</strong> ' + value + '</div>';
                    }});
                    nodeHtml += '</div>';
                }}
                
                if (span.children && span.children.length > 0) {{
                    span.children.forEach(function(child) {{
                        nodeHtml += renderNode(child, level + 1);
                    }});
                }}
                
                nodeHtml += '</div>';
                return nodeHtml;
            }}
            
            data.root_spans.forEach(function(root) {{
                html += renderNode(root);
            }});
            
            return html;
        }}
        
        function renderSpanDetailHTML(span) {{
            const duration = span.duration_ms || 0;
            const name = span.name || 'unknown';
            const status = span.status?.code || 'UNSET';
            const statusColor = status === 'ERROR' ? 'var(--error-color)' : 'var(--success-color)';
            
            let html = '<h3 style="margin-bottom: 15px; color: var(--primary-color);">🌱 根Span详情</h3>' +
                '<div class="span-tree-node-dark" style="background: rgba(102, 126, 234, 0.08); border-color: rgba(102, 126, 234, 0.2);">' +
                    '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">' +
                        '<span style="font-weight: 600; font-size: 1.1em; color: var(--text-primary);">' + name + '</span>' +
                        '<span style="color: ' + statusColor + '; font-size: 1em; font-weight: 600;">' + duration + 'ms</span>' +
                    '</div>' +
                    '<div style="font-size: 0.9em; color: var(--text-secondary); margin-bottom: 15px;">' +
                        '<div><strong>Span ID:</strong> ' + span.span_id + '</div>' +
                        '<div><strong>Trace ID:</strong> ' + span.trace_id + '</div>' +
                        '<div><strong>开始时间:</strong> ' + (span.start_time_human || span.start_time) + '</div>' +
                        '<div><strong>结束时间:</strong> ' + (span.end_time_human || span.end_time) + '</div>' +
                        '<div><strong>状态:</strong> <span style="color: ' + statusColor + ';">' + status + '</span></div>' +
                    '</div>';
            
            if (span.attributes) {{
                html += '<div style="margin-top: 15px;">' +
                    '<h4 style="margin-bottom: 8px; color: var(--primary-color);">属性</h4>' +
                    '<div class="span-tree-attr-dark" style="max-height: 200px;">';
                Object.entries(span.attributes).forEach(function(entry) {{
                    var key = entry[0];
                    var value = entry[1];
                    html += '<div><strong>' + key + ':</strong> ' + value + '</div>';
                }});
                html += '</div></div>';
            }}
            
            html += '</div>';
            return html;
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

        // === Loading spinner ===
        function showLoading(elementId) {{
            const el = document.getElementById(elementId);
            if (el) {{
                el.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><div class="loading-text">加载中...</div></div>';
            }}
        }}

        // === Toast notification ===
        function showToast(message, type) {{
            type = type || 'error';
            const container = document.getElementById('toast-container');
            if (!container) return;
            
            const icons = {{ error: '⚠️', success: '✅', info: 'ℹ️' }};
            const toast = document.createElement('div');
            toast.className = 'toast ' + type;
            toast.innerHTML = 
                '<span class="toast-icon">' + (icons[type] || '⚠️') + '</span>' +
                '<span class="toast-message">' + message + '</span>' +
                '<button class="toast-close" onclick="this.parentElement.remove()">✕</button>';
            container.appendChild(toast);
            
            setTimeout(function() {{
                toast.style.animation = 'toastOut 0.3s ease-in forwards';
                setTimeout(function() {{ toast.remove(); }}, 300);
            }}, 5000);
        }}

        // === Theme toggle ===
        function toggleTheme() {{
            const body = document.body;
            const current = body.getAttribute('data-theme');
            const newTheme = current === 'light' ? 'dark' : 'light';
            
            if (newTheme === 'light') {{
                body.setAttribute('data-theme', 'light');
                document.getElementById('theme-icon').textContent = '☀️';
                document.getElementById('theme-label').textContent = '浅色';
            }} else {{
                body.removeAttribute('data-theme');
                document.getElementById('theme-icon').textContent = '🌙';
                document.getElementById('theme-label').textContent = '深色';
            }}
            
            try {{ localStorage.setItem('monitor-theme', newTheme); }} catch(e) {{}}
        }}

        // === Auto-refresh ===
        let autoRefreshTimer = null;
        let autoRefreshEnabled = true;
        let refreshCountdown = 30;

        function toggleAutoRefresh() {{
            autoRefreshEnabled = !autoRefreshEnabled;
            const btn = document.getElementById('auto-refresh-btn');
            const dot = document.getElementById('refresh-dot');
            const label = document.getElementById('auto-refresh-label');
            
            if (autoRefreshEnabled) {{
                btn.classList.add('active');
                dot.classList.remove('paused');
                label.textContent = '自动刷新 30s';
                startAutoRefresh();
                showToast('自动刷新已开启', 'success');
            }} else {{
                btn.classList.remove('active');
                dot.classList.add('paused');
                label.textContent = '自动刷新已暂停';
                stopAutoRefresh();
                showToast('自动刷新已暂停', 'info');
            }}
        }}

        function startAutoRefresh() {{
            stopAutoRefresh();
            refreshCountdown = 30;
            autoRefreshTimer = setInterval(function() {{
                refreshCountdown--;
                if (refreshCountdown <= 0) {{
                    refreshActiveTab();
                    refreshCountdown = 30;
                }}
                if (autoRefreshEnabled) {{
                    document.getElementById('auto-refresh-label').textContent = '自动刷新 ' + refreshCountdown + 's';
                }}
            }}, 1000);
        }}

        // 统一刷新：服务状态 + 当前激活 tab 的数据
        function refreshActiveTab() {{
            // 始终刷新服务卡片（核心观测）
            refreshServiceStatus();
            // 按当前 tab 刷新对应数据
            if (currentTab === 'tracing') {{
                refreshTracingStats(tracingTimeRange);
                loadRecentTraces();
            }} else if (currentTab === 'cleaning') {{
                refreshCleaningProgress();
            }} else if (currentTab === 'topology') {{
                fetch('/api/topology').then(r => r.json()).then(data => {{
                    // 重新渲染拓扑节点状态
                    data.nodes.forEach(function(node) {{
                        const el = document.querySelector('.topo-node[data-id="' + node.id + '"]');
                        if (el) {{
                            const statusEl = el.querySelector('.topo-node-status');
                            if (statusEl) {{
                                statusEl.textContent = node.status;
                                statusEl.className = 'topo-node-status ' + node.status;
                            }}
                            el.className = 'topo-node ' + node.status;
                        }}
                    }});
                }}).catch(function() {{}});
            }}
        }}

        function stopAutoRefresh() {{
            if (autoRefreshTimer) {{
                clearInterval(autoRefreshTimer);
                autoRefreshTimer = null;
            }}
        }}

        // === AJAX service status refresh (no full page reload) ===
        function refreshServiceStatus() {{
            const grid = document.querySelector('.services-grid');
            if (!grid) {{
                location.reload();
                return;
            }}
            
            // Add refreshing animation
            grid.querySelectorAll('.service-card').forEach(function(card) {{
                card.classList.add('refreshing');
            }});
            
            fetch('/api/services/status')
                .then(response => response.json())
                .then(data => {{
                    if (Array.isArray(data)) {{
                        updateServiceCards(data);
                    }}
                    grid.querySelectorAll('.service-card').forEach(function(card) {{
                        card.classList.remove('refreshing');
                        card.classList.add('refreshed');
                        setTimeout(function() {{ card.classList.remove('refreshed'); }}, 600);
                    }});
                }})
                .catch(err => {{
                    grid.querySelectorAll('.service-card').forEach(function(card) {{
                        card.classList.remove('refreshing');
                    }});
                    showToast('刷新服务状态失败: ' + err.message, 'error');
                }});
        }}

        function updateServiceCards(services) {{
            const grid = document.querySelector('.services-grid');
            if (!grid) return;
            
            // Update stat cards
            const total = services.length;
            const healthy = services.filter(s => s.status === 'healthy').length;
            const enabled = services.filter(s => s.enabled).length;
            const coreServices = services.filter(s => s.is_core);
            const coreHealthy = coreServices.filter(s => s.status === 'healthy').length;
            
            const statValues = grid.parentElement.querySelectorAll('.stat-card .value');
            if (statValues.length >= 4) {{
                statValues[0].textContent = coreHealthy + '/' + coreServices.length;
                statValues[1].textContent = enabled + '/' + total;
                statValues[2].textContent = healthy + '/' + enabled;
                statValues[3].textContent = total;
            }}
            
            // Update individual service cards
            services.forEach(function(service) {{
                const card = grid.querySelector('.service-card[data-service-id="' + service.key + '"]');
                if (card) {{
                    // Update status indicator
                    const indicator = card.querySelector('.status-indicator');
                    if (indicator) {{
                        indicator.className = 'status-indicator status-' + service.status;
                    }}
                    
                    // Update status text
                    const statusTexts = {{
                        'healthy': '✅ 运行正常',
                        'unreachable': '❌ 无法连接',
                        'timeout': '⏰ 超时',
                        'disabled': '⏸️ 已禁用',
                        'unknown': '❓ 未知'
                    }};
                    const statusSpan = card.querySelector('.service-header span:nth-child(3)');
                    if (statusSpan) {{
                        statusSpan.textContent = statusTexts[service.status] || service.status;
                    }}
                    
                    // Update response time
                    const infoValues = card.querySelectorAll('.info-value');
                    if (infoValues.length >= 2) {{
                        infoValues[1].textContent = service.response_time + ' ms';
                        // 响应时间颜色编码
                        const rt = parseFloat(service.response_time) || 0;
                        let rtClass = 'rt-fast';
                        if (rt > 2000) rtClass = 'rt-critical';
                        else if (rt > 500) rtClass = 'rt-slow';
                        else if (rt > 100) rtClass = 'rt-normal';
                        infoValues[1].className = 'info-value ' + rtClass;
                    }}
                    
                    // Update card border status class
                    card.className = 'service-card status-' + service.status;
                    card.setAttribute('data-service-id', service.key);
                }}
            }});
        }}

        // === Initialize on page load ===
        document.addEventListener('DOMContentLoaded', function() {{
            // Tab switching
            document.querySelectorAll('[id$="-tab-content"]').forEach(c => c.style.display = 'none');
            document.getElementById('services-tab-content').style.display = 'block';

            // Apply response time color coding to initially rendered cards
            document.querySelectorAll('.service-card').forEach(function(card) {{
                const infoValues = card.querySelectorAll('.info-value');
                if (infoValues.length >= 2) {{
                    const text = infoValues[1].textContent || '';
                    const rt = parseFloat(text.replace(/[^\d.]/g, '')) || 0;
                    let rtClass = 'rt-fast';
                    if (rt > 2000) rtClass = 'rt-critical';
                    else if (rt > 500) rtClass = 'rt-slow';
                    else if (rt > 100) rtClass = 'rt-normal';
                    infoValues[1].className = 'info-value ' + rtClass;
                }}
            }});

            // Restore theme
            try {{
                const savedTheme = localStorage.getItem('monitor-theme');
                if (savedTheme === 'light') {{
                    document.body.setAttribute('data-theme', 'light');
                    document.getElementById('theme-icon').textContent = '☀️';
                    document.getElementById('theme-label').textContent = '浅色';
                }}
            }} catch(e) {{}}

            // Start auto-refresh
            const autoBtn = document.getElementById('auto-refresh-btn');
            if (autoBtn) {{
                autoBtn.classList.add('active');
                startAutoRefresh();
            }}
        }});
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
    hours = request.args.get("hours", 24, type=int)
    return jsonify(get_tracing_stats(hours))


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
    return jsonify(get_cleaning_progress())


@app.route("/api/cleaning/reset")
def cleaning_reset():
    """重置数据清理进度"""
    tracker = CleaningProgressTracker()
    tracker.reset_progress()
    return jsonify({"status": "success", "message": "进度已重置"})


@app.route("/api/trace/<trace_id>/tree")
def trace_span_tree(trace_id):
    """获取追踪的Span树结构"""
    return jsonify(get_trace_span_tree(trace_id))


@app.route("/api/trace/<trace_id>/root")
def trace_root_span(trace_id):
    """获取追踪的根Span"""
    return jsonify(get_trace_root_span(trace_id))


@app.route("/api/trace/<trace_id>/span/<span_id>")
def span_details(trace_id, span_id):
    """获取指定Span的详情"""
    return jsonify(get_span_details(trace_id, span_id))


@app.route("/api/trace/<trace_id>/span/<span_id>/children")
def span_children(trace_id, span_id):
    """获取指定Span的子Span"""
    return jsonify(get_child_spans(trace_id, span_id))


@app.route("/api/tracing/search")
def search_traces_endpoint():
    """搜索追踪记录"""
    endpoint = request.args.get("endpoint")
    status = request.args.get("status")
    min_duration = request.args.get("min_duration", type=float)
    max_duration = request.args.get("max_duration", type=float)
    limit = request.args.get("limit", 20, type=int)
    return jsonify(search_traces_api(endpoint, status, min_duration, max_duration, limit))


if __name__ == "__main__":
    print(f"🚀 监控仪表板启动在 http://localhost:{MONITOR_PORT}")
    print(f"   📊 服务监控: http://localhost:{MONITOR_PORT}/")
    print(f"   🔗 API: http://localhost:{MONITOR_PORT}/api/health")
    app.run(
        host="0.0.0.0",
        port=MONITOR_PORT,
        debug=False,
    )
