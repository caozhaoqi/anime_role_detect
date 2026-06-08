#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控仪表板HTML模板生成模块
"""

from typing import List, Dict


def generate_trace_items_html(traces):
    """生成追踪列表HTML"""
    html = ""
    for trace in traces:
        status_class = trace['status'].lower()
        trace_id = trace['trace_id']
        duration_ms = trace['duration_ms']
        start_time = trace.get('start_time_human', '')
        span_count = trace['span_count']
        status = trace['status']
        
        html += f"""
            <div class="trace-item {status_class}" onclick="loadTraceDetails('{trace_id}')">
                <div class="trace-header">
                    <span class="trace-id">{trace_id[:16]}...</span>
                    <span class="trace-duration">{duration_ms}ms</span>
                </div>
                <div class="trace-time">{start_time}</div>
                <div class="trace-spans">
                    {span_count} 个Span | 状态: {status}
                </div>
            </div>
        """
    return html


def generate_endpoint_stats_html(endpoint_distribution):
    """生成端点分布统计HTML"""
    if not endpoint_distribution:
        return ""
    
    max_count = max(endpoint_distribution.values())
    html = ""
    
    for endpoint, count in sorted(endpoint_distribution.items(), key=lambda x: -x[1]):
        percentage = (count / max_count) * 100
        html += f"""
            <div class="endpoint-bar">
                <div class="endpoint-name">{endpoint}</div>
                <div class="endpoint-bar-fill">
                    <div class="endpoint-bar-inner" style="width: {percentage}%;"></div>
                </div>
                <div style="color: #888; font-size: 0.9em; width: 60px; text-align: right;">{count}</div>
            </div>
        """
    return html


def generate_trace_tree_html(trace):
    """生成追踪树形结构HTML"""
    spans = trace.get("spans", [])
    
    span_dict = {span["span_id"]: span for span in spans}
    
    for span_id, span_data in span_dict.items():
        span_data["children"] = []
    
    for span_id, span_data in span_dict.items():
        parent_id = span_data.get("parent_span_id")
        if parent_id and parent_id in span_dict:
            span_dict[parent_id]["children"].append(span_data)
    
    root = None
    for span_id, span_data in span_dict.items():
        if not span_data.get("parent_span_id"):
            root = span_data
            break
    
    def render_span(span, level=0):
        status_code = span.get("status", {}).get("code", "UNSET")
        status_class = {
            "OK": "status-ok",
            "ERROR": "status-error",
            "UNSET": "status-unset",
        }.get(status_code, "status-unset")
        
        kind = span.get("kind", "INTERNAL")
        kind_color = {
            "SERVER": "#4CAF50",
            "CLIENT": "#2196F3",
            "INTERNAL": "#9E9E9E",
            "PRODUCER": "#FF9800",
            "CONSUMER": "#E91E63",
        }.get(kind, "#9E9E9E")
        
        padding_left = level * 20

        attributes_html = ""
        attributes = span.get("attributes", {})
        if attributes:
            attr_lines = []
            for key, value in attributes.items():
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                attr_lines.append(f"<div><strong>{key}:</strong> {value}</div>")
            attributes_html = f"""
                <div class="span-attributes" style="display: none;">
                    <div class="attr-title">属性:</div>
                    {"".join(attr_lines)}
                </div>
            """
        
        events_html = ""
        events = span.get("events", [])
        if events:
            event_lines = []
            for event in events:
                event_lines.append(f"<div>• {event['name']}</div>")
            events_html = f"""
                <div class="span-events" style="display: none;">
                    <div class="attr-title">事件:</div>
                    {"".join(event_lines)}
                </div>
            """
        
        children_html = ""
        if span.get("children"):
            for child in span["children"]:
                children_html += render_span(child, level + 1)
        
        return f"""
            <div 
                class="span-item {status_class}" 
                style="padding-left: {padding_left}px;"
                onclick="toggleSpanDetails(this)"
            >
                <div class="span-header">
                    <span class="span-kind" style="background: {kind_color};">{kind}</span>
                    <span class="span-name">{span['name']}</span>
                    <span class="span-duration">{span['duration_ms']}ms</span>
                    <span class="span-status {status_class}">{status_code}</span>
                </div>
                {attributes_html}
                {events_html}
                {children_html}
            </div>
        """
    
    if root:
        return render_span(root)
    return "<div class='empty-state'>暂无Span数据</div>"


def generate_service_monitor_html(services_status: List[dict]) -> str:
    """生成服务监控HTML"""
    total = len(services_status)
    healthy = sum(1 for s in services_status if s["status"] == "healthy")
    enabled = sum(1 for s in services_status if s["enabled"])
    core_services = [s for s in services_status if s["is_core"]]
    core_healthy = sum(1 for s in core_services if s["status"] == "healthy")

    html = f"""
<div class="tab-content active" id="services-tab">
    <div class="stats-grid">
        <div class="stat-card">
            <h3>核心服务健康</h3>
            <div class="value">{core_healthy}/{len(core_services)}</div>
        </div>
        <div class="stat-card">
            <h3>已启用服务</h3>
            <div class="value">{enabled}/{total}</div>
        </div>
        <div class="stat-card">
            <h3>健康服务</h3>
            <div class="value">{healthy}/{enabled}</div>
        </div>
        <div class="stat-card">
            <h3>总服务数</h3>
            <div class="value">{total}</div>
        </div>
    </div>

    <div class="services-grid">
"""

    for service in services_status:
        status_class = f"status-{service['status']}"
        badge_class = "badge-core" if service["is_core"] else "badge-aux"
        badge_text = "核心服务" if service["is_core"] else "辅助服务"

        status_text = {
            "healthy": "✅ 运行正常",
            "unreachable": "❌ 无法连接",
            "timeout": "⏰ 超时",
            "disabled": "⏸️ 已禁用",
            "unknown": "❓ 未知",
        }.get(service["status"], f"⚠️ {service['status']}")

        api_url = f"http://localhost:{service['port']}"
        docs_url = f"{api_url}/docs"

        if service["status"] == "healthy":
            links_html = f"""
                <div class="api-links">
                    <a href="{api_url}" class="api-link" target="_blank">🔗 访问API</a>
                    <a href="{docs_url}" class="api-link" target="_blank">📚 Swagger文档</a>
                </div>
            """
        else:
            links_html = ""

        html += f"""
            <div class="service-card">
                <div class="service-header">
                    <div>
                        <span class="status-indicator {status_class}"></span>
                        <span class="service-name">{service['name']}</span>
                    </div>
                    <span class="service-badge {badge_class}">{badge_text}</span>
                </div>
                
                <div class="service-info">
                    <div class="info-item">
                        <div class="info-label">状态</div>
                        <div class="info-value">{status_text}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">端口</div>
                        <div class="info-value">{service['port']}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">响应时间</div>
                        <div class="info-value">{service['response_time']} ms</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">最后检查</div>
                        <div class="info-value" style="font-size: 0.9em;">{service['last_check']}</div>
                    </div>
                </div>
                
                {links_html}
            </div>
"""

    html += """
    </div>
</div>
"""

    return html


def generate_tracing_html(stats, recent_traces) -> str:
    """生成链路追踪HTML"""

    html = f"""
<div class="tab-content" id="tracing-tab">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <h2 style="color: #667eea;">🔗 API链路追踪</h2>
        <button class="refresh-btn" onclick="refreshTracing()">🔄 刷新追踪数据</button>
    </div>
    
    <div id="tracing-stats">
        <div class="stats-grid">
            <div class="stat-card">
                <h3>总追踪数</h3>
                <div class="value">{stats.get('total_traces', 0)}</div>
            </div>
            <div class="stat-card success">
                <h3>成功请求</h3>
                <div class="value">{stats.get('success_count', 0)}</div>
            </div>
            <div class="stat-card error">
                <h3>错误请求</h3>
                <div class="value">{stats.get('error_count', 0)}</div>
            </div>
            <div class="stat-card warning">
                <h3>错误率</h3>
                <div class="value">{stats.get('error_rate', 0)}%</div>
            </div>
            <div class="stat-card">
                <h3>平均耗时</h3>
                <div class="value">{stats.get('avg_duration_ms', 0)}ms</div>
            </div>
            <div class="stat-card">
                <h3>最大耗时</h3>
                <div class="value">{stats.get('max_duration_ms', 0)}ms</div>
            </div>
        </div>
    </div>

    <div class="main-content">
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">📋 最近追踪记录</span>
                <span style="color: #888; font-size: 0.9em;">共 {len(recent_traces)} 条</span>
            </div>
            
            <div id="trace-list" class="trace-list">
                {generate_trace_items_html(recent_traces)}
                
                {"" if recent_traces else "<div class='empty-state'>暂无追踪记录</div>"}
            </div>
        </div>
        
        <div class="panel detail-panel">
            <div class="panel-header">
                <span class="panel-title">🔍 追踪详情</span>
            </div>
            
            <div id="trace-detail">
                <div class="empty-state">选择一个追踪查看详情</div>
            </div>
        </div>
    </div>

    <div class="panel" style="margin-top: 20px;">
        <div class="panel-header">
            <span class="panel-title">📊 端点分布统计</span>
        </div>
        
        <div style="max-height: 300px; overflow-y: auto;">
            {generate_endpoint_stats_html(stats.get('endpoint_distribution', {}))}
            
            {"" if stats.get('endpoint_distribution', {}) else "<div class='empty-state'>暂无端点数据</div>"}
        </div>
    </div>
</div>
"""
    return html


def generate_tracing_stats_html(stats):
    """生成追踪统计HTML（用于动态刷新）"""
    return f"""
        <div class="stats-grid">
            <div class="stat-card">
                <h3>总追踪数</h3>
                <div class="value">{stats.get('total_traces', 0)}</div>
            </div>
            <div class="stat-card success">
                <h3>成功请求</h3>
                <div class="value">{stats.get('success_count', 0)}</div>
            </div>
            <div class="stat-card error">
                <h3>错误请求</h3>
                <div class="value">{stats.get('error_count', 0)}</div>
            </div>
            <div class="stat-card warning">
                <h3>错误率</h3>
                <div class="value">{stats.get('error_rate', 0)}%</div>
            </div>
            <div class="stat-card">
                <h3>平均耗时</h3>
                <div class="value">{stats.get('avg_duration_ms', 0)}ms</div>
            </div>
            <div class="stat-card">
                <h3>最大耗时</h3>
                <div class="value">{stats.get('max_duration_ms', 0)}ms</div>
            </div>
        </div>
    """


def generate_topology_html(topology_data) -> str:
    """生成微服务调用拓扑图HTML"""
    nodes = topology_data.get('nodes', [])
    edges = topology_data.get('edges', [])
    
    # 定义节点位置布局
    node_positions = {
        "api_gateway": {"x": 50, "y": 50},
        "api_service": {"x": 50, "y": 200},
        "multimedia_service": {"x": 50, "y": 350},
        "model_service": {"x": 50, "y": 500},
    }
    
    # 生成SVG连线
    edges_svg = ""
    for edge in edges:
        source_pos = node_positions.get(edge['source'], {"x": 100, "y": 100})
        target_pos = node_positions.get(edge['target'], {"x": 300, "y": 100})
        
        # 从源节点右侧到目标节点左侧的贝塞尔曲线
        start_x = source_pos["x"] + 120  # 节点宽度的一半
        start_y = source_pos["y"] + 40   # 节点高度的一半
        end_x = target_pos["x"]
        end_y = target_pos["y"] + 40
        
        mid_x = (start_x + end_x) / 2
        
        edges_svg += f"""
            <path 
                d="M {start_x} {start_y} C {mid_x} {start_y}, {mid_x} {end_y}, {end_x} {end_y}"
                stroke="#667eea" 
                stroke-width="2" 
                fill="none"
                marker-end="url(#arrowhead)"
                class="edge-line"
            />
        """
    
    # 生成节点
    nodes_html = ""
    for node in nodes:
        pos = node_positions.get(node['id'], {"x": 100, "y": 100})
        status = node['status']
        is_core = node['is_core']
        
        status_color = {
            "healthy": "#4CAF50",
            "unreachable": "#f44336",
            "timeout": "#ff9800",
            "disabled": "#9E9E9E",
        }.get(status, "#9E9E9E")
        
        border_class = "border-core" if is_core else ""
        
        nodes_html += f"""
            <div 
                class="topology-node {border_class}" 
                style="left: {pos['x']}px; top: {pos['y']}px;"
            >
                <div class="node-header">
                    <span class="node-status" style="background: {status_color};"></span>
                    <span class="node-name">{node['name']}</span>
                </div>
                <div class="node-info">
                    <div>端口: <strong>{node['port']}</strong></div>
                    <div>状态: <span class="status-text" style="color: {status_color};">{status}</span></div>
                    <div>响应: <strong>{node['response_time']}ms</strong></div>
                </div>
            </div>
        """
    
    html = f"""
<div class="tab-content" id="topology-tab">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <h2 style="color: #667eea;">📊 微服务调用拓扑图</h2>
        <button class="refresh-btn" onclick="refreshTopology()">🔄 刷新拓扑图</button>
    </div>
    
    <div class="topology-container">
        <svg class="topology-svg" width="100%" height="100%">
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#667eea" />
                </marker>
            </defs>
            {edges_svg}
        </svg>
        <div class="topology-nodes">
            {nodes_html}
        </div>
    </div>
    
    <div class="legend">
        <div class="legend-item">
            <span class="legend-dot healthy"></span>
            <span>运行正常</span>
        </div>
        <div class="legend-item">
            <span class="legend-dot unreachable"></span>
            <span>无法连接</span>
        </div>
        <div class="legend-item">
            <span class="legend-dot timeout"></span>
            <span>超时</span>
        </div>
        <div class="legend-item">
            <span class="legend-dot disabled"></span>
            <span>已禁用</span>
        </div>
        <div class="legend-item">
            <span class="legend-core"></span>
            <span>核心服务</span>
        </div>
    </div>
</div>
"""
    return html