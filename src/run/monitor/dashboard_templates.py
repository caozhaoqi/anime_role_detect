#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控仪表板HTML模板生成模块
"""

from typing import List, Dict
import json


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
        return '<div class="empty-state" style="color: #666; text-align: center; padding: 20px;">暂无端点统计数据</div>'
    
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

        api_url = f"http://localhost:{service['port']}{service.get('api_base', '/')}"
        has_swagger = service.get("has_swagger", False)

        if service["status"] == "healthy":
            links_parts = [
                f'                    <a href="{api_url}" class="api-link" target="_blank">🔗 访问API</a>',
            ]
            if has_swagger:
                docs_url = f"http://localhost:{service['port']}/docs"
                links_parts.append(f'                    <a href="{docs_url}" class="api-link" target="_blank">📚 Swagger文档</a>')
            links_html = '\n'.join(links_parts)
            links_html = f"""
                <div class="api-links">
{links_html}
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
                """
    
    # 预计算端点统计HTML（避免在f-string内嵌 {} 导致语法错误）
    endpoint_html = generate_endpoint_stats_html(stats.get("endpoint_distribution", {}))

    html += f"""
            </div>
        </div>

        <div class="panel detail-panel">
            <div class="panel-header">
                <span class="panel-title">📊 追踪详情</span>
            </div>
            <div id="trace-detail" class="panel-content">
                <div class='empty-state'>点击左侧追踪记录查看详情</div>
            </div>
        </div>
    </div>

    <div class="main-content">
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">📈 端点分布</span>
            </div>
            <div class="panel-content">
                {endpoint_html}
            </div>
        </div>
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">🔁 时序图</span>
            </div>
            <div class="panel-content" id="gantt-chart">
                <div class='empty-state'>选择追踪记录后可查看时序图</div>
            </div>
        </div>
    </div>
</div>
"""
    return html


def generate_tracing_stats_html(stats):
    """生成追踪统计HTML"""
    return f"""
    <div class="stats-grid">
        <div class="stat-card success">
            <h3>总追踪数</h3>
            <div class="value">{stats.get('total_traces', 0)}</div>
        </div>
        <div class="stat-card">
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
    </div>
    """


def generate_topology_html(services_status, topology_data):
    """生成拓扑图HTML"""
    
    # 渲染节点
    nodes_html = ""
    for node in topology_data.get("nodes", []):
        status_color = {
            "healthy": "#4CAF50",
            "unreachable": "#f44336",
            "timeout": "#ff9800",
            "disabled": "#9E9E9E",
        }.get(node["status"], "#9E9E9E")
        
        nodes_html += f"""
            <div class="topo-node" style="border-color: {status_color};" data-id="{node['id']}">
                <div class="topo-node-name">{node['name']}</div>
                <div class="topo-node-port">:{node['port']}</div>
                <div class="topo-node-status" style="color: {status_color};">{node['status']}</div>
            </div>
        """
    
    # 渲染边（连线通过JS实现）
    edges_data = json.dumps(topology_data.get("edges", []))
    
    html = f"""
<div class="tab-content" id="topology-tab">
    <div class="topology-container">
        <div class="topology-header">
            <h2>📊 微服务拓扑图</h2>
        </div>
        <div class="topology-canvas" id="topology-canvas">
            <svg class="topology-svg" id="topology-svg"></svg>
            <div class="topology-nodes">
                {nodes_html}
            </div>
        </div>
    </div>
</div>
<script>
    const edgesData = {edges_data};
</script>
"""
    return html


def generate_gantt_html(trace):
    """生成甘特图/时序图HTML"""
    spans = trace.get("spans", [])
    if not spans:
        return '<div class="empty-state" style="color: #666; text-align: center; padding: 20px;">无时序数据</div>'
    
    # 找到最早开始时间
    start_times = [s.get("start_time", 0) for s in spans]
    min_start = min(start_times) if start_times else 0
    max_end = max(s.get("end_time", s.get("start_time", 0)) for s in spans)
    total_range = max_end - min_start if max_end > min_start else 1
    
    # 排序spans: 按start_time升序
    sorted_spans = sorted(spans, key=lambda s: s.get("start_time", 0))
    
    bar_rows = []
    for span in sorted_spans:
        name = span.get("name", "unknown")
        kind = span.get("kind", "INTERNAL")
        duration = span.get("duration_ms", 0)
        s_start = span.get("start_time", min_start)
        s_end = span.get("end_time", s_start)
        
        left_pct = max(0, (s_start - min_start) / total_range * 100)
        width_pct = max(3, (s_end - s_start) / total_range * 100)
        
        kind_color = {
            "SERVER": "#4CAF50",
            "CLIENT": "#2196F3",
            "INTERNAL": "#9E9E9E",
            "PRODUCER": "#FF9800",
            "CONSUMER": "#E91E63",
        }.get(kind, "#9E9E9E")
        
        bar_rows.append(f"""
            <div class="gantt-row">
                <div class="gantt-label">{name}
                    <span style="color: #888; font-size: 0.8em;">({duration}ms)</span>
                </div>
                <div class="gantt-track">
                    <div class="gantt-bar" style="
                        margin-left: {left_pct:.1f}%;
                        width: {width_pct:.1f}%;
                        background: {kind_color};
                    "></div>
                </div>
            </div>
        """)
    
    return """
        <div class="gantt-chart">
            <div class="gantt-header">
                <div class="gantt-header-label">Span名称</div>
                <div class="gantt-header-timeline">时间线</div>
            </div>
            """ + "\n".join(bar_rows) + """
        </div>
        <style>
            .gantt-chart { font-size: 0.9em; }
            .gantt-header {
                display: grid;
                grid-template-columns: 180px 1fr;
                padding: 8px 0;
                border-bottom: 1px solid #333;
                color: #888;
                font-weight: bold;
            }
            .gantt-row {
                display: grid;
                grid-template-columns: 180px 1fr;
                align-items: center;
                padding: 4px 0;
                border-bottom: 1px solid rgba(255,255,255,0.05);
            }
            .gantt-label {
                padding-right: 10px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                color: #ccc;
            }
            .gantt-track {
                position: relative;
                height: 24px;
                background: rgba(255,255,255,0.05);
                border-radius: 4px;
            }
            .gantt-bar {
                height: 20px;
                border-radius: 4px;
                opacity: 0.8;
                min-width: 4px;
                position: absolute;
                top: 2px;
            }
        </style>
    """