/**
 * 监控仪表板前端脚本
 */

function switchTab(tabName) {
    // 隐藏所有标签内容
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        content.classList.remove('active');
    });

    // 移除所有标签按钮的活跃状态
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.classList.remove('active');
    });

    // 显示选中的标签内容
    const activeContent = document.getElementById(tabName + '-tab');
    if (activeContent) {
        activeContent.classList.add('active');
    }

    // 添加标签按钮的活跃状态
    const activeTab = document.querySelector(`[onclick="switchTab('${tabName}')"]`);
    if (activeTab) {
        activeTab.classList.add('active');
    }
}

function loadTraceDetails(traceId) {
    fetch(`/api/tracing/trace/${traceId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.data) {
                displayTraceDetail(data.data);
            } else {
                document.getElementById('trace-detail').innerHTML = '<div class="empty-state">加载失败</div>';
            }
        })
        .catch(error => {
            console.error('加载追踪详情失败:', error);
            document.getElementById('trace-detail').innerHTML = '<div class="empty-state">加载失败</div>';
        });
}

function displayTraceDetail(trace) {
    const detailDiv = document.getElementById('trace-detail');
    
    if (!trace) {
        detailDiv.innerHTML = '<div class="empty-state">暂无数据</div>';
        return;
    }

    const spans = trace.spans || [];
    const duration_ms = trace.duration_ms || 0;
    const start_time = trace.start_time_human || '';

    // 创建Span表格
    let spansHtml = '<div class="span-tree">';
    
    // 构建Span树
    const spanDict = {};
    spans.forEach(span => {
        span.children = [];
        spanDict[span.span_id] = span;
    });

    // 构建父子关系
    let rootSpan = null;
    spans.forEach(span => {
        const parentId = span.parent_span_id;
        if (parentId && spanDict[parentId]) {
            spanDict[parentId].children.push(span);
        } else if (!parentId) {
            rootSpan = span;
        }
    });

    // 递归渲染Span树
    function renderSpan(span, level = 0) {
        const statusCode = span.status?.code || 'UNSET';
        const kind = span.kind || 'INTERNAL';
        const kindColor = {
            'SERVER': '#4CAF50',
            'CLIENT': '#2196F3',
            'INTERNAL': '#9E9E9E',
            'PRODUCER': '#FF9800',
            'CONSUMER': '#E91E63'
        }[kind] || '#9E9E9E';

        let html = `
            <div class="span-item" style="padding-left: ${level * 20}px;">
                <div class="span-header" onclick="toggleSpan(this)">
                    <span class="span-kind" style="background: ${kindColor};">${kind}</span>
                    <span class="span-name">${span.name}</span>
                    <span class="span-duration">${span.duration_ms}ms</span>
                    <span class="span-status status-${statusCode.toLowerCase()}">${statusCode}</span>
                </div>
        `;

        // 添加属性
        if (span.attributes && Object.keys(span.attributes).length > 0) {
            html += '<div class="span-attributes" style="display: none;">';
            html += '<div class="attr-title">属性:</div>';
            for (const [key, value] of Object.entries(span.attributes)) {
                const displayValue = typeof value === 'string' && value.length > 50 
                    ? value.substring(0, 50) + '...' 
                    : value;
                html += `<div><strong>${key}:</strong> ${displayValue}</div>`;
            }
            html += '</div>';
        }

        // 添加子Span
        if (span.children && span.children.length > 0) {
            span.children.forEach(child => {
                html += renderSpan(child, level + 1);
            });
        }

        html += '</div>';
        return html;
    }

    if (rootSpan) {
        spansHtml += renderSpan(rootSpan);
    } else if (spans.length > 0) {
        spansHtml += renderSpan(spans[0]);
    } else {
        spansHtml += '<div class="empty-state">暂无Span数据</div>';
    }
    
    spansHtml += '</div>';

    detailDiv.innerHTML = `
        <div class="trace-summary">
            <div class="summary-row">
                <span class="summary-label">Trace ID:</span>
                <span class="summary-value trace-id-full">${trace.trace_id}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">状态:</span>
                <span class="summary-value status-${trace.status.toLowerCase()}">${trace.status}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">总耗时:</span>
                <span class="summary-value">${duration_ms}ms</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">开始时间:</span>
                <span class="summary-value">${start_time}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Span数量:</span>
                <span class="summary-value">${spans.length}</span>
            </div>
        </div>
        <div class="trace-spans-header">
            <h4>📋 Span调用链</h4>
        </div>
        ${spansHtml}
    `;
}

function toggleSpan(element) {
    const nextEl = element.nextElementSibling;
    if (nextEl && nextEl.classList.contains('span-attributes')) {
        nextEl.style.display = nextEl.style.display === 'none' ? 'block' : 'none';
    }
}

function refreshDashboard() {
    fetch('/api/reload')
        .then(response => response.text())
        .then(html => {
            document.open();
            document.write(html);
            document.close();
        });
}

function refreshTracing() {
    fetch('/api/tracing/reload')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('tracing-stats').innerHTML = data.stats_html;
                document.getElementById('trace-list').innerHTML = data.traces_html;
                document.getElementById('last-update').textContent = '最后更新: ' + new Date().toLocaleString();
            }
        });
}

function refreshTopology() {
    fetch('/api/topology/reload')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('topology-tab').innerHTML = data.html;
                document.getElementById('last-update').textContent = '最后更新: ' + new Date().toLocaleString();
            }
        });
}