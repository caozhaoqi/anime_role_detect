#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控仪表板

提供系统运行状态的实时可视化界面
"""

import os
import sys
import json
from datetime import datetime
from flask import Flask, render_template, jsonify, request

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.monitoring_system import MonitoringSystem

app = Flask(__name__, template_folder='../templates')

# 创建监控系统实例
monitoring_system = MonitoringSystem()

# 启动监控系统
monitoring_system.start()


@app.route('/')
def index():
    """
    仪表板首页
    """
    return render_template('dashboard.html')


@app.route('/api/stats')
def get_stats():
    """
    获取监控统计信息
    """
    try:
        stats = monitoring_system.get_all_stats()
        return jsonify({
            'success': True,
            'data': stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/alerts')
def get_alerts():
    """
    获取告警信息
    """
    try:
        limit = int(request.args.get('limit', 10))
        alerts = monitoring_system.get_alerts(limit=limit)
        return jsonify({
            'success': True,
            'data': alerts,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/monitor-data/<monitor_name>')
def get_monitor_data(monitor_name):
    """
    获取监控器数据
    """
    try:
        limit = int(request.args.get('limit', 20))
        data = monitoring_system.get_monitor_data(monitor_name, limit=limit)
        return jsonify({
            'success': True,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/dashboard')
def get_dashboard_data():
    """
    获取仪表板数据
    """
    try:
        data = monitoring_system.get_dashboard_data()
        return jsonify({
            'success': True,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/save-stats')
def save_stats():
    """
    保存统计信息
    """
    try:
        output_file = request.args.get('output_file', './monitoring_stats.json')
        monitoring_system.save_stats(output_file=output_file)
        return jsonify({
            'success': True,
            'message': f'统计信息已保存到 {output_file}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    # 创建templates目录
    templates_dir = os.path.join(os.path.dirname(__file__), '../templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # 创建dashboard.html模板
    dashboard_html = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>系统监控仪表板</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.3.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.4/dist/jquery.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }
        .dashboard-container {
            margin-top: 20px;
        }
        .card {
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .card-header {
            background-color: #007bff;
            color: white;
            font-weight: bold;
        }
        .alert {
            margin-bottom: 10px;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        .refresh-btn {
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container dashboard-container">
        <h1 class="text-center mb-4">系统监控仪表板</h1>
        
        <!-- 统计卡片 -->
        <div class="stats-grid" id="stats-grid">
            <!-- 统计卡片将通过JavaScript动态生成 -->
        </div>
        
        <!-- 系统监控 -->
        <div class="card">
            <div class="card-header">
                系统监控
            </div>
            <div class="card-body">
                <div class="chart-container">
                    <canvas id="systemChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- 网络监控 -->
        <div class="card">
            <div class="card-header">
                网络监控
            </div>
            <div class="card-body">
                <div class="chart-container">
                    <canvas id="networkChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- 任务监控 -->
        <div class="card">
            <div class="card-header">
                任务监控
            </div>
            <div class="card-body">
                <div class="chart-container">
                    <canvas id="taskChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- 告警信息 -->
        <div class="card">
            <div class="card-header">
                告警信息
            </div>
            <div class="card-body">
                <div id="alerts-container">
                    <!-- 告警信息将通过JavaScript动态生成 -->
                </div>
            </div>
        </div>
        
        <!-- 刷新按钮 -->
        <div class="text-center">
            <button class="btn btn-primary refresh-btn" onclick="refreshData()">
                刷新数据
            </button>
        </div>
    </div>
    
    <script>
        // 图表实例
        let systemChart, networkChart, taskChart;
        
        // 初始化图表
        function initCharts() {
            // 系统图表
            const systemCtx = document.getElementById('systemChart').getContext('2d');
            systemChart = new Chart(systemCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'CPU使用率 (%)',
                            data: [],
                            borderColor: 'rgb(255, 99, 132)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            tension: 0.4
                        },
                        {
                            label: '内存使用率 (%)',
                            data: [],
                            borderColor: 'rgb(54, 162, 235)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            tension: 0.4
                        },
                        {
                            label: '磁盘使用率 (%)',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    },
                    animation: {
                        duration: 0
                    }
                }
            });
            
            // 网络图表
            const networkCtx = document.getElementById('networkChart').getContext('2d');
            networkChart = new Chart(networkCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: '活跃连接数',
                            data: [],
                            borderColor: 'rgb(255, 159, 64)',
                            backgroundColor: 'rgba(255, 159, 64, 0.2)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    },
                    animation: {
                        duration: 0
                    }
                }
            });
            
            // 任务图表
            const taskCtx = document.getElementById('taskChart').getContext('2d');
            taskChart = new Chart(taskCtx, {
                type: 'bar',
                data: {
                    labels: ['总任务', '已完成', '失败', '运行中', '待处理'],
                    datasets: [
                        {
                            label: '任务数量',
                            data: [0, 0, 0, 0, 0],
                            backgroundColor: [
                                'rgba(255, 99, 132, 0.6)',
                                'rgba(75, 192, 192, 0.6)',
                                'rgba(255, 159, 64, 0.6)',
                                'rgba(54, 162, 235, 0.6)',
                                'rgba(153, 102, 255, 0.6)'
                            ],
                            borderColor: [
                                'rgb(255, 99, 132)',
                                'rgb(75, 192, 192)',
                                'rgb(255, 159, 64)',
                                'rgb(54, 162, 235)',
                                'rgb(153, 102, 255)'
                            ],
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    },
                    animation: {
                        duration: 0
                    }
                }
            });
        }
        
        // 更新统计卡片
        function updateStatsCards(stats) {
            const statsGrid = document.getElementById('stats-grid');
            statsGrid.innerHTML = '';
            
            // 系统统计
            if (stats.system) {
                const systemStats = stats.system;
                statsGrid.innerHTML += `
                    <div class="stat-card">
                        <div class="stat-value">${systemStats.average_cpu.toFixed(2)}%</div>
                        <div class="stat-label">平均CPU使用率</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${systemStats.average_memory.toFixed(2)}%</div>
                        <div class="stat-label">平均内存使用率</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${systemStats.average_disk.toFixed(2)}%</div>
                        <div class="stat-label">平均磁盘使用率</div>
                    </div>
                `;
            }
            
            // 网络统计
            if (stats.network) {
                const networkStats = stats.network;
                const requestStats = networkStats.request_stats || {};
                statsGrid.innerHTML += `
                    <div class="stat-card">
                        <div class="stat-value">${requestStats.total_requests || 0}</div>
                        <div class="stat-label">总请求数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${requestStats.successful_requests || 0}</div>
                        <div class="stat-label">成功请求数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${requestStats.failed_requests || 0}</div>
                        <div class="stat-label">失败请求数</div>
                    </div>
                `;
            }
            
            // 任务统计
            if (stats.task) {
                const taskStats = stats.task;
                statsGrid.innerHTML += `
                    <div class="stat-card">
                        <div class="stat-value">${taskStats.total_tasks || 0}</div>
                        <div class="stat-label">总任务数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${taskStats.completed_tasks || 0}</div>
                        <div class="stat-label">已完成任务</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${taskStats.failed_tasks || 0}</div>
                        <div class="stat-label">失败任务</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${(taskStats.success_rate || 0).toFixed(2)}%</div>
                        <div class="stat-label">任务成功率</div>
                    </div>
                `;
            }
        }
        
        // 更新系统图表
        function updateSystemChart(data) {
            if (!systemChart) return;
            
            const labels = data.map(d => {
                const date = new Date(d.timestamp);
                return date.toLocaleTimeString();
            });
            
            const cpuData = data.map(d => d.cpu_percent || 0);
            const memoryData = data.map(d => d.memory_percent || 0);
            const diskData = data.map(d => d.disk_percent || 0);
            
            systemChart.data.labels = labels;
            systemChart.data.datasets[0].data = cpuData;
            systemChart.data.datasets[1].data = memoryData;
            systemChart.data.datasets[2].data = diskData;
            systemChart.update();
        }
        
        // 更新网络图表
        function updateNetworkChart(data) {
            if (!networkChart) return;
            
            const labels = data.map(d => {
                const date = new Date(d.timestamp);
                return date.toLocaleTimeString();
            });
            
            const connectionsData = data.map(d => d.active_connections || 0);
            
            networkChart.data.labels = labels;
            networkChart.data.datasets[0].data = connectionsData;
            networkChart.update();
        }
        
        // 更新任务图表
        function updateTaskChart(stats) {
            if (!taskChart) return;
            
            if (stats.task) {
                const taskStats = stats.task;
                taskChart.data.datasets[0].data = [
                    taskStats.total_tasks || 0,
                    taskStats.completed_tasks || 0,
                    taskStats.failed_tasks || 0,
                    taskStats.running_tasks || 0,
                    taskStats.pending_tasks || 0
                ];
                taskChart.update();
            }
        }
        
        // 更新告警信息
        function updateAlerts(alerts) {
            const alertsContainer = document.getElementById('alerts-container');
            alertsContainer.innerHTML = '';
            
            if (alerts.length === 0) {
                alertsContainer.innerHTML = '<div class="alert alert-info">暂无告警信息</div>';
                return;
            }
            
            alerts.forEach(alert => {
                let alertClass = 'alert-info';
                if (alert.level === 'warning') {
                    alertClass = 'alert-warning';
                } else if (alert.level === 'error') {
                    alertClass = 'alert-danger';
                }
                
                alertsContainer.innerHTML += `
                    <div class="alert ${alertClass}">
                        <strong>${alert.level.toUpperCase()}</strong> - ${alert.message}
                        <br>
                        <small>${new Date(alert.timestamp).toLocaleString()}</small>
                    </div>
                `;
            });
        }
        
        // 刷新数据
        function refreshData() {
            // 获取仪表板数据
            $.ajax({
                url: '/api/dashboard',
                method: 'GET',
                success: function(response) {
                    if (response.success) {
                        const data = response.data;
                        
                        // 更新统计卡片
                        updateStatsCards(data.stats);
                        
                        // 更新系统图表
                        updateSystemChart(data.system_data);
                        
                        // 更新网络图表
                        updateNetworkChart(data.network_data);
                        
                        // 更新任务图表
                        updateTaskChart(data.stats);
                        
                        // 更新告警信息
                        updateAlerts(data.alerts);
                    }
                },
                error: function(error) {
                    console.error('获取数据失败:', error);
                }
            });
        }
        
        // 页面加载完成后初始化
        $(document).ready(function() {
            // 初始化图表
            initCharts();
            
            // 首次加载数据
            refreshData();
            
            // 自动刷新数据（每30秒）
            setInterval(refreshData, 30000);
        });
    </script>
</body>
</html>
    '''
    
    # 写入dashboard.html模板
    with open(os.path.join(templates_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    # 运行Flask应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
