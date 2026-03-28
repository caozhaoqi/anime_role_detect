#!/bin/bash

# 监控系统部署脚本
# 用于部署Prometheus、Grafana和告警规则

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Anime Role Detect 监控系统部署"
echo "=========================================="

# 检查kubectl是否可用
if ! command -v kubectl &> /dev/null; then
    echo "错误: kubectl未安装"
    echo "请先安装kubectl: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

# 检查是否连接到Kubernetes集群
if ! kubectl cluster-info &> /dev/null; then
    echo "错误: 无法连接到Kubernetes集群"
    echo "请确保已配置kubeconfig文件"
    exit 1
fi

echo "✓ Kubernetes集群连接正常"

# 创建monitoring命名空间
echo ""
echo "1. 创建monitoring命名空间..."
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# 部署Prometheus
echo ""
echo "2. 部署Prometheus..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - /etc/prometheus/rules/*.yaml
    
    alerting:
      alertmanagers:
        - static_configs:
            - targets: []
    
    scrape_configs:
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__
          - action: labelmap
            regex: __meta_kubernetes_pod_label_(.+)
          - source_labels: [__meta_kubernetes_namespace]
            action: replace
            target_label: kubernetes_namespace
          - source_labels: [__meta_kubernetes_pod_name]
            action: replace
            target_label: kubernetes_pod_name
      
      - job_name: 'kubernetes-nodes'
        kubernetes_sd_configs:
          - role: node
        relabel_configs:
          - action: labelmap
            regex: __meta_kubernetes_node_label_(.+)
      
      - job_name: 'kubernetes-services'
        kubernetes_sd_configs:
          - role: service
        metrics_path: /probe
        params:
          module: [http_2xx]
        relabel_configs:
          - source_labels: [__address__]
            target_label: __param_target
          - source_labels: [__param_target]
            target_label: instance
          - target_label: __address__
            replacement: blackbox-exporter:9115
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:v2.30.0
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        - name: prometheus-rules
          mountPath: /etc/prometheus/rules
        - name: prometheus-storage
          mountPath: /prometheus
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
      - name: prometheus-rules
        configMap:
          name: prometheus-rules
      - name: prometheus-storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP
EOF

# 部署告警规则
echo ""
echo "3. 部署告警规则..."
kubectl create configmap prometheus-rules --from-file="$SCRIPT_DIR/prometheus-alerts.yaml" -n monitoring --dry-run=client -o yaml | kubectl apply -f -

# 部署Grafana
echo ""
echo "4. 部署Grafana..."
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:8.2.0
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: admin
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
      volumes:
      - name: grafana-storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: monitoring
spec:
  selector:
    app: grafana
  ports:
  - port: 3000
    targetPort: 3000
  type: ClusterIP
EOF

# 等待Pod启动
echo ""
echo "5. 等待Pod启动..."
kubectl wait --for=condition=ready pod -l app=prometheus -n monitoring --timeout=300s
kubectl wait --for=condition=ready pod -l app=grafana -n monitoring --timeout=300s

# 导入Grafana仪表板
echo ""
echo "6. 导入Grafana仪表板..."
echo "请手动导入仪表板配置文件: $SCRIPT_DIR/grafana-dashboard.json"
echo "Grafana访问地址: http://grafana.monitoring.svc.cluster.local:3000"
echo "默认用户名: admin"
echo "默认密码: admin"

# 显示部署状态
echo ""
echo "=========================================="
echo "监控系统部署完成！"
echo "=========================================="
echo ""
echo "Prometheus: http://prometheus.monitoring.svc.cluster.local:9090"
echo "Grafana: http://grafana.monitoring.svc.cluster.local:3000"
echo ""
echo "查看监控状态:"
echo "  kubectl get pods -n monitoring"
echo "  kubectl get services -n monitoring"
echo ""
echo "查看Prometheus日志:"
echo "  kubectl logs -f deployment/prometheus -n monitoring"
echo ""
echo "查看Grafana日志:"
echo "  kubectl logs -f deployment/grafana -n monitoring"
echo ""
echo "端口转发（本地访问）:"
echo "  kubectl port-forward svc/prometheus 9090:9090 -n monitoring"
echo "  kubectl port-forward svc/grafana 3000:3000 -n monitoring"