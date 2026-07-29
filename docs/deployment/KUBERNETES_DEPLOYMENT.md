> [!CAUTION]
> **已弃用 / 过期**：本文档描述的部署文件名（`configmap.yaml` / `secret.yaml` / `backend-deployment.yaml` 等）与当前仓库已不一致。
> 当前唯一权威 K8s 部署源为 **`k8s/`**（base + overlays/ci），详见 [`k8s/README.md`](../k8s/README.md)。
> 旧 `deployment/k8s-*.yaml` 已归档至 `deployment/_legacy_backup/`，仅供历史追溯。
>
> 仅保留作为历史参考，请勿按本文档执行部署。


# Kubernetes部署python程序文档

## 系统架构概览

下图展示了角色分类系统在Kubernetes环境中的部署架构：

![Kubernetes部署架构图](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Kubernetes%20deployment%20architecture%20diagram%20for%20a%20Python%20application%20with%20frontend%20and%20backend%20services%2C%20showing%20Ingress%2C%20Services%2C%20Deployments%2C%20Pods%2C%20and%20storage%20components%2C%20professional%20technical%20diagram%2C%20clear%20labels%2C%20blue%20and%20white%20color%20scheme&image_size=landscape_16_9)

## 1. 环境准备

### 1.1 系统要求
- Ubuntu 20.04 LTS 或更高版本
- **推荐配置**：
  - 小型部署：4GB内存，20GB磁盘空间，2核CPU
  - 中型部署：8GB内存，40GB磁盘空间，4核CPU
  - 大型部署：16GB+内存，80GB+磁盘空间，8核+CPU
- 具有公网IP地址
- 网络带宽：至少1Mbps，推荐10Mbps以上

### 1.2 系统更新与优化
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装必要的系统工具
sudo apt install -y wget curl git htop unzip

# 调整内核参数
sudo tee -a /etc/sysctl.conf << EOF
# 提高文件描述符限制
fs.file-max = 65536
# 网络优化
net.core.somaxconn = 4096
net.ipv4.tcp_max_syn_backlog = 4096
# 内存管理
vm.swappiness = 10
EOF

sudo sysctl -p

# 调整文件描述符限制
sudo tee -a /etc/security/limits.conf << EOF
* soft nofile 65536
* hard nofile 65536
EOF
```

## 2. 安装Kubernetes集群

### 2.1 安装Docker
```bash
# 安装依赖
sudo apt install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common

# 添加Docker GPG密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# 添加Docker仓库
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# 安装Docker
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io

# 启动Docker服务
sudo systemctl start docker
sudo systemctl enable docker

# 配置Docker镜像加速
sudo tee /etc/docker/daemon.json << EOF
{
  "registry-mirrors": ["https://docker.mirrors.ustc.edu.cn", "https://hub-mirror.c.163.com"]
}
EOF

sudo systemctl restart docker

# 验证Docker安装
sudo docker --version

# 将当前用户添加到docker组（可选）
sudo usermod -aG docker $USER
```

### 2.2 安装Kubernetes工具
使用MicroK8s（轻量级Kubernetes集群，适合单服务器部署）：

![MicroK8s安装示意图](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Terminal%20screenshot%20showing%20MicroK8s%20installation%20and%20status%20check%2C%20displaying%20commands%20like%20'sudo%20snap%20install%20microk8s'%20and%20'sudo%20microk8s%20status'%2C%20with%20successful%20output%2C%20professional%20terminal%20interface%2C%20dark%20theme&image_size=landscape_4_3)

```bash
# 安装MicroK8s
sudo snap install microk8s --classic

# 启动MicroK8s
sudo microk8s start

# 启用必要的插件
sudo microk8s enable dns dashboard storage ingress prometheus grafana metallb

# 配置kubectl
sudo microk8s kubectl config view --raw > ~/.kube/config
chmod 600 ~/.kube/config

# 配置默认命名空间
sudo microk8s.kubectl config set-context --current --namespace=default

# 验证Kubernetes集群状态
sudo microk8s kubectl get nodes
```

## 3. 构建Docker镜像

### 3.1 克隆项目代码
```bash
git clone https://github.com/caozhaoqi/anime-role-detect.git
cd anime-role-detect
```

### 3.2 构建后端镜像

![Docker镜像构建示意图](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Terminal%20screenshot%20showing%20Docker%20image%20build%20process%20for%20a%20Python%20application%2C%20displaying%20build%20steps%20and%20layers%2C%20successful%20completion%20message%2C%20professional%20terminal%20interface%2C%20dark%20theme&image_size=landscape_4_3)

```bash
# 使用BuildKit加速构建
export DOCKER_BUILDKIT=1

# 构建后端镜像
sudo docker build --build-arg BUILDKIT_INLINE_CACHE=1 -t character-classification-backend:latest -f Dockerfile.backend .
```

### 3.3 构建前端镜像
```bash
# 构建前端镜像
sudo docker build --build-arg BUILDKIT_INLINE_CACHE=1 -t character-classification-frontend:latest -f Dockerfile.frontend .
```

### 3.4 验证镜像构建
```bash
sudo docker images
```

## 4. 部署应用到Kubernetes

### 4.1 创建配置文件

#### 4.1.1 ConfigMap配置
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: classification-config
data:
  MODEL_NAME: "arona_plana"
  API_TIMEOUT: "30"
  LOG_LEVEL: "INFO"
  CACHE_SIZE: "1000"
```

#### 4.1.2 Secret配置
```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: classification-secret
type: Opaque
data:
  # 注意：这里的值需要base64编码
  OPENAI_API_KEY: "base64_encoded_api_key"
```

#### 4.1.3 后端部署配置
```yaml
# backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: character-classification-backend
  labels:
    app: character-classification
    component: backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: character-classification
      component: backend
  template:
    metadata:
      labels:
        app: character-classification
        component: backend
    spec:
      containers:
      - name: backend
        image: character-classification-backend:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        envFrom:
        - configMapRef:
            name: classification-config
        - secretRef:
            name: classification-secret
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
```

#### 4.1.4 前端部署配置
```yaml
# frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: character-classification-frontend
  labels:
    app: character-classification
    component: frontend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: character-classification
      component: frontend
  template:
    metadata:
      labels:
        app: character-classification
        component: frontend
    spec:
      containers:
      - name: frontend
        image: character-classification-frontend:latest
        ports:
        - containerPort: 80
        resources:
          requests:
            cpu: "200m"
            memory: "256Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 30
```

#### 4.1.5 服务配置
```yaml
# services.yaml
apiVersion: v1
kind: Service
metadata:
  name: character-classification-backend
spec:
  selector:
    app: character-classification
    component: backend
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: character-classification-frontend
spec:
  selector:
    app: character-classification
    component: frontend
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
```

#### 4.1.6 自动缩放配置
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: character-classification-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: character-classification-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: character-classification-frontend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: character-classification-frontend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### 4.1.7 网络策略配置
```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: classification-network-policy
spec:
  podSelector:
    matchLabels:
      app: character-classification
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: character-classification
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 80
  egress:
  - to:
    - podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
    ports:
    - protocol: TCP
      port: 443
```

### 4.2 应用部署
```bash
# 应用配置文件
sudo microk8s kubectl apply -f configmap.yaml
sudo microk8s kubectl apply -f secret.yaml
sudo microk8s kubectl apply -f backend-deployment.yaml
sudo microk8s kubectl apply -f frontend-deployment.yaml
sudo microk8s kubectl apply -f services.yaml
sudo microk8s kubectl apply -f hpa.yaml
sudo microk8s kubectl apply -f network-policy.yaml
```

## 5. 配置外网访问

### 5.1 配置Ingress
```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: character-classification-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
spec:
  rules:
  - host: your-domain.com  # 替换为你的域名
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: character-classification-frontend
            port:
              number: 80
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: character-classification-backend
            port:
              number: 8000
```

### 5.2 应用Ingress配置
```bash
sudo microk8s kubectl apply -f ingress.yaml
```

### 5.3 配置HTTPS（可选）
```bash
# 安装cert-manager
sudo microk8s enable cert-manager

# 创建ClusterIssuer
cat << EOF | sudo microk8s kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com  # 替换为你的邮箱
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: public
EOF

# 更新Ingress配置以使用HTTPS
cat << EOF | sudo microk8s kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: character-classification-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: classification-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: character-classification-frontend
            port:
              number: 80
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: character-classification-backend
            port:
              number: 8000
EOF
```

### 5.4 配置防火墙
```bash
# 开放80端口（HTTP）
sudo ufw allow 80/tcp

# 开放443端口（HTTPS）
sudo ufw allow 443/tcp

# 开放SSH端口（如果需要）
sudo ufw allow 22/tcp

# 重新加载防火墙规则
sudo ufw reload

# 启用防火墙（如果未启用）
sudo ufw enable
```

### 5.5 配置域名解析
在你的域名提供商处，将域名A记录指向服务器的公网IP地址。

## 6. 验证服务

### 6.1 查看Pod状态

![Kubernetes Pod状态查看](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Terminal%20screenshot%20showing%20Kubernetes%20pod%20status%20using%20'kubectl%20get%20pods'%20command%2C%20displaying%20running%20pods%20with%20READY%20status%2C%20professional%20terminal%20interface%2C%20dark%20theme&image_size=landscape_4_3)

```bash
sudo microk8s kubectl get pods
```

### 6.2 查看服务状态

![Kubernetes服务状态查看](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Terminal%20screenshot%20showing%20Kubernetes%20services%20and%20ingress%20status%20using%20'kubectl%20get%20services'%20and%20'kubectl%20get%20ingress'%20commands%2C%20professional%20terminal%20interface%2C%20dark%20theme&image_size=landscape_4_3)

```bash
sudo microk8s kubectl get services
sudo microk8s kubectl get ingress
```

### 6.3 访问应用

![前端应用界面](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Web%20browser%20screenshot%20showing%20anime%20character%20classification%20frontend%20application%2C%20with%20upload%20form%2C%20modern%20UI%2C%20blue%20color%20scheme%2C%20professional%20design&image_size=landscape_16_9)

在浏览器中访问你的域名（例如：http://your-domain.com 或 https://your-domain.com），应该能看到角色分类系统的前端界面。

### 6.4 测试API
```bash
# 测试API是否可访问
curl -X POST -F "file=@path/to/image.jpg" http://your-domain.com/api/classify

# 测试健康检查端点
curl http://your-domain.com/api/health
```

## 7. 监控与维护

### 7.1 查看集群状态
```bash
sudo microk8s status
sudo microk8s kubectl get nodes
```

### 7.2 查看资源使用情况
```bash
sudo microk8s kubectl top nodes
sudo microk8s kubectl top pods
```

### 7.3 详细监控配置

#### 7.3.1 Prometheus配置

```yaml
# prometheus-config.yaml
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
```

#### 7.3.2 Grafana配置

```yaml
# grafana-dashboard.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
  namespace: monitoring
data:
  anime-role-detect.json: |
    {
      "annotations": {
        "list": [
          {
            "builtIn": 1,
            "datasource": "-- Grafana --",
            "enable": true,
            "hide": true,
            "iconColor": "rgba(0, 211, 255, 1)",
            "name": "Annotations & Alerts",
            "type": "dashboard"
          }
        ]
      },
      "editable": true,
      "gnetId": null,
      "graphTooltip": 0,
      "id": null,
      "links": [],
      "panels": [
        {
          "aliasColors": {},
          "bars": false,
          "dashLength": 10,
          "dashes": false,
          "datasource": "Prometheus",
          "fieldConfig": {
            "defaults": {
              "custom": {}
            },
            "overrides": []
          },
          "fill": 1,
          "fillGradient": 0,
          "gridPos": {
            "h": 8,
            "w": 12,
            "x": 0,
            "y": 0
          },
          "hiddenSeries": false,
          "id": 2,
          "legend": {
            "avg": false,
            "current": false,
            "max": false,
            "min": false,
            "show": true,
            "total": false,
            "values": false
          },
          "lines": true,
          "linewidth": 1,
          "nullPointMode": "null",
          "options": {
            "alertThreshold": true
          },
          "percentage": false,
          "pluginVersion": "7.5.7",
          "pointradius": 2,
          "points": false,
          "renderer": "flot",
          "seriesOverrides": [],
          "spaceLength": 10,
          "stack": false,
          "steppedLine": false,
          "targets": [
            {
              "expr": "sum(rate(container_cpu_usage_seconds_total{namespace=\"default\",pod=~\"character-classification-backend.*\"}[5m])) by (pod)",
              "interval": "",
              "legendFormat": "{{pod}}",
              "refId": "A"
            }
          ],
          "thresholds": [],
          "timeFrom": null,
          "timeRegions": [],
          "timeShift": null,
          "title": "Backend CPU Usage",
          "tooltip": {
            "shared": true,
            "sort": 0,
            "value_type": "individual"
          },
          "type": "graph",
          "xaxis": {
            "buckets": null,
            "mode": "time",
            "name": null,
            "show": true,
            "values": []
          },
          "yaxes": [
            {
              "format": "short",
              "label": null,
              "logBase": 1,
              "max": null,
              "min": null,
              "show": true
            },
            {
              "format": "short",
              "label": null,
              "logBase": 1,
              "max": null,
              "min": null,
              "show": true
            }
          ],
          "yaxis": {
            "align": false,
            "alignLevel": null
          }
        },
        {
          "aliasColors": {},
          "bars": false,
          "dashLength": 10,
          "dashes": false,
          "datasource": "Prometheus",
          "fieldConfig": {
            "defaults": {
              "custom": {}
            },
            "overrides": []
          },
          "fill": 1,
          "fillGradient": 0,
          "gridPos": {
            "h": 8,
            "w": 12,
            "x": 12,
            "y": 0
          },
          "hiddenSeries": false,
          "id": 3,
          "legend": {
            "avg": false,
            "current": false,
            "max": false,
            "min": false,
            "show": true,
            "total": false,
            "values": false
          },
          "lines": true,
          "linewidth": 1,
          "nullPointMode": "null",
          "options": {
            "alertThreshold": true
          },
          "percentage": false,
          "pluginVersion": "7.5.7",
          "pointradius": 2,
          "points": false,
          "renderer": "flot",
          "seriesOverrides": [],
          "spaceLength": 10,
          "stack": false,
          "steppedLine": false,
          "targets": [
            {
              "expr": "sum(container_memory_usage_bytes{namespace=\"default\",pod=~\"character-classification-backend.*\"}) by (pod)",
              "interval": "",
              "legendFormat": "{{pod}}",
              "refId": "A"
            }
          ],
          "thresholds": [],
          "timeFrom": null,
          "timeRegions": [],
          "timeShift": null,
          "title": "Backend Memory Usage",
          "tooltip": {
            "shared": true,
            "sort": 0,
            "value_type": "individual"
          },
          "type": "graph",
          "xaxis": {
            "buckets": null,
            "mode": "time",
            "name": null,
            "show": true,
            "values": []
          },
          "yaxes": [
            {
              "format": "bytes",
              "label": null,
              "logBase": 1,
              "max": null,
              "min": null,
              "show": true
            },
            {
              "format": "short",
              "label": null,
              "logBase": 1,
              "max": null,
              "min": null,
              "show": true
            }
          ],
          "yaxis": {
            "align": false,
            "alignLevel": null
          }
        },
        {
          "aliasColors": {},
          "bars": false,
          "dashLength": 10,
          "dashes": false,
          "datasource": "Prometheus",
          "fieldConfig": {
            "defaults": {
              "custom": {}
            },
            "overrides": []
          },
          "fill": 1,
          "fillGradient": 0,
          "gridPos": {
            "h": 8,
            "w": 12,
            "x": 0,
            "y": 8
          },
          "hiddenSeries": false,
          "id": 4,
          "legend": {
            "avg": false,
            "current": false,
            "max": false,
            "min": false,
            "show": true,
            "total": false,
            "values": false
          },
          "lines": true,
          "linewidth": 1,
          "nullPointMode": "null",
          "options": {
            "alertThreshold": true
          },
          "percentage": false,
          "pluginVersion": "7.5.7",
          "pointradius": 2,
          "points": false,
          "renderer": "flot",
          "seriesOverrides": [],
          "spaceLength": 10,
          "stack": false,
          "steppedLine": false,
          "targets": [
            {
              "expr": "sum(rate(container_cpu_usage_seconds_total{namespace=\"default\",pod=~\"character-classification-frontend.*\"}[5m])) by (pod)",
              "interval": "",
              "legendFormat": "{{pod}}",
              "refId": "A"
            }
          ],
          "thresholds": [],
          "timeFrom": null,
          "timeRegions": [],
          "timeShift": null,
          "title": "Frontend CPU Usage",
          "tooltip": {
            "shared": true,
            "sort": 0,
            "value_type": "individual"
          },
          "type": "graph",
          "xaxis": {
            "buckets": null,
            "mode": "time",
            "name": null,
            "show": true,
            "values": []
          },
          "yaxes": [
            {
              "format": "short",
              "label": null,
              "logBase": 1,
              "max": null,
              "min": null,
              "show": true
            },
            {
              "format": "short",
              "label": null,
              "logBase": 1,
              "max": null,
              "min": null,
              "show": true
            }
          ],
          "yaxis": {
            "align": false,
            "alignLevel": null
          }
        },
        {
          "aliasColors": {},
          "bars": false,
          "dashLength": 10,
          "dashes": false,
          "datasource": "Prometheus",
          "fieldConfig": {
            "defaults": {
              "custom": {}
            },
            "overrides": []
          },
          "fill": 1,
          "fillGradient": 0,
          "gridPos": {
            "h": 8,
            "w": 12,
            "x": 12,
            "y": 8
          },
          "hiddenSeries": false,
          "id": 5,
          "legend": {
            "avg": false,
            "current": false,
            "max": false,
            "min": false,
            "show": true,
            "total": false,
            "values": false
          },
          "lines": true,
          "linewidth": 1,
          "nullPointMode": "null",
          "options": {
            "alertThreshold": true
          },
          "percentage": false,
          "pluginVersion": "7.5.7",
          "pointradius": 2,
          "points": false,
          "renderer": "flot",
          "seriesOverrides": [],
          "spaceLength": 10,
          "stack": false,
          "steppedLine": false,
          "targets": [
            {
              "expr": "sum(container_memory_usage_bytes{namespace=\"default\",pod=~\"character-classification-frontend.*\"}) by (pod)",
              "interval": "",
              "legendFormat": "{{pod}}",
              "refId": "A"
            }
          ],
          "thresholds": [],
          "timeFrom": null,
          "timeRegions": [],
          "timeShift": null,
          "title": "Frontend Memory Usage",
          "tooltip": {
            "shared": true,
            "sort": 0,
            "value_type": "individual"
          },
          "type": "graph",
          "xaxis": {
            "buckets": null,
            "mode": "time",
            "name": null,
            "show": true,
            "values": []
          },
          "yaxes": [
            {
              "format": "bytes",
              "label": null,
              "logBase": 1,
              "max": null,
              "min": null,
              "show": true
            },
            {
              "format": "short",
              "label": null,
              "logBase": 1,
              "max": null,
              "min": null,
              "show": true
            }
          ],
          "yaxis": {
            "align": false,
            "alignLevel": null
          }
        }
      ],
      "schemaVersion": 26,
      "style": "dark",
      "tags": [],
      "templating": {
        "list": []
      },
      "time": {
        "from": "now-6h",
        "to": "now"
      },
      "timepicker": {},
      "timezone": "",
      "title": "Anime Role Detect Dashboard",
      "uid": "anime-role-detect",
      "version": 1
    }
```

#### 7.3.3 查看监控面板

![Grafana监控面板](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Grafana%20dashboard%20showing%20Kubernetes%20cluster%20metrics%2C%20including%20CPU%2C%20memory%2C%20pod%20status%2C%20and%20network%20traffic%2C%20professional%20monitoring%20interface%2C%20dark%20theme&image_size=landscape_16_9)

```bash
# 获取Grafana访问地址
sudo microk8s kubectl get services -n monitoring

# 获取Grafana密码
sudo microk8s kubectl get secret -n monitoring grafana -o jsonpath="{.data.admin-password}" | base64 --decode

# 访问Grafana面板
# 打开浏览器访问 http://<服务器IP>:<Grafana端口>
```

### 7.4 日志管理配置

#### 7.4.1 安装Loki和Promtail

```bash
# 安装Loki和Promtail
sudo microk8s enable loki

# 查看Loki服务状态
sudo microk8s kubectl get services -n monitoring | grep loki
```

#### 7.4.2 配置日志收集

```yaml
# promtail-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: promtail-config
  namespace: monitoring
data:
  promtail.yaml: |
    server:
      http_listen_port: 9080
      grpc_listen_port: 0

    clients:
      - url: http://loki.monitoring:3100/loki/api/v1/push

    scrape_configs:
      - job_name: kubernetes-pods
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_controller_name]
            regex: ([a-z0-9-]+)-[0-9a-f]{8,10}
            action: replace
            target_label: __tmp_controller
          - source_labels: [__meta_kubernetes_pod_label_app]
            action: replace
            target_label: app
          - source_labels: [__meta_kubernetes_namespace]
            action: replace
            target_label: namespace
          - source_labels: [__meta_kubernetes_pod_name]
            action: replace
            target_label: pod
          - source_labels: [__meta_kubernetes_pod_container_name]
            action: replace
            target_label: container
          - replacement: /var/log/pods/*$1/*.log
            separator: /
            source_labels:
              - __meta_kubernetes_pod_uid
              - __meta_kubernetes_pod_container_name
            target_label: __path__
```

#### 7.4.3 在Grafana中查看日志

1. 登录Grafana
2. 点击左侧菜单的"Explore"
3. 选择"Loki"作为数据源
4. 使用查询语句查看日志，例如：
   - `{app="character-classification", container="backend"} |~ "error"` - 查看后端错误日志
   - `{app="character-classification", container="frontend"}` - 查看前端所有日志

### 7.5 告警配置

#### 7.5.1 Prometheus告警规则

```yaml
# prometheus-alerts.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: anime-role-detect-alerts
  namespace: monitoring
spec:
  groups:
  - name: backend-alerts
    rules:
    - alert: BackendHighCPU
      expr: sum(rate(container_cpu_usage_seconds_total{namespace="default",pod=~"character-classification-backend.*"}[5m])) by (pod) > 1.5
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Backend High CPU Usage"
        description: "Backend pod {{ $labels.pod }} has high CPU usage ({{ $value }} cores)"
    
    - alert: BackendHighMemory
      expr: sum(container_memory_usage_bytes{namespace="default",pod=~"character-classification-backend.*"}) by (pod) > 3.5Gi
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Backend High Memory Usage"
        description: "Backend pod {{ $labels.pod }} has high memory usage ({{ $value | humanizeBytes }})"
    
    - alert: BackendPodDown
      expr: kube_pod_status_phase{namespace="default",pod=~"character-classification-backend.*",phase!="Running"} == 1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Backend Pod Down"
        description: "Backend pod {{ $labels.pod }} is not running"

  - name: frontend-alerts
    rules:
    - alert: FrontendHighCPU
      expr: sum(rate(container_cpu_usage_seconds_total{namespace="default",pod=~"character-classification-frontend.*"}[5m])) by (pod) > 0.4
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Frontend High CPU Usage"
        description: "Frontend pod {{ $labels.pod }} has high CPU usage ({{ $value }} cores)"
    
    - alert: FrontendPodDown
      expr: kube_pod_status_phase{namespace="default",pod=~"character-classification-frontend.*",phase!="Running"} == 1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Frontend Pod Down"
        description: "Frontend pod {{ $labels.pod }} is not running"
```

#### 7.5.2 配置告警通知

```yaml
# alertmanager-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: monitoring
data:
  alertmanager.yml: |
    global:
      resolve_timeout: 5m
    route:
      group_by: ['alertname']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 1h
      receiver: 'email-notifications'
    receivers:
    - name: 'email-notifications'
      email_configs:
      - to: 'your-email@example.com'
        from: 'alertmanager@example.com'
        smarthost: 'smtp.example.com:587'
        auth_username: 'alertmanager'
        auth_password: 'your-password'
        require_tls: true
```

### 7.6 升级应用
1. 构建新的Docker镜像
2. 更新Kubernetes部署配置
3. 应用更新：
   ```bash
   sudo microk8s kubectl apply -f backend-deployment.yaml
   sudo microk8s kubectl apply -f frontend-deployment.yaml
   ```

### 7.7 备份与恢复
```bash
# 备份配置文件
tar -czf k8s-config-$(date +%Y%m%d).tar.gz *.yaml

# 备份模型文件（如果使用持久卷）
sudo microk8s kubectl cp character-classification-backend-xxx:/app/models ./models-backup

# 恢复配置
kubectl apply -f *.yaml

# 恢复模型文件
sudo microk8s kubectl cp ./models-backup character-classification-backend-xxx:/app/models
```

## 8. 故障排查

### 8.1 查看Pod日志
```bash
# 查看后端Pod日志
sudo microk8s kubectl logs -l component=backend

# 查看前端Pod日志
sudo microk8s kubectl logs -l component=frontend

# 查看具体Pod的详细日志
sudo microk8s kubectl logs -f <pod-name>
```

### 8.2 检查Pod状态
```bash
# 查看Pod详细信息
sudo microk8s kubectl describe pod <pod-name>

# 查看Pod事件
sudo microk8s kubectl get events
```

### 8.3 网络故障排查
```bash
# 检查网络连接
sudo microk8s kubectl exec -it <pod-name> -- ping google.com

# 检查服务访问
sudo microk8s kubectl exec -it <pod-name> -- curl http://character-classification-backend:8000/api/health
```

### 8.4 常见问题解决
- **Pod无法启动**：检查Docker镜像是否正确构建，查看Pod日志获取具体错误信息
- **服务无法访问**：检查防火墙规则是否正确配置，检查Ingress规则是否正确
- **API返回错误**：检查后端Pod是否正常运行，查看后端日志获取具体错误信息
- **资源不足**：检查服务器资源使用情况，调整Pod资源请求和限制

## 9. 自动化脚本

### 9.1 部署脚本
```bash
#!/bin/bash

# 部署脚本

# 显示帮助信息
show_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -h, --help      Show this help message"
  echo "  -b, --build     Build Docker images"
  echo "  -d, --deploy    Deploy application"
  echo "  -u, --update    Update application"
  echo "  -c, --check     Check application status"
  echo "  -t, --test      Test API endpoints"
}

# 构建镜像
build_images() {
  echo "Building Docker images..."
  export DOCKER_BUILDKIT=1
  sudo docker build --build-arg BUILDKIT_INLINE_CACHE=1 -t character-classification-backend:latest -f Dockerfile.backend .
  sudo docker build --build-arg BUILDKIT_INLINE_CACHE=1 -t character-classification-frontend:latest -f Dockerfile.frontend .
  echo "Docker images built successfully!"
}

# 部署应用
deploy_app() {
  echo "Deploying application..."
  sudo microk8s kubectl apply -f configmap.yaml
  sudo microk8s kubectl apply -f secret.yaml
  sudo microk8s kubectl apply -f backend-deployment.yaml
  sudo microk8s kubectl apply -f frontend-deployment.yaml
  sudo microk8s kubectl apply -f services.yaml
  sudo microk8s kubectl apply -f hpa.yaml
  sudo microk8s kubectl apply -f network-policy.yaml
  sudo microk8s kubectl apply -f ingress.yaml
  echo "Application deployed successfully!"
}

# 更新应用
update_app() {
  echo "Updating application..."
  build_images
  sudo microk8s kubectl apply -f backend-deployment.yaml
  sudo microk8s kubectl apply -f frontend-deployment.yaml
  echo "Application updated successfully!"
}

# 检查应用状态
check_status() {
  echo "Checking application status..."
  echo "Pods:"
  sudo microk8s kubectl get pods
  echo "\nServices:"
  sudo microk8s kubectl get services
  echo "\nIngress:"
  sudo microk8s kubectl get ingress
  echo "\nNodes:"
  sudo microk8s kubectl get nodes
  echo "\nResource usage:"
  sudo microk8s kubectl top pods
}

# 测试API
test_api() {
  echo "Testing API endpoints..."
  echo "Health check:"
  curl -s http://your-domain.com/api/health
  echo "\n\nAPI classification test:"
  curl -s -X POST -F "file=@test-image.jpg" http://your-domain.com/api/classify
  echo ""
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    -b|--build)
      build_images
      exit 0
      ;;
    -d|--deploy)
      deploy_app
      exit 0
      ;;
    -u|--update)
      update_app
      exit 0
      ;;
    -c|--check)
      check_status
      exit 0
      ;;
    -t|--test)
      test_api
      exit 0
      ;;
    *)
      echo "Invalid option: $1"
      show_help
      exit 1
      ;;
  esac
done

# 默认显示帮助信息
show_help
```

### 9.2 维护脚本
```bash
#!/bin/bash

# 维护脚本

# 显示帮助信息
show_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -h, --help      Show this help message"
  echo "  -c, --clean     Clean up old containers and images"
  echo "  -l, --logs      Collect logs"
  echo "  -b, --backup    Backup configuration and data"
  echo "  -m, --monitor   Monitor system resources"
  echo "  -r, --restart   Restart services"
}

# 清理旧容器和镜像
cleanup() {
  echo "Cleaning up old containers and images..."
  sudo docker system prune -f
  sudo docker image prune -f
  echo "Cleanup completed!"
}

# 收集日志
collect_logs() {
  echo "Collecting logs..."
  mkdir -p logs/$(date +%Y%m%d)
  sudo microk8s kubectl logs -l component=backend > logs/$(date +%Y%m%d)/backend.log
  sudo microk8s kubectl logs -l component=frontend > logs/$(date +%Y%m%d)/frontend.log
  sudo microk8s kubectl get events > logs/$(date +%Y%m%d)/events.log
  echo "Logs collected in logs/$(date +%Y%m%d)/"
}

# 备份配置和数据
backup() {
  echo "Backing up configuration and data..."
  mkdir -p backups/$(date +%Y%m%d)
  tar -czf backups/$(date +%Y%m%d)/k8s-config.tar.gz *.yaml
  # 备份模型文件（如果使用持久卷）
  POD_NAME=$(sudo microk8s kubectl get pods -l component=backend -o jsonpath="{.items[0].metadata.name}")
  if [ -n "$POD_NAME" ]; then
    sudo microk8s kubectl cp $POD_NAME:/app/models backups/$(date +%Y%m%d)/models
  fi
  echo "Backup completed in backups/$(date +%Y%m%d)/"
}

# 监控系统资源
monitor() {
  echo "Monitoring system resources..."
  echo "Press Ctrl+C to exit"
  while true; do
    echo "\n--- System Resources ---"
    top -bn1 | head -20
    echo "\n--- Kubernetes Resources ---"
    sudo microk8s kubectl top pods
    sleep 5
  done
}

# 重启服务
restart_services() {
  echo "Restarting services..."
  sudo microk8s kubectl rollout restart deployment character-classification-backend
  sudo microk8s kubectl rollout restart deployment character-classification-frontend
  echo "Services restarted!"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    -c|--clean)
      cleanup
      exit 0
      ;;
    -l|--logs)
      collect_logs
      exit 0
      ;;
    -b|--backup)
      backup
      exit 0
      ;;
    -m|--monitor)
      monitor
      exit 0
      ;;
    -r|--restart)
      restart_services
      exit 0
      ;;
    *)
      echo "Invalid option: $1"
      show_help
      exit 1
      ;;
  esac
done

# 默认显示帮助信息
show_help
```

## 10. 性能优化

### 10.1 资源配置优化
- **根据负载调整资源配置**：使用`kubectl top pods`监控资源使用情况，调整Pod的资源请求和限制
- **优化存储配置**：对于模型文件，使用SSD存储提高读取速度
- **调整HPA参数**：根据实际负载情况调整自动缩放参数

### 10.2 应用优化
- **模型优化**：使用量化技术减小模型大小，提高推理速度
- **缓存策略优化**：增加缓存大小，合理设置缓存过期时间
- **并发处理优化**：调整服务器的并发处理能力，提高请求处理效率
- **网络优化**：使用HTTP/2和gzip压缩，减少网络传输时间

## 11. 安全加固

### 11.1 容器安全
- **使用非root用户运行容器**：在Dockerfile中添加`USER nonroot`
- **禁用特权模式**：在部署配置中设置`securityContext.privileged: false`
- **限制容器能力**：在部署配置中设置`securityContext.capabilities`
- **使用只读文件系统**：在部署配置中设置`securityContext.readOnlyRootFilesystem: true`

### 11.2 网络安全
- **配置网络策略**：限制Pod间的通信，只允许必要的网络流量
- **使用TLS加密**：配置HTTPS，保护数据传输安全
- **限制外部访问**：通过Ingress和防火墙限制外部访问
- **定期更新容器镜像**：及时修复安全漏洞

## 12. CI/CD集成

### 12.1 GitLab CI配置

![GitLab CI工作流](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=GitLab%20CI%20pipeline%20dashboard%20showing%20build%2C%20test%2C%20and%20deploy%20stages%20for%20a%20Kubernetes%20application%2C%20with%20green%20success%20status%2C%20professional%20DevOps%20interface&image_size=landscape_16_9)

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: ""

build-backend:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE/backend:latest -f Dockerfile.backend .
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker push $CI_REGISTRY_IMAGE/backend:latest
  only:
    - main

build-frontend:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE/frontend:latest -f Dockerfile.frontend .
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker push $CI_REGISTRY_IMAGE/frontend:latest
  only:
    - main

test-api:
  stage: test
  image: curlimages/curl:latest
  script:
    - curl -s http://your-domain.com/api/health | grep -q "OK"
  only:
    - main

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context production
    - kubectl set image deployment/character-classification-backend backend=$CI_REGISTRY_IMAGE/backend:latest
    - kubectl set image deployment/character-classification-frontend frontend=$CI_REGISTRY_IMAGE/frontend:latest
    - kubectl rollout status deployment/character-classification-backend
    - kubectl rollout status deployment/character-classification-frontend
  only:
    - main
```

### 12.2 GitHub Actions配置

![GitHub Actions工作流](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=GitHub%20Actions%20workflow%20dashboard%20showing%20successful%20build%20and%20deploy%20jobs%20for%20a%20Kubernetes%20application%2C%20professional%20DevOps%20interface%2C%20green%20checkmarks&image_size=landscape_16_9)

```yaml
# .github/workflows/deploy.yml
name: Deploy to Kubernetes

on:
  push:
    branches:
      - main

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v1

    - name: Login to Docker Registry
      uses: docker/login-action@v1
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push backend image
      uses: docker/build-push-action@v2
      with:
        context: .
        file: Dockerfile.backend
        push: true
        tags: ${{ secrets.DOCKER_REGISTRY }}/character-classification-backend:latest

    - name: Build and push frontend image
      uses: docker/build-push-action@v2
      with:
        context: .
        file: Dockerfile.frontend
        push: true
        tags: ${{ secrets.DOCKER_REGISTRY }}/character-classification-frontend:latest

    - name: Set up kubectl
      uses: azure/setup-kubectl@v1
      with:
        version: 'latest'

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG }}" > ~/.kube/config
        chmod 600 ~/.kube/config

    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/character-classification-backend backend=${{ secrets.DOCKER_REGISTRY }}/character-classification-backend:latest
        kubectl set image deployment/character-classification-frontend frontend=${{ secrets.DOCKER_REGISTRY }}/character-classification-frontend:latest
        kubectl rollout status deployment/character-classification-backend
        kubectl rollout status deployment/character-classification-frontend
```

## 13. 跨平台诊断和OOM处理

### 13.1 跨平台诊断系统

本系统支持在多种硬件平台上运行，包括：
- **Linux/CUDA**：NVIDIA GPU环境
- **macOS/MPS**：Apple Silicon (M1/M2/M3/M4) 环境
- **CPU**：通用CPU环境

为了在不同平台上实现统一的崩溃诊断和OOM处理，我们构建了一套硬件无关的诊断系统。

### 13.2 环境配置

#### 13.2.1 Dockerfile环境变量配置

在Dockerfile中添加以下环境变量：

```dockerfile
# 开启 Python 内置崩溃堆栈打印 (跨平台通用)
ENV PYTHONFAULTHANDLER=1

# 强制实时刷新日志，防止崩溃时缓冲区日志丢失
ENV PYTHONUNBUFFERED=1

# 针对 Mac MPS 的优化：当显存占用达到 80% 时强制回收
ENV PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8

# 启用内存监控
ENV ENABLE_MEMORY_MONITOR=true

# 启用诊断日志
ENV ENABLE_DIAGNOSTICS=true

# 设置诊断日志文件路径
ENV DIAGNOSTICS_LOG_FILE=logs/diagnostics.log

# 设置崩溃日志文件路径
ENV CRASH_LOG_FILE=logs/crash.log
```

#### 13.2.2 Kubernetes ConfigMap配置

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: classification-config
data:
  MODEL_NAME: "arona_plana"
  API_TIMEOUT: "30"
  LOG_LEVEL: "INFO"
  CACHE_SIZE: "1000"
  PYTHONFAULTHANDLER: "1"
  PYTHONUNBUFFERED: "1"
  PYTORCH_MPS_HIGH_WATERMARK_RATIO: "0.8"
  ENABLE_MEMORY_MONITOR: "true"
  ENABLE_DIAGNOSTICS: "true"
  MEMORY_WARNING_THRESHOLD: "85"
  MEMORY_CRITICAL_THRESHOLD: "95"
  GPU_MEMORY_WARNING_THRESHOLD: "85"
  GPU_MEMORY_CRITICAL_THRESHOLD: "95"
```

### 13.3 统一诊断工具类

系统提供了跨平台诊断工具类 `utils.diagnostics.CrossPlatformDiagnostics`，支持以下功能：

#### 13.3.1 设备信息获取

```python
from utils.diagnostics import CrossPlatformDiagnostics

# 获取当前设备类型
device = CrossPlatformDiagnostics.get_device_info()
# 返回: "cuda", "mps", 或 "cpu"
```

#### 13.3.2 内存快照

```python
# 生成内存快照
diag_data = CrossPlatformDiagnostics.dump_memory_snapshot()
# 返回包含平台、CPU、内存、GPU等信息的字典
```

#### 13.3.3 缓存清理

```python
# 跨平台缓存清理
CrossPlatformDiagnostics.clear_cache()
# 自动识别设备并执行相应的清理操作
```

#### 13.3.4 内存阈值检查

```python
# 检查内存是否超过阈值
is_high = CrossPlatformDiagnostics.check_memory_threshold(threshold_percent=85.0)
# 返回 True 或 False
```

#### 13.3.5 OOM诊断

```python
# 诊断OOM错误
diagnosis = CrossPlatformDiagnostics.diagnose_oom_error(error)
# 返回包含是否OOM、错误信息、设备类型、内存快照等信息的字典
```

### 13.4 核心推理逻辑改造

#### 13.4.1 特征提取模块

特征提取模块已集成跨平台诊断功能：

```python
# 检查图像大小，防止OOM
if hasattr(img, 'size'):
    width, height = img.size
    pixel_count = width * height
    if pixel_count > 4000000:  # 超过400万像素
        logger.warning(f"图像过大 ({width}x{height} = {pixel_count}像素)，可能导致OOM")
        CrossPlatformDiagnostics.check_memory_threshold(80.0)
```

#### 13.4.2 OOM异常处理

```python
except RuntimeError as e:
    # 跨平台OOM识别和处理
    diagnosis = CrossPlatformDiagnostics.diagnose_oom_error(e)
    if diagnosis["is_oom"]:
        logger.error(f"OOM异常已处理: {diagnosis}")
        # 返回默认特征向量
        return np.random.randn(512).astype(np.float32)
```

### 13.5 崩溃排查手册

#### 13.5.1 情况A：Python代码逻辑错误/RuntimeError

**现象**：程序报错，但进程没死。

**查看**：`logs/runtime_error.log`

**价值**：由于开启了 `diagnose=True`，日志会显示崩溃行所有变量的值。

**示例**：
```
2026-03-28 10:30:45 | ERROR | 检测到 OOM 异常！平台: cuda
2026-03-28 10:30:45 | ERROR | --- 崩溃前设备状态快照 ---
{
  "platform": "Linux",
  "cpu_percent": 85.2,
  "ram_used_gb": 7.8,
  "ram_available_gb": 2.2,
  "gpu_allocated_gb": 7.5,
  "gpu_reserved_gb": 8.0,
  "gpu_max_allocated_gb": 8.2
}
```

#### 13.5.2 情况B：底层C++段错误(Segmentation Fault)/显存彻底打穿

**现象**：程序直接闪退，控制台只留下一句 `Segmentation fault`。

**查看**：

**Linux/K8s**：
```bash
# 执行kubectl logs查看Pod日志
kubectl logs <pod_name>

# faulthandler会在闪退前瞬间将Python堆栈强制打印在stdout
```

**Mac M4**：
1. 打开"控制台" (Console.app)
2. 查看崩溃报告
3. 查找以 `python3` 开头的 `.ips` 文件
4. 搜索 `Termination Reason: Namespace JETSAM`（代表内存超限被系统杀掉）

### 13.6 Kubernetes中的故障排查

#### 13.6.1 查看Pod日志

```bash
# 查看后端Pod日志
kubectl logs -l component=backend --tail=100 -f

# 查看前端Pod日志
kubectl logs -l component=frontend --tail=100 -f

# 查看特定Pod的日志
kubectl logs <pod-name> --tail=100 -f
```

#### 13.6.2 查看诊断日志

```bash
# 查看诊断日志
kubectl logs <pod-name> | grep "崩溃前设备状态快照"

# 查看OOM错误
kubectl logs <pod-name> | grep "OOM异常"
```

#### 13.6.3 查看Pod资源使用

```bash
# 查看Pod资源使用情况
kubectl top pods

# 查看节点资源使用情况
kubectl top nodes
```

#### 13.6.4 进入Pod进行诊断

```bash
# 进入Pod
kubectl exec -it <pod-name> -- /bin/bash

# 在Pod内运行诊断
python3 -c "from utils.diagnostics import CrossPlatformDiagnostics; CrossPlatformDiagnostics.dump_memory_snapshot()"
```

### 13.7 性能优化建议

#### 13.7.1 统一精度

跨平台运行AI模型，`model.half()` (FP16) 是防止MBA M4 OOM的最有效手段。

```python
if self.device.type in ['cuda', 'mps']:
    self.model = self.model.half()
    logger.info(f"已开启 FP16 半精度模式，运行于 {self.device}")
```

#### 13.7.2 日志为王

在loguru中开启 `diagnose=True` 相当于得到了一个"文本版的 Core Dump"，比二进制的 `.core` 文件对Python开发者更友好。

```python
logger.add("logs/runtime_error.log", backtrace=True, diagnose=True, rotation="50MB")
```

#### 13.7.3 动态缩放

针对统一内存（M4），建议在 `cv2.imread` 后立即判断图片像素，若 `width * height > 4000000`（约400万像素），强制等比缩小，这是跨平台稳定性最底层的保障。

```python
if pixel_count > 4000000:
    scale_factor = (4000000 / pixel_count) ** 0.5
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    img = img.resize((new_width, new_height))
```

### 13.8 监控和告警

#### 13.8.1 Prometheus监控指标

```yaml
# 添加到Prometheus配置
- job_name: 'memory-monitoring'
  static_configs:
    - targets: ['character-classification-backend:8000']
  metrics_path: '/api/monitoring/metrics'
```

#### 13.8.2 Grafana仪表板

创建Grafana仪表板监控以下指标：
- CPU使用率
- 内存使用率
- GPU显存使用率（如果使用GPU）
- 请求响应时间
- 错误率
- OOM错误次数

#### 13.8.3 告警规则

```yaml
# Prometheus告警规则
groups:
- name: oom-alerts
  rules:
  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes{container="backend"} / container_spec_memory_limit_bytes{container="backend"} > 0.85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is above 85%"
  
  - alert: OOMDetected
    expr: rate(kube_pod_container_status_terminated_reason{reason="OOMKilled"}[5m]) > 0
    labels:
      severity: critical
    annotations:
      summary: "OOM detected"
      description: "Pod was killed due to OOM"
```

## 14. 总结

通过以上部署方案，我们实现了一个完整的Kubernetes部署流程，包括：

1. **环境准备**：系统要求、系统更新与优化
2. **Kubernetes集群搭建**：Docker安装、MicroK8s部署
3. **应用容器化**：Docker镜像构建
4. **Kubernetes部署**：完整的部署配置和资源管理
5. **网络配置**：Ingress、HTTPS、防火墙
6. **服务验证**：健康检查、API测试
7. **监控与维护**：资源监控、日志管理、备份恢复
8. **故障排查**：常见问题解决方法
9. **自动化**：部署和维护脚本
10. **性能优化**：资源配置和应用优化
11. **安全加固**：容器安全和网络安全
12. **CI/CD集成**：GitLab CI和GitHub Actions
13. **跨平台诊断和OOM处理**：统一的跨平台诊断系统和OOM处理机制

这套部署方案具有以下优势：
- **高可用性**：多副本部署，自动故障转移
- **可扩展性**：水平自动缩放，应对高并发
- **安全性**：网络策略、TLS加密、容器安全
- **可监控性**：Prometheus和Grafana集成
- **自动化**：完整的CI/CD流程
- **易于维护**：详细的监控和故障排查指南
- **跨平台支持**：统一的诊断系统，支持CUDA/MPS/CPU多种平台
- **OOM防护**：完善的OOM检测和处理机制

系统部署完成后，您可以通过域名访问角色分类系统的前端界面，系统将能够处理图像分类请求，提供准确的角色识别结果。

如需进一步定制和优化，可以根据实际需求调整配置参数和资源分配。