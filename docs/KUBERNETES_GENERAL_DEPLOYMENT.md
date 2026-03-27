# Kubernetes通用部署指南

## 1. Kubernetes基本原理

### 1.1 什么是Kubernetes
Kubernetes（简称K8s）是一个开源的容器编排平台，用于自动化部署、扩展和管理容器化应用程序。它最初由Google设计，现在由Cloud Native Computing Foundation（CNCF）维护。

### 1.2 Kubernetes核心概念

| 概念 | 描述 |
|------|------|
| Pod | 最小的部署单元，包含一个或多个容器 |
| Deployment | 管理Pod的副本数量和更新策略 |
| Service | 提供稳定的网络访问点 |
| Ingress | 管理外部访问到集群服务的规则 |
| ConfigMap | 存储配置数据 |
| Secret | 存储敏感信息 |
| Volume | 提供持久化存储 |
| Namespace | 逻辑上隔离资源 |
| Node | 集群中的工作机器 |
| Cluster | 由多个Node组成的集合 |

### 1.3 Kubernetes架构

![Kubernetes架构图](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Kubernetes%20architecture%20diagram%20showing%20control%20plane%20components%20(API%20server%2C%20etcd%2C%20scheduler%2C%20controller%20manager)%20and%20worker%20nodes%20with%20kubelet%2C%20kube-proxy%2C%20and%20containers%2C%20professional%20technical%20diagram%2C%20clear%20labels%2C%20blue%20and%20white%20color%20scheme&image_size=landscape_16_9)

**控制平面组件**：
- **API Server**：集群的控制中心，处理所有REST请求
- **etcd**：分布式键值存储，存储集群状态
- **Scheduler**：调度Pod到合适的Node
- **Controller Manager**：管理各种控制器

**节点组件**：
- **Kubelet**：管理节点上的Pod
- **Kube-proxy**：维护网络规则
- **Container Runtime**：运行容器（如Docker）

## 2. 环境准备

### 2.1 系统要求
- Ubuntu 20.04 LTS 或更高版本
- **推荐配置**：
  - 小型部署：4GB内存，20GB磁盘空间，2核CPU
  - 中型部署：8GB内存，40GB磁盘空间，4核CPU
  - 大型部署：16GB+内存，80GB+磁盘空间，8核+CPU
- 网络连接良好

### 2.2 服务器操作基础

#### 2.2.1 服务器登录方法

**通过SSH登录**：
```bash
# 使用密码登录
ssh username@server_ip

# 使用密钥登录
ssh -i ~/.ssh/id_rsa username@server_ip
```

**通过堡垒机登录**：
1. 先登录堡垒机
2. 从堡垒机登录目标服务器

#### 2.2.2 文件上传工具

**使用SCP上传文件**：
```bash
# 上传单个文件
scp local_file username@server_ip:/remote/path

# 上传目录
scp -r local_directory username@server_ip:/remote/path
```

**使用Filezilla**：
1. 下载并安装Filezilla
2. 打开Filezilla，输入服务器IP、用户名、密码和端口
3. 连接后，拖拽文件进行上传下载

#### 2.2.3 Linux基础命令

**常用命令**：
| 命令 | 功能 | 示例 |
|------|------|------|
| `ls` | 列出文件和目录 | `ls -la` |
| `mkdir` | 创建目录 | `mkdir -p /path/to/dir` |
| `cd` | 切换目录 | `cd /path/to/dir` |
| `ping` | 测试网络连通性 | `ping google.com` |
| `telnet` | 测试端口可用性 | `telnet localhost 8080` |
| `curl` | 发送HTTP请求 | `curl http://localhost:8080` |
| `top` | 查看系统资源使用情况 | `top` |
| `df` | 查看磁盘空间 | `df -h` |
| `free` | 查看内存使用情况 | `free -h` |

**命令注意事项**：
- 命令和参数之间需要有空格
- 路径中的空格需要使用引号或转义符
- 区分大小写

### 2.3 安装Docker
```bash
# 安装依赖
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common

# 添加Docker GPG密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# 添加Docker仓库
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# 安装Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

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
```

### 2.4 安装Kubernetes工具

#### 2.4.1 使用MicroK8s（轻量级，适合单节点）
```bash
# 安装MicroK8s
sudo snap install microk8s --classic

# 启动MicroK8s
sudo microk8s start

# 启用必要的插件
sudo microk8s enable dns dashboard storage ingress

# 配置kubectl
sudo microk8s kubectl config view --raw > ~/.kube/config
chmod 600 ~/.kube/config

# 验证集群状态
sudo microk8s kubectl get nodes
```

#### 2.4.2 使用Minikube（适合开发和测试）
```bash
# 安装Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# 启动Minikube
minikube start --driver=docker

# 验证集群状态
kubectl get nodes
```

## 3. Kubernetes命令详解

### 3.1 kubectl基础命令

**集群管理命令**：
| 命令 | 功能 | 示例 |
|------|------|------|
| `kubectl get nodes` | 查看节点信息 | `kubectl get nodes -o wide` |
| `kubectl cluster-info` | 查看集群信息 | `kubectl cluster-info` |
| `kubectl version` | 查看Kubernetes版本 | `kubectl version` |

**Pod管理命令**：
| 命令 | 功能 | 示例 |
|------|------|------|
| `kubectl get pods` | 查看Pod状态 | `kubectl get pods -n default` |
| `kubectl get pods -o wide` | 查看详细Pod信息 | `kubectl get pods -o wide` |
| `kubectl logs -f <pod-name>` | 实时查看Pod日志 | `kubectl logs -f --tail=200 python-app-7d4f9f6c7f-5x7k9` |
| `kubectl exec -it <pod-name> -- bash` | 进入容器内部 | `kubectl exec -it python-app-7d4f9f6c7f-5x7k9 -- bash` |
| `kubectl describe pod <pod-name>` | 查看Pod详细信息和故障原因 | `kubectl describe pod python-app-7d4f9f6c7f-5x7k9` |
| `kubectl delete pod <pod-name>` | 删除Pod | `kubectl delete pod python-app-7d4f9f6c7f-5x7k9` |

**Deployment管理命令**：
| 命令 | 功能 | 示例 |
|------|------|------|
| `kubectl get deployments` | 查看Deployment状态 | `kubectl get deployments` |
| `kubectl scale deployment <deployment-name> --replicas=<num>` | 调整Pod数量 | `kubectl scale deployment python-app --replicas=3` |
| `kubectl rollout status deployment <deployment-name>` | 查看滚动更新状态 | `kubectl rollout status deployment python-app` |
| `kubectl rollout undo deployment <deployment-name>` | 回滚Deployment | `kubectl rollout undo deployment python-app` |

**Service管理命令**：
| 命令 | 功能 | 示例 |
|------|------|------|
| `kubectl get services` | 查看Service状态 | `kubectl get services` |
| `kubectl describe service <service-name>` | 查看Service详细信息 | `kubectl describe service python-app-service` |

**配置管理命令**：
| 命令 | 功能 | 示例 |
|------|------|------|
| `kubectl get configmaps` | 查看ConfigMap | `kubectl get configmaps` |
| `kubectl get secrets` | 查看Secret | `kubectl get secrets` |
| `kubectl get persistentvolumeclaims` | 查看持久卷声明 | `kubectl get pvc` |

### 3.2 命令执行示例

**查看集群状态**：
```bash
kubectl get nodes
# 输出示例：
# NAME       STATUS   ROLES    AGE     VERSION
# microk8s   Ready    <none>   2d1h    v1.26.1
```

**查看Pod状态**：
```bash
kubectl get pods
# 输出示例：
# NAME                          READY   STATUS    RESTARTS   AGE
# python-app-7d4f9f6c7f-5x7k9   1/1     Running   0          10m
# python-app-7d4f9f6c7f-8z2p7   1/1     Running   0          10m
```

**查看Pod日志**：
```bash
kubectl logs python-app-7d4f9f6c7f-5x7k9
# 输出示例：
# * Running on http://0.0.0.0:8080/ (Press CTRL+C to quit)
# 10.1.1.1 - - [01/Jan/2024 00:00:00] "GET / HTTP/1.1" 200 -
```

## 4. Kubernetes通用部署方法

### 4.1 基本部署流程

1. **容器化应用**：创建Dockerfile并构建镜像
2. **编写部署配置**：创建Deployment、Service等资源配置
3. **应用配置**：使用kubectl apply部署应用
4. **验证部署**：检查Pod状态和服务可用性
5. **监控和维护**：设置监控和日志收集

### 4.2 部署配置文件结构

#### 4.2.1 Deployment配置
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: my-app:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "100m"
            memory: "256Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
```

#### 4.2.2 Service配置
```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

#### 4.2.3 Ingress配置
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-app-ingress
spec:
  rules:
  - host: my-app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: my-app-service
            port:
              number: 80
```

### 4.3 部署命令

```bash
# 应用部署配置
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# 查看部署状态
kubectl get deployments
kubectl get pods
kubectl get services
kubectl get ingress

# 查看Pod日志
kubectl logs -f <pod-name>

# 查看Pod详细信息
kubectl describe pod <pod-name>
```

## 5. Demo应用部署示例

### 5.1 简单的Python Web应用

#### 5.1.1 创建应用代码
```python
# app.py
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, Kubernetes!"

@app.route('/health')
def health():
    return "OK"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

#### 5.1.2 创建Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8080

CMD ["python", "app.py"]
```

#### 5.1.3 创建requirements.txt
```
Flask==2.0.1
```

#### 5.1.4 构建Docker镜像
```bash
docker build -t my-python-app:latest .
```

#### 5.1.5 创建部署配置

**deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: python-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: python-app
  template:
    metadata:
      labels:
        app: python-app
    spec:
      containers:
      - name: python-app
        image: my-python-app:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "100m"
            memory: "256Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 20
```

**service.yaml**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: python-app-service
spec:
  selector:
    app: python-app
  ports:
  - port: 80
    targetPort: 8080
  type: NodePort
```

#### 5.1.6 部署应用
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# 查看部署状态
kubectl get pods
kubectl get services

# 获取服务访问地址
minikube service python-app-service --url  # Minikube
# 或
kubectl get service python-app-service  # MicroK8s
```

### 5.2 多容器应用示例

#### 5.2.1 创建部署配置
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multi-container-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: multi-container
  template:
    metadata:
      labels:
        app: multi-container
    spec:
      containers:
      - name: web
        image: nginx:latest
        ports:
        - containerPort: 80
      - name: sidecar
        image: busybox:latest
        command: ["sh", "-c", "while true; do echo $(date) >> /var/log/date.log; sleep 10; done"]
        volumeMounts:
        - name: shared-logs
          mountPath: /var/log
      volumes:
      - name: shared-logs
        emptyDir: {}
```

#### 5.2.2 部署多容器应用
```bash
kubectl apply -f multi-container-deployment.yaml

# 查看Pod状态
kubectl get pods

# 查看容器日志
kubectl logs <pod-name> web
kubectl logs <pod-name> sidecar
```

## 6. 高级功能

### 6.1 配置管理

#### 6.1.1 使用ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  APP_ENV: "production"
  DEBUG: "false"
  DATABASE_URL: "postgresql://user:pass@db:5432/mydb"
```

#### 6.1.2 使用Secret
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secret
type: Opaque
data:
  API_KEY: "base64_encoded_key"
  PASSWORD: "base64_encoded_password"
```

### 6.2 自动缩放

#### 6.2.1 水平Pod自动缩放
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
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

### 6.3 持久化存储

#### 6.3.1 使用PersistentVolumeClaim
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: app-storage
 spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

## 7. 监控与日志

### 7.1 监控

#### 7.1.1 安装Prometheus和Grafana
```bash
# MicroK8s
sudo microk8s enable prometheus grafana

# 查看监控服务
sudo microk8s kubectl get services -n monitoring

# 获取Grafana密码
sudo microk8s kubectl get secret -n monitoring grafana -o jsonpath="{.data.admin-password}" | base64 --decode
```

### 7.2 日志管理

#### 7.2.1 查看Pod日志
```bash
# 查看最近的日志
kubectl logs <pod-name>

# 实时查看日志
kubectl logs -f <pod-name>

# 查看特定容器的日志
kubectl logs <pod-name> -c <container-name>
```

## 8. 常见问题与解决方案

### 8.1 Pod无法启动
- **检查镜像是否存在**：`docker pull <image-name>`
- **查看Pod日志**：`kubectl logs <pod-name>`
- **查看Pod事件**：`kubectl describe pod <pod-name>`

### 8.2 服务无法访问
- **检查Service配置**：`kubectl get service <service-name>`
- **检查网络策略**：`kubectl get networkpolicy`
- **检查Ingress配置**：`kubectl get ingress`

### 8.3 资源不足
- **检查节点资源**：`kubectl top nodes`
- **调整Pod资源配置**：修改deployment.yaml中的resources部分
- **启用自动缩放**：配置HPA

### 8.4 配置错误
- **验证配置文件**：`kubectl apply --dry-run=client -f <file.yaml>`
- **查看配置状态**：`kubectl get configmap` 和 `kubectl get secret`

## 9. 最佳实践

### 9.1 部署最佳实践
- **使用命名空间**：隔离不同环境的资源
- **设置资源限制**：避免单个Pod消耗过多资源
- **使用健康检查**：确保Pod状态正常
- **配置就绪探针**：确保应用准备就绪后再接收流量
- **使用滚动更新**：避免服务中断

### 9.2 安全最佳实践
- **使用非root用户**：在Dockerfile中设置USER
- **限制容器权限**：配置securityContext
- **使用Secret管理敏感信息**：避免明文存储密码
- **配置网络策略**：限制Pod间通信
- **定期更新镜像**：修复安全漏洞

### 9.3 性能最佳实践
- **优化镜像大小**：使用alpine基础镜像
- **合理设置资源请求和限制**：根据实际需求调整
- **使用水平自动缩放**：根据负载自动调整Pod数量
- **优化存储配置**：使用合适的存储类型
- **配置缓存**：减少重复计算

## 10. 总结

Kubernetes是一个强大的容器编排平台，通过本文的指南，您应该能够：

1. **理解Kubernetes的基本原理和架构**
2. **搭建Kubernetes集群环境**
3. **使用通用方法部署应用**
4. **部署和管理demo应用**
5. **使用高级功能如配置管理、自动缩放和持久化存储**
6. **监控和维护Kubernetes集群**
7. **解决常见问题**
8. **遵循最佳实践**

Kubernetes的学习曲线可能较陡，但一旦掌握，它将成为您部署和管理容器化应用的强大工具。通过不断实践和学习，您可以充分利用Kubernetes的优势，构建更加可靠、可扩展的应用系统。

## 11. 参考资源

- [Kubernetes官方文档](https://kubernetes.io/docs/home/)
- [Kubernetes中文文档](https://kubernetes.io/zh-cn/docs/home/)
- [MicroK8s文档](https://microk8s.io/docs/)
- [Minikube文档](https://minikube.sigs.k8s.io/docs/)
- [Docker官方文档](https://docs.docker.com/)

---