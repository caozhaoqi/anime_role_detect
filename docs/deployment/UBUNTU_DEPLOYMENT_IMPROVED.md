> [!CAUTION]
> **已弃用 / 过期**：本文档描述的部署文件名（`configmap.yaml` / `secret.yaml` / `backend-deployment.yaml` 等）与当前仓库已不一致。
> 当前唯一权威 K8s 部署源为 **`k8s/`**（base + overlays/ci），详见 [`k8s/README.md`](../k8s/README.md)。
> 旧 `deployment/k8s-*.yaml` 已归档至 `deployment/_legacy_backup/`，仅供历史追溯。
>
> 仅保留作为历史参考，请勿按本文档执行部署。


# Ubuntu服务器部署角色分类系统文档（改进版）

## 1. 服务器准备

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

## 2. 安装必要软件

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
```bash
sudo microk8s kubectl get pods
```

### 6.2 查看服务状态
```bash
sudo microk8s kubectl get services
sudo microk8s kubectl get ingress
```

### 6.3 访问应用
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

### 7.3 查看监控面板
```bash
# 获取Grafana访问地址
sudo microk8s kubectl get services -n monitoring

# 获取Grafana密码
sudo microk8s kubectl get secret -n monitoring grafana -o jsonpath="{.data.admin-password}" | base64 --decode
```

### 7.4 升级应用
1. 构建新的Docker镜像
2. 更新Kubernetes部署配置
3. 应用更新：
   ```bash
   sudo microk8s kubectl apply -f backend-deployment.yaml
   sudo microk8s kubectl apply -f frontend-deployment.yaml
   ```

### 7.5 备份与恢复
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

## 13. 总结

通过以上改进，我们实现了一个更加健壮、安全、高效的角色分类系统部署方案。主要改进包括：

1. **系统优化**：增加了系统级别的优化配置，提高服务器性能
2. **容器化改进**：优化了Docker镜像构建过程，添加了镜像加速
3. **Kubernetes增强**：添加了更详细的部署配置，包括资源管理、健康检查、自动缩放等
4. **网络安全**：配置了HTTPS、网络策略等安全措施
5. **监控与维护**：集成了Prometheus和Grafana，提供了详细的监控和告警
6. **自动化**：提供了部署和维护脚本，简化了运维工作
7. **CI/CD集成**：配置了GitLab CI和GitHub Actions，实现了自动化构建和部署
8. **性能优化**：提供了资源配置和应用优化建议
9. **安全加固**：增加了容器安全和网络安全措施

这些改进将使角色分类系统更加稳定、可靠、安全，同时提高了部署和维护的效率。系统将能够更好地应对高并发请求，提供更快的响应速度和更准确的分类结果。

如需进一步优化，可以考虑：
- 使用分布式存储系统存储模型文件
- 实现多区域部署，提高系统可用性
- 使用服务网格技术，进一步优化服务间通信
- 集成更高级的监控和告警系统
- 实现自动故障恢复和灾难备份机制
