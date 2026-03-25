# Ubuntu服务器部署角色分类系统文档

## 1. 服务器准备

### 1.1 系统要求
- Ubuntu 20.04 LTS 或更高版本
- 至少 4GB 内存
- 至少 20GB 磁盘空间
- 具有公网IP地址

### 1.2 更新系统
```bash
sudo apt update && sudo apt upgrade -y
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

# 验证Docker安装
sudo docker --version
```

### 2.2 安装Kubernetes工具
使用MicroK8s（轻量级Kubernetes集群，适合单服务器部署）：
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
sudo docker build -t character-classification-backend:latest -f Dockerfile.backend .
```

### 3.3 构建前端镜像
```bash
sudo docker build -t character-classification-frontend:latest -f Dockerfile.frontend .
```

### 3.4 验证镜像构建
```bash
sudo docker images
```

## 4. 部署应用到Kubernetes

### 4.1 应用后端部署
```bash
sudo microk8s kubectl apply -f backend-deployment.yaml
```

### 4.2 应用前端部署
```bash
sudo microk8s kubectl apply -f frontend-deployment.yaml
```

### 4.3 应用服务配置
```bash
sudo microk8s kubectl apply -f services.yaml
```

### 4.4 应用自动缩放配置
```bash
sudo microk8s kubectl apply -f hpa.yaml
```

## 5. 配置外网访问

### 5.1 查看服务状态
```bash
sudo microk8s kubectl get services
```

### 5.2 配置Ingress（推荐）
创建ingress.yaml文件：
```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: character-classification-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
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

应用Ingress配置：
```bash
sudo microk8s kubectl apply -f ingress.yaml
```

### 5.3 配置防火墙
```bash
# 开放80端口（HTTP）
sudo ufw allow 80/tcp

# 开放443端口（HTTPS，可选）
sudo ufw allow 443/tcp

# 重新加载防火墙规则
sudo ufw reload
```

### 5.4 配置域名解析
在你的域名提供商处，将域名A记录指向服务器的公网IP地址。

## 6. 验证服务

### 6.1 查看Pod状态
```bash
sudo microk8s kubectl get pods
```

### 6.2 查看服务状态
```bash
sudo microk8s kubectl get services
```

### 6.3 访问应用
在浏览器中访问你的域名（例如：http://your-domain.com），应该能看到角色分类系统的前端界面。

### 6.4 测试API
```bash
# 测试API是否可访问
curl -X POST -F "file=@path/to/image.jpg" http://your-domain.com/api/classify
```

## 7. 故障排查

### 7.1 查看Pod日志
```bash
# 查看后端Pod日志
sudo microk8s kubectl logs -l component=backend

# 查看前端Pod日志
sudo microk8s kubectl logs -l component=frontend
```

### 7.2 检查Ingress状态
```bash
sudo microk8s kubectl get ingress
```

### 7.3 检查服务端口
```bash
# 检查前端服务
sudo microk8s kubectl describe service character-classification-frontend

# 检查后端服务
sudo microk8s kubectl describe service character-classification-backend
```

### 7.4 常见问题解决
- **Pod无法启动**：检查Docker镜像是否正确构建，查看Pod日志获取具体错误信息
- **服务无法访问**：检查防火墙规则是否正确配置，检查Ingress规则是否正确
- **API返回错误**：检查后端Pod是否正常运行，查看后端日志获取具体错误信息

## 8. 维护与监控

### 8.1 查看集群状态
```bash
sudo microk8s status
```

### 8.2 查看资源使用情况
```bash
sudo microk8s kubectl top nodes
sudo microk8s kubectl top pods
```

### 8.3 升级应用
1. 构建新的Docker镜像
2. 更新Kubernetes部署配置
3. 应用更新：
   ```bash
   sudo microk8s kubectl apply -f backend-deployment.yaml
   sudo microk8s kubectl apply -f frontend-deployment.yaml
   ```

### 8.4 备份与恢复
- 定期备份Kubernetes配置文件
- 定期备份模型文件和数据

## 9. 总结

通过以上步骤，你可以在Ubuntu服务器上成功部署角色分类系统，并通过公网访问。系统将具备高可用性、自动扩缩容和监控能力，为用户提供稳定的角色识别服务。

如需进一步优化，可以考虑：
- 使用HTTPS加密传输
- 配置更详细的监控和告警
- 实现CI/CD自动化部署流程
- 优化资源配置以提高性能