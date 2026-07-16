# 1. 强制创建命名空间（防止 YAML 文件中没有包含创建逻辑）
kubectl create ns anime-role-detect || true

# 2. 部署持久化存储卷 (Volumes)
kubectl apply -f ../../deployment/k8s-volumes.yaml

# 3. 部署服务发现与网络端口 (Services)
kubectl apply -f ../../deployment/k8s-services.yaml

# 4. 部署应用容器核心 (Deployments)
kubectl apply -f ../../deployment/k8s-deployments.yaml

# 5. 部署其他外围组件（水平伸缩 HPA、路由 Ingress、中断预算 PDB）
kubectl apply -f ../../deployment/k8s-hpa.yaml
kubectl apply -f ../../deployment/k8s-ingress.yaml
kubectl apply -f ../../deployment/k8s-pdb.yaml




# 将本地 docker 镜像直接导入到 K8s 的 containerd 运行时中
docker save your-frontend-image:latest | sudo ctr -n=k8s.io images import -
docker save your-backend-image:latest | sudo ctr -n=k8s.io images import -


kubectl -n anime-role-detect get pods -o wide -w