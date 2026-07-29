# 1. 强制创建命名空间（防止 YAML 文件中没有包含创建逻辑）
kubectl create ns anime-role-detect || true

# 2. 部署全部 K8s 资源（权威源：k8s/base/）
#    旧 deployment/k8s-*.yaml 已归档至 deployment/_legacy_backup/
kubectl apply -k ../../k8s/base/



# 将本地 docker 镜像直接导入到 K8s 的 containerd 运行时中
docker save your-frontend-image:latest | sudo ctr -n=k8s.io images import -
docker save your-backend-image:latest | sudo ctr -n=k8s.io images import -


kubectl -n anime-role-detect get pods -o wide -w
