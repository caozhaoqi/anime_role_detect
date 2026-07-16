- delete all pods

```sh
sudo kubectl -n anime-role-detect delete pods --all
```

- get pods with wide output
```sh
sudo kubectl -n anime-role-detect get pods -o wide
```

- get pod log
```sh
sudo kubectl -n anime-role-detect logs <pod-name>
```

- get svc ip and port forward config list

```sh
kubectl -n anime-role-detect get svc
```

- set dns server

```sh
sudo bash -c 'echo "nameserver 223.5.5.5" > /etc/resolv.conf'
```


# 1. 将修改后的代码同步到服务器（如果还没同步）
scp -r /path/to/local/anime_role_detect/* user@server:~/anime_role_detect/

# 2. 重新构建基础镜像（包含 scikit-image）
cd ~/anime_role_detect
bash scripts/k8s/build_k8s_images.sh --skip-base

# 3. 删除旧的 Pod
kubectl -n anime-role-detect delete pods --all

# 4. 重新部署（会自动导入镜像到 containerd）
bash scripts/k8s/deploy_ubuntu.sh --tag 60f40b1 --local --skip-k8s

# 5. 查看 Pod 状态
kubectl -n anime-role-detect get pods -o wide

# 6. 查看各服务日志
kubectl -n anime-role-detect logs api-gateway-xxx-xxx
kubectl -n anime-role-detect logs api-service-xxx-xxx



# 1. 删除旧的 Elasticsearch Pod 和 PVC（清理锁文件）
cd ~/anime_role_detect
sudo kubectl -n anime-role-detect delete pod elasticsearch-bf8f494df-99nzh --force --grace-period=0 2>/dev/null || true
sudo kubectl -n anime-role-detect delete pvc elasticsearch-data --force --grace-period=0 2>/dev/null || true
sleep 10

# 2. 重新构建镜像（确保使用最新的代码）
sudo bash scripts/k8s/build_k8s_images.sh

# 3. 重新应用部署配置
sudo bash scripts/k8s/deploy_final.sh

# 4. 删除旧 Pod 触发重新部署
sudo kubectl -n anime-role-detect delete pods --all

# 5. 等待 60 秒后查看状态
sleep 60
sudo kubectl -n anime-role-detect get pods -o wide

sudo bash -c 'echo "nameserver 223.5.5.5" > /etc/resolv.conf'

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

# 删除所有异常 Pod（强制删除）
kubectl -n anime-role-detect delete pods --force --grace-period=0 $(kubectl -n anime-role-detect get pods -o jsonpath='{.items[?(@.status.phase!="Running")].metadata.name}') 2>/dev/null || true

# 等待资源释放
sleep 30