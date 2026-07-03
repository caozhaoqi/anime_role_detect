# 1. 重新复制修改后的部署文件
sudo cp -rf ~/anime_role_detect/deployment/* /opt/ardc/deployment/

# 2. 删除之前失败的 Pod（让它们重新创建）
sudo kubectl -n anime-role-detect delete pods --all

# 3. 重新应用 Deployments
sudo kubectl -n anime-role-detect apply -f /opt/ardc/deployment/k8s-deployments.yaml

# 4. 等待并查看状态
sleep 30
sudo kubectl -n anime-role-detect get pods -o wide