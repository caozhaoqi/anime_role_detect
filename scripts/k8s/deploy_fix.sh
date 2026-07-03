# 1. 检查 Docker 中已有的镜像
sudo docker images | grep ardc

# 2. 为所有 c8cd26b 镜像添加 latest 标签
for img in $(sudo docker images --format '{{.Repository}}:{{.Tag}}' | grep ardc | grep c8cd26b); do
    new_tag=$(echo "$img" | sed 's/:c8cd26b/:latest/')
    echo "添加标签: $img -> $new_tag"
    sudo docker tag "$img" "$new_tag"
done

# 3. 验证标签是否正确
sudo docker images | grep ardc

# 4. 检查 frontend 镜像是否存在
sudo docker images | grep frontend || echo "frontend 镜像缺失！"

# 5. 如果 frontend 缺失，临时禁用 frontend 部署
sudo kubectl -n anime-role-detect scale deployment/frontend --replicas=0

# 6. 删除所有 ImagePullBackOff 的 Pod，让它们重新创建
sudo kubectl -n anime-role-detect delete pods --field-selector=status.phase=Pending

# 7. 等待并查看状态
sleep 30
sudo kubectl -n anime-role-detect get pods -o wide