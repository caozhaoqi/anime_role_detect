# 在服务器上执行端口转发
# 查看服务器 IP
curl -s ifconfig.me
# 或
hostname -I

sudo kubectl -n anime-role-detect port-forward service/frontend 3000:3000 --address 0.0.0.0 &