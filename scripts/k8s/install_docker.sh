#!/bin/bash

# ==============================================================================
# 脚本功能：一键在 Ubuntu 系统安装 Docker, Docker Compose 和 Kubernetes 组件
# 适用系统：Ubuntu 20.04 / 22.04 / 24.04
# ==============================================================================

# 严格模式：遇到任何错误立刻停止执行
set -e

# ================= 配置区 =================
# 是否使用国内阿里云镜像源（国内服务器请保持 true，海外服务器请修改为 false）
USE_CHINA_MIRROR=true

# 预设要安装的 K8s 主版本号（例如：v1.31）
K8S_VERSION="v1.31"
# =========================================

# 颜色控制输出
INFO() { echo -e "\033[32m[INFO] $1\033[0m"; }
WARN() { echo -e "\033[33m[WARN] $1\033[0m"; }
ERROR() { echo -e "\033[31m[ERROR] $1\033[0m"; exit 1; }

# 1. 权限检查
if [ "$EUID" -ne 0 ]; then
    ERROR "请使用 sudo 或 root 账户运行此脚本！"
fi

# 2. 基础系统更新
INFO "正在更新系统包索引..."
apt-get update -y && apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release

# 3. 关闭 SWAP 分区（Kubernetes 必备要求）
INFO "正在关闭系统 SWAP 分区..."
swapoff -a
# 永久关闭（防止重启后失效）
sed -i '/swap/s/^\(.*\)$/#\1/g' /etc/fstab

# 4. 开启内核转发与网桥过滤模块
INFO "正在配置网桥与内核参数功能..."
cat <<EOF | tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

modprobe overlay
modprobe br_netfilter

# 设置所需的 sysctl 参数
cat <<EOF | tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sysctl --system

# 5. 安装 Docker 与 Docker Compose
INFO "正在配置 Docker 存储库..."
mkdir -p /etc/apt/keyrings

if [ "$USE_CHINA_MIRROR" = true ]; then
    INFO "使用 阿里云 Docker 镜像源..."
    curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list
else
    INFO "使用 Docker 官方源..."
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list
fi

INFO "正在安装 Docker 引擎与 Docker Compose 插件..."
apt-get update -y
# docker-compose-plugin 会安装原生的 "docker compose" 命令行命令
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 软链接支持旧版的带横杠命令：使 "docker-compose" 指向 "docker compose"
ln -sf /usr/libexec/docker/cli-plugins/docker-compose /usr/local/bin/docker-compose

# 启动并配置开机自启
systemctl enable docker --now

# 6. 配置 Containerd 运行时（K8s 强依赖容器运行时，且默认推荐使用 SystemdCgroup）
INFO "正在生成并配置 containerd 规则..."
mkdir -p /etc/containerd
containerd config default | tee /etc/containerd/config.toml
# 将 SystemdCgroup 设置为 true，这是 kubeadm 启动的强制规范
sed -i 's/SystemdCgroup = false/SystemdCgroup = true/g' /etc/containerd/config.toml
systemctl restart containerd
systemctl enable containerd

# 7. 安装 Kubernetes 组件 (使用最新的社区原生源 pkgs.k8s.io 架构)
INFO "正在配置 Kubernetes 存储库..."

# 清理可能存在的旧版本 K8s 冲突源
rm -f /etc/apt/sources.list.d/kubernetes.list

if [ "$USE_CHINA_MIRROR" = true ]; then
    INFO "使用 阿里云 Kubernetes-New 社区镜像源（适配新社区规范架构）..."
    curl -fsSL https://mirrors.aliyun.com/kubernetes-new/core/stable/${K8S_VERSION}/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
    echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://mirrors.aliyun.com/kubernetes-new/core/stable/${K8S_VERSION}/deb/ /" | tee /etc/apt/sources.list.d/kubernetes.list
else
    INFO "使用 官方 pkgs.k8s.io 原生源..."
    curl -fsSL https://pkgs.k8s.io/core:/stable:/${K8S_VERSION}/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
    echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/${K8S_VERSION}/deb/ /" | tee /etc/apt/sources.list.d/kubernetes.list
fi

INFO "正在安装 Kubelet, Kubeadm, Kubectl..."
apt-get update -y
apt-get install -y kubelet kubeadm kubectl
# 锁定版本，防止系统自动更新 K8s 相关包导致集群损坏
apt-mark hold kubelet kubeadm kubectl

# 启动并开机自启 kubelet
systemctl enable kubelet --now

# 8. 验证
INFO "================================================================="
INFO "安装已完成！各组件版本验证如下："
INFO "================================================================="
docker --version
docker compose version
kubeadm version
kubectl version --client --output=yaml | grep -i gitVersion
INFO "================================================================="

if [ "$USE_CHINA_MIRROR" = true ]; then
    WARN "【提示】由于您使用的是国内镜像源，在后续初始化控制节点时，"
    WARN "请使用如下命令拉取 K8s 系统镜像，否则可能因网络问题超时失败："
    WARN "kubeadm config images pull --image-repository=registry.aliyuncs.com/google_containers"
else
    WARN "【提示】初始化主节点时请使用标准的：kubeadm init"
fi