# 安装 tmux（如未安装）
sudo apt install tmux     # Ubuntu/Debian
sudo yum install tmux     # CentOS

# 启动一个后台 session
tmux new -s collector

# 在 tmux 里启动采集
bash scripts/data_collection/start_collector.sh

# 按 Ctrl+B 然后按 D 断开（采集继续后台运行）
# 想回来查看：
tmux attach -t collector
# 不想看了再按 Ctrl+B, D 断开
# 想彻底关掉采集：attach 回来按 Ctrl+C