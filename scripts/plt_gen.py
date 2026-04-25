import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
import sys

# ================= 解决中文乱码的核心代码 =================
# 根据不同的操作系统，自动选择自带的中文字体
if sys.platform.startswith('win'):
    # Windows 系统
    plt.rcParams['font.sans-serif'] =['Microsoft YaHei', 'SimHei', 'SimSun']
elif sys.platform.startswith('darwin'):
    # Mac 苹果系统
    plt.rcParams['font.sans-serif'] =['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
else:
    # Linux 系统
    plt.rcParams['font.sans-serif'] =['WenQuanYi Micro Hei', 'Droid Sans Fallback']

plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# ==========================================================


# X轴数据：时间 (小时)，转换为numpy数组
time = np.array([1, 2, 4, 8, 16, 24, 48])

# Y轴数据：扩散长度 (mm)
油脂型 = np.array([0.1, 0.1, 0.3, 0.6, 2.0, 2.0, 3.0])
乳剂型 = np.array([1.7, 1.9, 2.3, 2.8, 3.5, 3.5, 5.5])
水溶性 = np.array([0.5, 0.6, 1.4, 1.75, 2.7, 2.7, 4.0])

# 生成平滑曲线数据 (300个密集的点)
time_smooth = np.linspace(time.min(), time.max(), 300)

# 使用 PchipInterpolator 进行平滑拟合
油脂型_smooth = PchipInterpolator(time, 油脂型)(time_smooth)
乳剂型_smooth = PchipInterpolator(time, 乳剂型)(time_smooth)
水溶性_smooth = PchipInterpolator(time, 水溶性)(time_smooth)

# 创建图表
plt.figure(figsize=(10, 6), dpi=120)

# 绘制平滑曲线 (linestyle='-')
plt.plot(time_smooth, 油脂型_smooth, linestyle='-', color='#1f77b4', linewidth=2, label='油脂型')
plt.plot(time_smooth, 乳剂型_smooth, linestyle='-', color='#ff7f0e', linewidth=2, label='乳剂型')
plt.plot(time_smooth, 水溶性_smooth, linestyle='-', color='#2ca02c', linewidth=2, label='水溶性')

# 绘制原始实验数据点 (linestyle='')
plt.plot(time, 油脂型, marker='o', linestyle='', color='#1f77b4', markersize=6)
plt.plot(time, 乳剂型, marker='s', linestyle='', color='#ff7f0e', markersize=6)
plt.plot(time, 水溶性, marker='^', linestyle='', color='#2ca02c', markersize=6)

# 设置图表标题和坐标轴标签
plt.title('药物在不同软膏基质中的释放情况', fontsize=16, pad=15)
plt.xlabel('时间 (h)', fontsize=12)
plt.ylabel('扩散长度 (cm)', fontsize=12)

# 设置X轴刻度
plt.xticks(time)

# 添加图例和网格
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 紧凑布局并显示
plt.tight_layout()
plt.show()