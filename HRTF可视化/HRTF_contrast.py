import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 字体与字号全局设置
# -------------------------
plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体设置
title_fontsize = 22    # 子图标题字号
label_fontsize = 26    # 共享坐标轴标签字号
tick_fontsize = 22     # 坐标轴刻度字号
linewidth = 1.6        # 线条粗细
markersize = 8         # 数据点大小
legend_fontsize = 14   # 图例字号
# ------------------------- 

# --- 创建 2x2 子图网格 ---
# sharex=True 和 sharey=True 会自动处理刻度标签的显示
fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=120, sharex=False, sharey=False)
axes = axes.flatten()  # 将 2x2 的 axes 数组展平为一维，方便遍历

# --- 定义四个子图对应的 theta 值 ---
thetas_to_plot = ["0_0", "0_90", "90_0", "20_54"]
path = 'HRTF可视化'

# --- 循环绘制每个子图 ---
for ax, theta in zip(axes, thetas_to_plot):
    # 从文件导入对应 theta 的数据
    freq_list = np.loadtxt(f'{path}\\freq_data_Wi.txt')
    freq_list = freq_list / 1000  # 转换为kHz单位
    
    # 注意：确保以下文件都存在
    try:
        HRTF_VAE = np.loadtxt(f'{path}\\hrtf_VAE_{theta}.txt')
        HRTF_VQVAE = np.loadtxt(f'{path}\\hrtf_AE_{theta}_2D_Wi.txt')
        HRTF_VQVAE_3D = np.loadtxt(f'{path}\\hrtf_AE_{theta}_3D_Wi.txt')
        HRTF_ResNet = np.loadtxt(f'{path}\\hrtf_base_{theta}.txt')  
        HRTF_TRUE = np.loadtxt(f'{path}\\hrtf_true_{theta}_2D_Wi.txt')
    except FileNotFoundError as e:
        print(f"警告: 找不到文件 {e.filename}。该子图将为空。")
        continue

    # --- 在当前子图 (ax) 上绘图 ---
    
    # 绘制True曲线
    ax.plot(freq_list, HRTF_TRUE, 'k-', linewidth=1.4, label="True")

    # 绘制Proposed曲线
    ax.plot(freq_list, HRTF_VQVAE, '-', color='#D95319', linewidth=linewidth, 
            markersize=markersize, markerfacecolor='none', markeredgecolor='#D95319', 
            markeredgewidth=1.1, label="Proposed")
        # 绘制Proposed 3D曲线
    ax.plot(freq_list, HRTF_VQVAE_3D, '-', color="#C02222", linewidth=linewidth, 
            markersize=markersize, markerfacecolor='none', markeredgecolor="#C02222",
            markeredgewidth=1.1, label="Proposed 3D")

    # 绘制Hybrid曲线
    ax.plot(freq_list, HRTF_VAE, '--', color='#0072BD', linewidth=linewidth, 
            markersize=markersize, markerfacecolor='none', markeredgecolor='#0072BD', 
            markeredgewidth=1.5, label="Hybrid")

    # 绘制ResNet曲线
    ax.plot(freq_list, HRTF_ResNet, 'g--', linewidth=linewidth, 
            markersize=markersize, markerfacecolor='none', markeredgecolor='green', 
            markeredgewidth=1.5, label="ResNet")

    # --- 单个子图的设置 ---
    
    # 设置标题
    angle_list = theta.split('_')
    ax.set_title(f"({angle_list[0]}°, {angle_list[1]}°)", fontsize=title_fontsize, pad=10)

    # 网格线
    ax.grid(True, which="both", linestyle="--", alpha=0.6, linewidth=0.8)

    # 刻度参数
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, 
                   direction='in', width=1.2)

    # 添加图例
    ax.legend(fontsize=legend_fontsize, loc='best', frameon=True, 
              edgecolor='black', fancybox=False)

# --- 整个图形的共享设置 ---

# 设置所有子图的坐标轴范围
plt.setp(axes, xlim=(0, 18), ylim=(-50, 5))

# 为左侧子图添加Y轴标签
axes[0].set_ylabel('Magnitude (dB)', fontsize=label_fontsize, labelpad=8)
axes[2].set_ylabel('Magnitude (dB)', fontsize=label_fontsize, labelpad=8)

# 为底部子图添加X轴标签
axes[2].set_xlabel('Frequency(kHz)', fontsize=label_fontsize, labelpad=8)
axes[3].set_xlabel('Frequency(kHz)', fontsize=label_fontsize, labelpad=8)

# 特别设置左下角子图的坐标范围
axes[2].set_ylim(-55, 0)
# 调整布局以防止重叠
plt.tight_layout(pad=1.5)

# 保存和显示图像
# 注意：保存的文件名不再包含单个theta
plt.savefig(f'{path}\\HRTF_contrast_subplots.pdf', bbox_inches='tight')
plt.show()