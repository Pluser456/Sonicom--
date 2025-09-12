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


path = 'HRTF可视化'

freq_list = np.loadtxt(f'{path}\\freq_data_Wi.txt')
freq_list = freq_list / 1000  # 转换为kHz单位

# 注意：确保以下文件都存在
try:
    # HRTF_VAE = np.loadtxt(f'{path}\\lsd_VAE_2D_Wi.txt')
    HRTF_VQVAE = np.loadtxt(f'{path}\\lsd_AE_2D_Wi.txt')
    HRTF_VQVAE_3D = np.loadtxt(f'{path}\\lsd_AE_3D_Wi.txt')
    HRTF_ResNet = np.loadtxt(f'{path}\\lsd_CNN_2D_Wi.txt')  
    HRTF_ResNet_3D = np.loadtxt(f'{path}\\lsd_CNN_3D_Wi.txt')
except FileNotFoundError as e:
    print(f"警告: 找不到文件 {e.filename}。该子图将为空。")

# --- 在当前子图 (ax) 上绘图 ---

plt.figure(figsize=(10, 6))

# 绘制True曲线
# plt.plot(freq_list, HRTF_TRUE, 'k-', linewidth=1.4, label="True")

# 绘制Proposed曲线
plt.plot(freq_list, HRTF_VQVAE, '-', color='#D95319', linewidth=linewidth, 
        markersize=markersize, markerfacecolor='none', markeredgecolor='#D95319', 
        markeredgewidth=1.1, label="Proposed")
# 绘制Proposed 3D曲线
plt.plot(freq_list, HRTF_VQVAE_3D, '-', color="#C02222", linewidth=linewidth, 
        markersize=markersize, markerfacecolor='none', markeredgecolor="#C02222",
        markeredgewidth=1.1, label="Proposed 3D")

# 绘制Hybrid曲线
# plt.plot(freq_list, HRTF_VAE, '--', color='#0072BD', linewidth=linewidth, 
#         markersize=markersize, markerfacecolor='none', markeredgecolor='#0072BD', 
#         markeredgewidth=1.5, label="Hybrid")

# 绘制ResNet曲线
plt.plot(freq_list, HRTF_ResNet, 'g--', linewidth=linewidth, 
        markersize=markersize, markerfacecolor='none', markeredgecolor='green', 
        markeredgewidth=1.5, label="ResNet")

plt.plot(freq_list, HRTF_ResNet_3D, 'm--', linewidth=linewidth,
    markersize=markersize, markerfacecolor='none', markeredgecolor='m', 
    markeredgewidth=1.5, label="ResNet 3D")

# 网格线
plt.grid(True, which="both", linestyle="--", alpha=0.6, linewidth=0.8)

# 刻度参数
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize, 
                direction='in', width=1.2)

# 添加图例
plt.legend(fontsize=legend_fontsize, loc='best', frameon=True, 
            edgecolor='black', fancybox=False)

# 设置坐标轴标签
plt.xlabel('Frequency (kHz)', fontsize=label_fontsize, labelpad=15)
plt.ylabel('LSD (dB)', fontsize=label_fontsize, labelpad=15)
plt.xlim(0,18)
plt.ylim(0,5)
plt.xticks([0, 5, 10, 15])
# 保存和显示图像
# 注意：保存的文件名不再包含单个theta
plt.savefig(f'{path}\\LSD_contrast.pdf', bbox_inches='tight')
plt.show()