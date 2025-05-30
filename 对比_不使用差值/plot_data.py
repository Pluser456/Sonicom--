import numpy as np
import matplotlib.pyplot as plt

# 从文件导入数据
freq_list = np.loadtxt('freq_data.txt')
avg_lsd_per_freq = np.loadtxt('lsd_data.txt')
avg_lsd_per_freq1 = np.loadtxt('lsd_data1.txt')
avg_lsd_per_freq_of_mean = np.loadtxt('lsd_mean_data.txt')
#-------------------------
# 字体与字号全局设置（只需修改这里即可统一调整）
#-------------------------
plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体设置
title_fontsize = 16     # 标题字号
label_fontsize = 14     # 坐标轴标签字号
tick_fontsize = 12       # 坐标轴刻度字号
linewidth = 1.5         # 线条粗细
markersize = 8          # 数据点大小
legend_fontsize = 12    # 图例字号 - 新增
#-------------------------    
# 绘制频率-LSD图

plt.figure(figsize=(10, 6), dpi=120)  # 设置更高分辨率

# 绘制曲线 (红色实线+圆形标记)
plt.plot(freq_list, avg_lsd_per_freq, 
                'r-o',                    # 红色实线+圆形标记
                linewidth=linewidth, 
                markersize=markersize,
                markeredgecolor='black',  # 增加黑色描边
                markeredgewidth=0.5,
                label = "ResNet")      # 标记描边粗细

plt.plot(freq_list, avg_lsd_per_freq1, 
                'b-o',                    # 红色实线+圆形标记
                linewidth=linewidth, 
                markersize=markersize,
                markeredgecolor='black',  # 增加黑色描边
                markeredgewidth=0.5,
                label = "VQVAE")      # 标记描边粗细

plt.plot(freq_list, avg_lsd_per_freq_of_mean, 
                'y-o',                    # 红色实线+圆形标记
                linewidth=1.2, 
                markersize=5,
                markeredgecolor='black',  # 增加黑色描边
                markeredgewidth=0.5,
                label = "Mean")      # 标记描边粗细

# 坐标轴设置
plt.xlim(min(freq_list)*0.9, max(freq_list)*1.1)  # 留出10%空白边距
plt.ylim(0, 5.2)                                  # 根据你的数据示例设置

# 标签与标题
# plt.title('Frequency vs LSD', fontsize=title_fontsize, pad=15)  # pad是标题间距
plt.xlabel('Frequency(Hz)', fontsize=label_fontsize, labelpad=8)
plt.ylabel('LSD (dB)', fontsize=label_fontsize, labelpad=8)

# 刻度参数
plt.tick_params(axis='both', which='major', 
                labelsize=tick_fontsize, 
                direction='in',          # 刻度线朝内
                width=1.2)               # 刻度线粗细

# 网格线
plt.grid(True, which="both", 
        linestyle="--", 
        alpha=0.6,                    # 透明度
        linewidth=0.8)
# 添加图例 - 这是关键修复点
plt.legend(fontsize=legend_fontsize, 
           loc='best',                 # 自动选择最佳位置
           frameon=True,               # 显示图例框
           edgecolor='black',          # 图例框边框颜色
           fancybox=False)             # 不使用圆角边框
# 先保存再显示 (避免保存空白图片)
plt.savefig("LSD_per_frequency.png", bbox_inches='tight', dpi=300)  # 保存高清图
plt.savefig("LSD_per_frequency.pdf", bbox_inches='tight')           # 矢量图格式
plt.show()
