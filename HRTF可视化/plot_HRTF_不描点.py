import numpy as np
import matplotlib.pyplot as plt

theta = "0_0"
# 从文件导入数据
freq_list = np.loadtxt('HRTF可视化\\freq_data.txt')
freq_list = freq_list / 1000  # 转换为kHz单位
# avg_lsd_per_freq = np.loadtxt(f'HRTF可视化\hrtf_base_{theta}.txt')      # Baseline
HRTF_VAE = np.loadtxt(f'HRTF可视化\hrtf_VAE_{theta}.txt')      # Baseline

HRTF_VQVAE = np.loadtxt(f'HRTF可视化\hrtf_AE_{theta}.txt')      # PRTFNet
HRTF_ResNet = np.loadtxt(f'HRTF可视化\hrtf_base_{theta}.txt')      # PRTFNet

HRTF_TRUE = np.loadtxt(f'HRTF可视化\hrtf_true_{theta}.txt') # True

#-------------------------
# 字体与字号全局设置
#-------------------------
plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体设置
title_fontsize = 28    # 标题字号
label_fontsize = 26    # 坐标轴标签字号
tick_fontsize = 26      # 坐标轴刻度字号
linewidth = 1.6         # 线条粗细
markersize = 8          # 数据点大小
legend_fontsize = 26  # 图例字号
#------------------------- 

# 创建图形
plt.figure(figsize=(10, 6), dpi=120)  # 设置更高分辨率

# 绘制True曲线 (黑色实线，无标记)
plt.plot(freq_list, HRTF_TRUE, 
         'k-',                   # 黑色实线
         linewidth=1.4,            # 稍粗的线宽
         markersize=0,           # 无标记
         label = "True")

# 绘制PRTFNet曲线 (蓝色虚线，带标记)
plt.plot(freq_list, HRTF_VQVAE, 
         '-*',                  # 蓝色虚线
         color='#D95319', 
         linewidth=linewidth, 
         markersize=markersize,
         markerfacecolor='none', # 空心标记
         markeredgecolor='#D95319',  # 蓝色边框
         markeredgewidth=1.1,
         label = "Ours")

# 绘制Baseline曲线 (红色虚线，带标记)
plt.plot(freq_list, HRTF_VAE, 
         '--',   
         color='#0072BD',# 红色虚线
         linewidth=linewidth, 
         markersize=markersize,
         markerfacecolor='none', # 空心标记
         markeredgecolor='red',  # 红色边框
         markeredgewidth=1.5,
         label = "VAE")

plt.plot(freq_list, HRTF_ResNet, 
         'g--',                  # 绿色虚线
         linewidth=linewidth, 
         markersize=markersize,
         markerfacecolor='none', # 空心标记
         markeredgecolor='green',  # 绿色边框
         markeredgewidth=1.5,
         label = "ResNet")


# 坐标轴设置
plt.xlim(min(freq_list)*0.9, max(freq_list)*1.1)  # 留出10%空白边距
plt.ylim(min(HRTF_TRUE)-1,max(HRTF_TRUE)+2)                                  # 根据你的数据示例设置

# 标签与标题
plt.xlabel('Frequency(kHz)', fontsize=label_fontsize, labelpad=8)  # 改为kHz单位
plt.ylabel('Magnitude (dB)', fontsize=label_fontsize, labelpad=8)

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

# 添加图例
plt.legend(fontsize=legend_fontsize, 
           loc='best',                 # 自动选择最佳位置
           frameon=True,               # 显示图例框
           edgecolor='black',          # 图例框边框颜色
           fancybox=False)             # 不使用圆角边框

# 设置标题为(0,0)
plt.title("(0°, 0°)", fontsize=title_fontsize+4, 
           pad=15,  # 增加标题与图表之间的垂直距离
           y=0.88)  # 略微提升标题位置(y在1.0是默认位置)

# 保存图像
# plt.savefig("LSD_per_frequency.png", bbox_inches='tight', dpi=300)  # 保存高清图
plt.savefig(f"HRTF_contrast_{theta}.pdf", bbox_inches='tight')           # 矢量图格式

# 显示图像
plt.show()