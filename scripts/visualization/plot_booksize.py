import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# 全局配置 - Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 300  # 高分辨率

#-------------------------
# 统一字号配置
title_fontsize = 16    # 标题字号
label_fontsize = 22     # 坐标轴标签字号
tick_fontsize = 20      # 刻度字号
legend_fontsize = 14    # 图例字号
linewidth = 2         # 线条粗细
markersize = 10          # 数据点大小
#-------------------------

# 数据准备
codebook_sizes = [4, 6, 8, 16, 32, 64, 128]
reconstruction_losses = [11.95889703, 11.7882678, 11.53199933, 
                        11.13526226, 10.98240986, 10.9420694, 
                        10.88412229]
reconstruction_losses = np.sqrt(reconstruction_losses)  # 转换为平方根形式

plt.figure(figsize=(9, 5.5), dpi=120)

# 使用蓝色实线+圆形标记(带黑色描边)
plt.plot(codebook_sizes, reconstruction_losses, 
                'b-o',                    # 蓝色实线+圆形标记
                linewidth=linewidth, 
                markersize=markersize,
                markerfacecolor='royalblue',  # 内部填充色
                markeredgecolor='black',   # 黑色描边
                markeredgewidth=0.8,
                label="Reconstruction Loss")  # 图例标签

# 设置对数横坐标
plt.xscale('log')
plt.xlim(3, 200)

# 手动设置横坐标刻度和标签（避免科学计数法）
plt.xticks(codebook_sizes, [str(x) for x in codebook_sizes], 
           rotation=0)  # 旋转标签避免重叠

# 设置纵轴范围以突出变化
plt.ylim(min(reconstruction_losses) * 0.995, max(reconstruction_losses) * 1.005)

# 添加标签
plt.xlabel('Codebook Size', fontsize=label_fontsize, labelpad=8)
plt.ylabel('Reconstruction Loss(LSD)', fontsize=label_fontsize, labelpad=8)
# plt.title('Reconstruction Loss vs. Codebook Size', 
#           fontsize=title_fontsize, pad=15)

# 刻度参数设置
plt.tick_params(axis='both', which='major', 
                labelsize=tick_fontsize, 
                direction='in',          # 刻度线朝内
                width=1.2)               # 刻度线粗细

# 移除了网格线（根据用户要求）
# plt.grid(False)  # 显式关闭网格（可选，但非必要）

# # 添加图例
# plt.legend(fontsize=legend_fontsize, 
#            loc='upper right',          # 选择右上角
#            frameon=True,               # 显示图例框
#            edgecolor='black',          # 图例框边框颜色
#            fancybox=False)             # 不使用圆角边框

# 添加数据标签（横坐标和纵坐标都标注）
for i, (x, y) in enumerate(zip(codebook_sizes, reconstruction_losses)):
    # 标注横坐标值（在x轴下方）
    # plt.annotate(f'{x}', 
    #             (x, min(reconstruction_losses) * 0.998),
    #             textcoords="offset points", 
    #             xytext=(0, -20),  # 位置调整到x轴下方
    #             ha='center',
    #             fontsize=tick_fontsize - 2,
    #             color='dimgray')
    
    # 标注纵坐标值（在数据点上方）
    plt.annotate(f'{y:.3f}', 
                (x, y),
                textcoords="offset points", 
                xytext=(28, 3), 
                ha='center',
                fontsize=tick_fontsize - 2)

# 紧凑布局
plt.tight_layout()

# 保存图片（双格式保存）
plt.savefig("reconstruction_loss.png", bbox_inches='tight', dpi=300)
plt.savefig("reconstruction_loss.pdf", bbox_inches='tight')

# 显示图形
plt.show()