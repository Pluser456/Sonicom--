import numpy as np
import matplotlib.pyplot as plt

# 1. 读取P00013_results.txt文件
data = np.loadtxt('P00013_results.txt', skiprows=1)  # 跳过表头行
indices = data[:, 0].astype(int)   # 第一列：索引(整数)
angles = data[:, 1]                # 第二列：角度(浮点数)

print(f"成功读取 {len(indices)} 个数据点")
print("索引示例:", indices[:5])
print("角度示例:", angles[:5])

# 2. 创建或加载hrtf数组 (2562行 × 90列)
# 假设这里使用随机数据，实际使用时替换为您的真实数据
np.random.seed(42)  # 设置随机种子保证结果可复现
hrtf = np.random.rand(2562, 90) * 60 - 40  # 生成在 -40dB 到 20dB 范围内的模拟数据

# 3. 设置绘图参数
freq_min, freq_max = 0, 20          # 频率范围 (kHz)
theta_min, theta_max = -150, 150    # 角度范围 (度)

# 4. 创建图像
plt.figure(figsize=(12, 8))

# 绘制hrtf伪彩色图
im = plt.imshow(hrtf.T,  # 转置数组使角度在y轴
                aspect='auto',
                extent=[freq_min, freq_max, theta_min, theta_max],
                cmap='jet',  # 使用jet色彩映射
                origin='lower',
                vmin=-40, vmax=20)  # 设置颜色范围与图片一致

# 添加颜色条
cbar = plt.colorbar(im, pad=0.02)
cbar.set_label('dB', rotation=0, labelpad=15)

# 5. 添加从txt文件读取的散点数据
# 假设索引对应频率位置，角度直接使用
freq_locations = indices * 20 / 2561  # 将索引映射到频率范围

# plt.scatter(freq_locations, angles, 
#             marker='+', color='white', 
#             s=40, linewidth=1.2,
#             label='Selected points (P00013_results.txt)')

# 设置标题和标签
plt.xlabel('Frequency, kHz', fontsize=12)
plt.ylabel(r'$\theta$, deg', fontsize=12)
plt.title('HRTF Analysis with Selected Points', fontsize=14)
plt.legend(loc='upper right', fontsize=9)

# 添加网格线增强可读性
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 显示并保存图像
plt.tight_layout()
plt.savefig('hrtf_analysis.png', dpi=300)
plt.show()