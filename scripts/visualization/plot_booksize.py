import matplotlib.pyplot as plt
import numpy as np
import os
import re
from collections import defaultdict

# 全局配置 - Times New Roman 字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 300  # 高分辨率

#-------------------------
# 统一字号配置
title_fontsize = 16    # 标题字号
label_fontsize = 22     # 坐标轴标签字号
tick_fontsize = 20      # 刻度字号
legend_fontsize = 14    # 图例字号
linewidth = 2         # 线条粗细
markersize = 4           # 数据点大小
#-------------------------

def extract_lsd_from_summary(summary_path):
    """从 summary.txt 中提取 Mean LSD 值"""
    with open(summary_path, 'r') as f:
        content = f.read()
    # 匹配 "Mean LSD (reconstructed vs original): X.XXXXXX dB"
    match = re.search(r'Mean LSD \(reconstructed vs original\): ([\d.]+)', content)
    if match:
        return float(match.group(1))
    return None

def load_data(result_dir):
    """
    加载数据并返回按 codebook_size 分组的结果

    Returns:
        dict: {codebook_size: [lsd_values]} - 每个码本大小对应 5 折的 LSD 值列表
    """
    data = defaultdict(list)

    # 遍历所有 res_XXX 文件夹
    for entry in os.listdir(result_dir):
        if not entry.startswith('res_'):
            continue

        res_path = os.path.join(result_dir, entry)
        if not os.path.isdir(res_path):
            continue

        # 读取 summary.txt 获取 LSD
        summary_path = os.path.join(res_path, 'summary.txt')
        if not os.path.exists(summary_path):
            continue

        lsd = extract_lsd_from_summary(summary_path)
        if lsd is None:
            continue

        # 从 res_XXX/config.yaml 中读取 vqvae_config 路径，再从中获取 codebook_size
        config_path = os.path.join(res_path, 'config.yaml')
        if not os.path.exists(config_path):
            continue

        with open(config_path, 'r') as f:
            eval_config = f.read()

        # 从评估配置中读取 vqvae_config 路径
        vqvae_config_match = re.search(r'vqvae_config:\s*(.+)', eval_config)
        if not vqvae_config_match:
            continue

        vqvae_config_path = vqvae_config_match.group(1).strip()
        if not os.path.exists(vqvae_config_path):
            continue

        # 从训练配置中读取 codebook_size
        with open(vqvae_config_path, 'r') as f:
            train_config = f.read()
        cs_match = re.search(r'codebook_size:\s*(\d+)', train_config)
        if cs_match:
            codebook_size = int(cs_match.group(1))
            data[codebook_size].append(lsd)

    return data

def plot_reconstruction_loss(data, dataset_name, output_prefix, output_dir=".",
                            default_offset=(28, 16), custom_offsets=None):
    """
    绘制重构损失曲线图

    Args:
        data: dict, {codebook_size: [lsd_values]}
        dataset_name: str, 数据集名称用于标题和文件名
        output_prefix: str, 输出文件前缀
        output_dir: str, 输出目录路径，默认为当前目录
        default_offset: tuple, (x, y) 默认偏移量，单位是 points
        custom_offsets: dict, {codebook_size: (x, y)} 自定义特定码本大小的偏移量
    """
    if custom_offsets is None:
        custom_offsets = {}
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 计算均值和标准差
    codebook_sizes = sorted(data.keys())
    means = []
    stds = []

    for cs in codebook_sizes:
        values = data[cs]
        means.append(np.mean(values))
        stds.append(np.std(values, ddof=1) if len(values) > 1 else 0)

    means = np.array(means)
    stds = np.array(stds)

    # 创建图形
    plt.figure(figsize=(9, 5.5), dpi=120)

    # 绘制带误差棒的曲线
    plt.errorbar(codebook_sizes, means,
                 yerr=stds,
                 color = 'royalblue',                          # 线条颜色
                 fmt='-s',                                  # 实线 + 圆形标记
                 capsize=5,                   # 误差棒顶端横线长度
                 capthick=linewidth,           # 误差棒顶端线宽
                 ecolor='gray',                # 误差棒颜色
                 elinewidth=1.5,               # 误差棒线宽
                 linewidth=linewidth,
                 markersize=markersize,
                 markerfacecolor='royalblue',  # 内部填充色
                 markeredgecolor='black',   # 黑色描边
                 markeredgewidth=0.8,
                 label=f"Mean ± Std ({len(data[codebook_sizes[0]])}-fold CV)")

    # 设置对数横坐标
    plt.xscale('log')
    plt.xlim(3, 200)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)

    # 手动设置横坐标刻度和标签
    plt.xticks(codebook_sizes, [str(x) for x in codebook_sizes],
               rotation=0)

    # 设置纵轴范围以突出变化
    if dataset_name == "widespread":
        plt.ylim(2.5, 3.5)
    elif dataset_name == "sonicom":
        plt.ylim(4.5, 5.5)

    # 添加标签
    plt.xlabel('Codebook Size', fontsize=label_fontsize, labelpad=8)
    plt.ylabel('Reconstruction Loss (LSD, dB)', fontsize=label_fontsize, labelpad=8)

    # 刻度参数设置
    plt.tick_params(axis='both', which='major',
                    labelsize=tick_fontsize,
                    direction='in',          # 刻度线朝内
                    width=1.2)               # 刻度线粗细

    # 添加数据标签（在数据点上方显示均值±标准差）
    for i, (x, y, s) in enumerate(zip(codebook_sizes, means, stds)):
        # 使用自定义偏移或默认偏移
        offset = custom_offsets.get(x, default_offset)
        plt.annotate(f'{y:.3f}±{s:.3f}',
                    (x, y),
                    textcoords="offset points",
                    xytext=offset,
                    ha='center',
                    fontsize=tick_fontsize - 2)

    # 紧凑布局
    plt.tight_layout()

    # 保存图片为 PDF
    pdf_path = os.path.join(output_dir, f"{output_prefix}_{dataset_name}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {pdf_path}")

    # 显示图形
    plt.show()

# 主程序
if __name__ == "__main__":
    base_dir = "results/data/vqvae"

    # 处理两个数据集
    datasets = [
        ("lsd_recon_widespread", "widespread"),
        ("lsd_recon_sonicom", "sonicom")
    ]

    for folder, name in datasets:
        result_dir = os.path.join(base_dir, folder)
        print(f"Processing {name} dataset...")

        if not os.path.exists(result_dir):
            print(f"Warning: {result_dir} not found!")
            continue

        # 加载数据
        data = load_data(result_dir)
        print(f"Found codebook sizes: {sorted(data.keys())}")
        for cs, values in sorted(data.items()):
            print(f"  Codebook size {cs}: {len(values)} folds, mean={np.mean(values):.4f}, std={np.std(values, ddof=1):.4f}")

        # 绘图（可以自定义特定数据点的偏移，避免重叠）
        # custom_offsets 格式：{codebook_size: (x_offset, y_offset)}
        BASE_OFFSET_X = 15
        if name == "widespread":
            custom_offsets = {
                # 例如：如果码本大小为 64 的标注和误差棒重叠，可以这样调整:
                4: (BASE_OFFSET_X, 36),   # 向上移动
                6: (BASE_OFFSET_X, 20),   # 向上移动
                # 32: (-15, 10), # 向左上方移动
            }
        elif name == "sonicom":
            custom_offsets = {
                4: (BASE_OFFSET_X, 65),   # 向上移动
                6: (BASE_OFFSET_X, 25),   # 向上移动
                8: (BASE_OFFSET_X, 25),   # 向上移动
                # 32: (-15, 10), # 向左上方移动
            }
        plot_reconstruction_loss(data, name, "reconstruction_loss",
                                output_dir="results/figure", default_offset=(BASE_OFFSET_X, 16),
                                custom_offsets=custom_offsets)
