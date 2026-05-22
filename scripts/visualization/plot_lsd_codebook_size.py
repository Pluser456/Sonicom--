import matplotlib.pyplot as plt
import os
import re

# 全局配置 - Times New Roman 字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 300

#-------------------------
# 统一字号配置
title_fontsize = 16    # 标题字号
label_fontsize = 22     # 坐标轴标签字号
tick_fontsize = 20      # 刻度字号
legend_fontsize = 14    # 图例字号
linewidth = 2         # 线条粗细
markersize = 4          # 数据点大小
#-------------------------

def parse_lsd_stats(stats_path):
    """
    解析 cnn_vqvae_lsd_stats.txt

    Returns:
        dict: {(cnn_type, dataset): {(codebook_size): (mean, std)}}
    """
    data = {}
    current_cnn = None
    current_dataset = None
    current_data = None

    with open(stats_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 解析标题行: # cnn_type: 2D, dataset: widespread
            if line.startswith('# cnn_type:'):
                match = re.match(r'# cnn_type:\s*(\w+),\s*dataset:\s*(\w+)', line)
                if match:
                    current_cnn = match.group(1)
                    current_dataset = match.group(2)
                    current_data = {}
                    data[(current_cnn, current_dataset)] = current_data
                continue

            # 跳过列头
            if line.startswith('Codebook Size'):
                continue

            # 解析数据行
            parts = line.split(',')
            if len(parts) == 4 and current_data is not None:
                cs = int(parts[0])
                mean = float(parts[1])
                std = float(parts[2])
                current_data[cs] = (mean, std)

    return data


def plot_lsd(data, dataset_name, output_dir="results/figure",
             default_offset=(0, 10), custom_offsets=None):
    """
    绘制 LSD vs 码本大小曲线图

    Args:
        data: dict, {(cnn_type, dataset): {codebook_size: (mean, std)}}
        dataset_name: str, 数据集名称
        output_dir: str, 输出目录
        default_offset: tuple, (x, y) 默认偏移量
        custom_offsets: dict, {(cnn_type, codebook_size): (x, y)} 自定义偏移
    """
    if custom_offsets is None:
        custom_offsets = {}

    codebook_sizes = sorted(
        set(cs for (_, dataset), d in data.items()
            for cs in d if dataset == dataset_name)
    )

    colors = {
        '2D': '#D95319',    # 橙色
        '3D': '#C02222'     # 红色
    }

    plt.figure(figsize=(9, 5.5), dpi=120)

    for cnn_type in ['2D', '3D']:
        key = (cnn_type, dataset_name)
        if key not in data:
            continue

        d = data[key]
        cs_sorted = sorted(d.keys())
        means = [d[cs][0] for cs in cs_sorted]
        stds = [d[cs][1] for cs in cs_sorted]

        plt.errorbar(cs_sorted, means,
                     yerr=stds,
                     fmt='-o',
                     capsize=5,
                     capthick=linewidth,
                     color=colors[cnn_type],
                     ecolor='gray',
                     elinewidth=1.5,
                     linewidth=linewidth,
                     markersize=markersize,
                     markerfacecolor=colors[cnn_type],
                     markeredgecolor='black',
                     markeredgewidth=0.8,
                     label=f"{cnn_type}-CNN")

    # 设置对数横坐标
    plt.xscale('log')
    plt.xlim(3, 200)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)

    plt.xticks(codebook_sizes, [str(x) for x in codebook_sizes], rotation=0)

    # 纵轴范围
    if dataset_name == "widespread":
        plt.ylim(3.0, 4.0)
    elif dataset_name == "sonicom":
        plt.ylim(5.0, 5.8)

    plt.xlabel('Codebook Size', fontsize=label_fontsize, labelpad=8)
    plt.ylabel('LSD (dB)', fontsize=label_fontsize, labelpad=8)

    plt.tick_params(axis='both', which='major',
                    labelsize=tick_fontsize,
                    direction='in',
                    width=1.2)

    # 添加数据标签
    for cnn_type in ['2D', '3D']:
        key = (cnn_type, dataset_name)
        if key not in data:
            continue

        d = data[key]
        for cs in sorted(d.keys()):
            mean, std = d[cs]
            offset = custom_offsets.get((cnn_type, cs), default_offset)
            plt.annotate(f'{mean:.3f}±{std:.3f}',
                         (cs, mean),
                         textcoords="offset points",
                         xytext=offset,
                         ha='center',
                         fontsize=tick_fontsize - 4)

    plt.legend(fontsize=legend_fontsize, loc='upper right', frameon=True,
               edgecolor='black', fancybox=False)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"lsd_codebook_size_{dataset_name}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")

 
def main():
    stats_path = "results/data/vqvae/cnn_vqvae_lsd_stats.txt"

    if not os.path.exists(stats_path):
        print(f"Error: {stats_path} not found!")
        return

    data = parse_lsd_stats(stats_path)

    # 打印摘要
    for (cnn_type, dataset), d in sorted(data.items()):
        print(f"{cnn_type}-CNN, {dataset}:")
        for cs in sorted(d.keys()):
            mean, std = d[cs]
            print(f"  CS={cs}: LSD={mean:.4f} +/- {std:.4f}")
        print()

    datasets = ["widespread", "sonicom"]

    for name in datasets:
        print(f"Plotting {name}...")
        if name == "widespread":
            custom_offsets = {
                ('2D', 4): (5, 35),
                ('2D', 6): (0, 25),
                ('2D', 8): (5, 10),
                ('2D', 16): (0, 10),
                ('2D', 32): (0, 10),
                ('2D', 64): (0, 10),
                ('2D', 128): (-10, 10),
                ('3D', 4): (5, -40),
                ('3D', 6): (0, -40),
                ('3D', 8): (5, -22),
                ('3D', 16): (5, -30),
                ('3D', 32): (10, 11),
                ('3D', 64): (10, 10),
            }
        elif name == "sonicom":
            custom_offsets = {
                ('2D', 4): (5, 95),
                ('2D', 6): (-1, -56),
                ('2D', 8): (5, 30),
                ('2D', 16): (0, 30),
                ('2D', 32): (0, 35),
                ('2D', 64): (0, 30),
                ('2D', 128): (0, 30),
                ('3D', 4): (5, -95),
                ('3D', 6): (-1, 40),
                ('3D', 8): (5, -45),
                ('3D', 16): (0, -49),
                ('3D', 32): (0, -40),
                ('3D', 64): (0, -49),
                ('3D', 128): (0, -30),
            }
        plot_lsd(data, name, custom_offsets=custom_offsets)


if __name__ == "__main__":
    main()
