import matplotlib.pyplot as plt
import numpy as np
import os
import re
from collections import defaultdict

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

def extract_accuracy_from_summary(summary_path):
    """从 summary.txt 中提取 Mean Accuracy 值"""
    with open(summary_path, 'r') as f:
        content = f.read()
    match = re.search(r'Mean Accuracy: ([\d.]+)', content)
    if match:
        return float(match.group(1))
    return None

def get_cnn_type_and_codebook_size(config_path):
    """
    从评估配置中读取 CNN 类型和 VQVAE 的码本大小

    Returns:
        tuple: (cnn_type, codebook_size) 例如 ('2D', 16)
    """
    with open(config_path, 'r') as f:
        eval_config = f.read()

    # 读取 CNN 类型
    cnn_type_match = re.search(r'model_type:\s*(\w+)', eval_config)
    if not cnn_type_match:
        return None, None

    model_type = cnn_type_match.group(1)
    cnn_type = '2D' if '2D' in model_type else '3D'

    # 读取 vqvae_config 路径获取 codebook_size
    vqvae_config_match = re.search(r'vqvae_config:\s*(.+)', eval_config)
    if not vqvae_config_match:
        return cnn_type, None

    vqvae_config_path = vqvae_config_match.group(1).strip()
    if not os.path.exists(vqvae_config_path):
        return cnn_type, None

    # 从训练配置中读取 codebook_size
    with open(vqvae_config_path, 'r') as f:
        train_config = f.read()
    cs_match = re.search(r'codebook_size:\s*(\d+)', train_config)
    if cs_match:
        return cnn_type, int(cs_match.group(1))

    return cnn_type, None

def load_accuracy_data(result_dir):
    """
    加载准确度数据并按 CNN 类型和码本大小分组

    Returns:
        dict: {(cnn_type, codebook_size): [accuracy_values]}
    """
    data = defaultdict(list)

    for entry in os.listdir(result_dir):
        if not entry.startswith('res_'):
            continue

        res_path = os.path.join(result_dir, entry)
        if not os.path.isdir(res_path):
            continue

        # 读取 summary.txt 获取准确度
        summary_path = os.path.join(res_path, 'summary.txt')
        if not os.path.exists(summary_path):
            continue

        acc = extract_accuracy_from_summary(summary_path)
        if acc is None:
            continue

        # 获取 CNN 类型和码本大小
        config_path = os.path.join(res_path, 'config.yaml')
        if not os.path.exists(config_path):
            continue

        cnn_type, codebook_size = get_cnn_type_and_codebook_size(config_path)
        if cnn_type is None or codebook_size is None:
            continue

        data[(cnn_type, codebook_size)].append(acc*100)  # 转换为百分比

    return data

def plot_accuracy(data, dataset_name, output_dir="results/figure",
                 default_offset=(0, 15), custom_offsets=None):
    """
    绘制准确度 vs 码本大小曲线图

    Args:
        data: dict, {(cnn_type, codebook_size): [accuracy_values]}
        dataset_name: str, 数据集名称
        output_dir: str, 输出目录
        default_offset: tuple, (x, y) 默认偏移量，单位是 points
        custom_offsets: dict, {(cnn_type, codebook_size): (x, y)} 自定义特定数据点的偏移量
    """
    if custom_offsets is None:
        custom_offsets = {}
    # 计算每个码本大小的均值和标准差（分 2D 和 3D）
    all_cs = set(cs for _, cs in data.keys())
    codebook_sizes = sorted(all_cs)

    # 颜色配置
    colors = {
        '2D': '#D95319',    # 橙色
        '3D': '#C02222'     # 红色
    }

    plt.figure(figsize=(9, 5.5), dpi=120)

    for cnn_type in ['2D', '3D']:
        means = []
        stds = []
        valid_cs = []

        for cs in codebook_sizes:
            key = (cnn_type, cs)
            if key in data and len(data[key]) > 0:
                values = data[key]
                means.append(np.mean(values))
                stds.append(np.std(values, ddof=1) if len(values) > 1 else 0)
                valid_cs.append(cs)

        if len(valid_cs) == 0:
            continue

        # 绘制带误差棒的曲线
        plt.errorbar(valid_cs, means,
                    yerr=stds,
                    fmt=f'{"-o"}',
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

    # 手动设置横坐标刻度
    plt.xticks(codebook_sizes, [str(x) for x in codebook_sizes], rotation=0)

    # 设置纵轴范围以突出变化
    if dataset_name == "widespread":
        plt.ylim(0, 60)
    elif dataset_name == "sonicom":
        plt.ylim(0, 40)

    # 添加标签
    plt.xlabel('Codebook Size', fontsize=label_fontsize, labelpad=8)
    plt.ylabel('Classification Accuracy', fontsize=label_fontsize, labelpad=8)

    # 刻度参数设置
    plt.tick_params(axis='both', which='major',
                    labelsize=tick_fontsize,
                    direction='in',
                    width=1.2)

    # 添加数据标签
    for cnn_type in ['2D', '3D']:
        means = []
        stds = []
        valid_cs = []
        for cs in codebook_sizes:
            key = (cnn_type, cs)
            if key in data and len(data[key]) > 0:
                values = data[key]
                means.append(np.mean(values))
                stds.append(np.std(values, ddof=1) if len(values) > 1 else 0)
                valid_cs.append(cs)

        for x, y, s in zip(valid_cs, means, stds):
            # 使用自定义偏移或默认偏移
            offset = custom_offsets.get((cnn_type, x), default_offset)
            plt.annotate(f'{y:.3f}±{s:.3f}',
                        (x, y),
                        textcoords="offset points",
                        xytext=offset,
                        ha='center',
                        fontsize=tick_fontsize - 4)

    # 图例
    plt.legend(fontsize=legend_fontsize, loc='upper right', frameon=True,
              edgecolor='black', fancybox=False)

    # 紧凑布局
    plt.tight_layout()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"cnn_accuracy_{dataset_name}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")

    plt.show()

def main():
    base_dir = "results/data/vqvae"

    # 处理两个数据集
    datasets = [
        ("cnn_accuracy_widespread", "widespread"),
        ("cnn_accuracy_sonicom", "sonicom")
    ]

    for folder, name in datasets:
        result_dir = os.path.join(base_dir, folder)
        print(f"Processing {name} dataset...")

        if not os.path.exists(result_dir):
            print(f"Warning: {result_dir} not found!")
            continue

        # 加载数据
        data = load_accuracy_data(result_dir)

        # 打印摘要
        codebook_sizes = sorted(set(cs for _, cs in data.keys()))
        print(f"Found codebook sizes: {codebook_sizes}")
        for cnn_type in ['2D', '3D']:
            for cs in codebook_sizes:
                key = (cnn_type, cs)
                if key in data:
                    values = data[key]
                    print(f"  {cnn_type} - Codebook size {cs}: {len(values)} folds, mean={np.mean(values):.4f}, std={np.std(values, ddof=1):.4f}")
        print()

        # 绘图（可以自定义特定数据点的偏移，避免重叠）
        # custom_offsets 格式：{(cnn_type, codebook_size): (x_offset, y_offset)}
        if name == "widespread":
            custom_offsets = {
                ('2D', 4): (5, -25),   # 向下移动
                ('2D', 6): (-20, -20),   # 向下移动
                ('2D', 8): (0, -25),   # 向下移动
                ('2D', 16): (0, -25),   # 向下移动
                ('2D', 32): (0, -25),   # 向下移动
                ('2D', 64): (0, -25),   # 向下移动
                ('2D', 128): (-10, -15),   # 向下移动
                ('3D', 6): (38, 15),   # 向下移动
                ('3D', 8): (20, 15),   # 向下移动
                ('3D', 16): (20, 15),   # 向下移动
            }
        elif name == "sonicom":
            custom_offsets = {
                # 例如：如果 2D CNN 在码本大小 16 的标注和误差棒重叠，可以这样调整:
                ('2D', 4): (5, 15),   # 向下移动
                ('2D', 6): (30, 20),   # 向上移动
                ('2D', 8): (-5, -30),   # 向上移动
                ('2D', 16): (-45, -15),   # 向上移动
                ('2D', 32): (-45, -15),   # 向上移动
                ('2D', 64): (-45, -10),   # 向上移动
                ('2D', 128): (-45, -10),   # 向上移动
                ('3D', 4): (5, -35),  # 向左上方移动
                ('3D', 6): (-50, -15),   # 向下移动
                ('3D', 8): (45, 10),   # 向下移动
                ('3D', 16): (45, 10),   # 向下移动
                # ('3D', 16): (20, 15),   # 向下移动
            }
        plot_accuracy(data, name, custom_offsets=custom_offsets)

if __name__ == "__main__":
    main()
