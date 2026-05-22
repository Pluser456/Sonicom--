import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# 字体与字号全局设置
# -------------------------
plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体设置
title_fontsize = 22    # 子图标题字号
label_fontsize = 26    # 共享坐标轴标签字号
tick_fontsize = 22     # 坐标轴刻度字号
linewidth = 1.6        # 线条粗细
legend_fontsize = 14   # 图例字号
# ------------------------- 

# 方向映射：数据集 -> [(theta, phi), label]
DIRECTIONS = {
    'widespread': [
        (0, 0, r'(0°, 0°)'),
        (0, 90, r'(0°, 90°)'),
        (20, 42, r'(20°, 42°)'),
        (90, 0, r'(90°, 0°)')
    ],
    'sonicom': [
        (0, 0, r'(0°, 0°)'),
        (0, 90, r'(0°, 90°)'),
        (20, 45, r'(20°, 45°)'),
        (90, 0, r'(90°, 0°)')
    ]
}

# 颜色配置
COLORS = {
    'PRTFNet': '#1E9E15',      # 绿色
    'VAE-DNN-CVAE': '#0072BD', # 蓝色
    '2D-CNN': '#D95319',  # 橙色
    '3D-CNN': '#C02222'   # 红色
}

def load_hrtf_data(folder_path, dataset):
    """
    加载某个模型在某个数据集上的 HRTF 数据

    Returns:
        dict: {
            'frequencies': array (kHz),
            'directions': {dir_name: {'true': array, 'pred': array}}
        }
    """
    result = {}
    freq_data = None

    # 找到第一个 res_XXX 文件夹
    res_folders = [d for d in os.listdir(folder_path) if d.startswith('res_')]
    if not res_folders:
        return None

    res_path = os.path.join(folder_path, sorted(res_folders)[0])

    # 读取频率数据
    freq_file = os.path.join(res_path, 'freq_data.txt')
    if os.path.exists(freq_file):
        with open(freq_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        if freq_data is None:
                            freq_data = []
                        freq_data.append(float(line))
                    except ValueError:
                        continue
        freq_data = np.array(freq_data) / 1000.0  # 转换为 kHz

    # 获取该数据集的方向列表
    directions = DIRECTIONS[dataset]
    result['frequencies'] = freq_data
    result['directions'] = {}

    for theta, phi, label in directions:
        dir_name = f"{theta}_{phi}"
        true_file = os.path.join(res_path, f'hrtf_true_{dir_name}.txt')
        pred_file = os.path.join(res_path, f'hrtf_pred_{dir_name}.txt')

        data = {}
        if os.path.exists(true_file):
            with open(true_file, 'r') as f:
                values = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            values.append(float(line))
                        except ValueError:
                            continue
                data['true'] = np.array(values)

        if os.path.exists(pred_file):
            with open(pred_file, 'r') as f:
                values = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            values.append(float(line))
                        except ValueError:
                            continue
                data['pred'] = np.array(values)

        if data:  # 只有当有数据时才保存
            result['directions'][label] = data

    return result

def plot_hrtf_comparison(base_dir, dataset, output_dir="results/figure"):
    """
    绘制 HRTF vs Frequency 对比图（4 个子图）
    """
    # 加载所有模型的数据
    models = {
        'PRTFNet': os.path.join(base_dir, 'prtfnet', f'picked_hrtf_{dataset}'),
        'VAE-DNN-CVAE': os.path.join(base_dir, 'vae-dnn-cvae', f'picked_hrtf_{dataset}'),
        '2D-CNN': os.path.join(base_dir, 'vqvae', f'picked_hrtf_2D_{dataset}'),
        '3D-CNN': os.path.join(base_dir, 'vqvae', f'picked_hrtf_3D_{dataset}')
    }

    model_data = {}
    for name, path in models.items():
        if os.path.exists(path):
            data = load_hrtf_data(path, dataset)
            if data and 'directions' in data:
                model_data[name] = data

    if len(model_data) == 0:
        print(f"No data found for {dataset}")
        return

    # 获取所有方向
    directions = list(model_data[list(model_data.keys())[0]]['directions'].keys())
    freqs = model_data[list(model_data.keys())[0]]['frequencies']

    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=120, sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, dir_label in enumerate(directions):
        ax = axes[idx]

        # 只绘制一次 True（从第一个可用模型获取）
        first_model_name = list(model_data.keys())[0]
        if dir_label in model_data[first_model_name]['directions']:
            true_data = model_data[first_model_name]['directions'][dir_label].get('true')
            if true_data is not None:
                ax.plot(freqs, true_data, 'k--', linewidth=linewidth*0.8,
                      label='True')  # 黑色虚线表示真实值

        # 绘制每个模型的预测值
        for model_name, data in model_data.items():
            if dir_label not in data['directions']:
                continue

            dir_data = data['directions'][dir_label]
            color = COLORS[model_name]

            # 只绘制预测值（如果存在）
            if 'pred' in dir_data:
                ax.plot(freqs, dir_data['pred'], '-', color=color,
                      linewidth=linewidth,
                      label=f'{model_name}')

        # 设置标题和标签
        ax.set_title(f'Direction: {dir_label}', fontsize=title_fontsize, pad=10)
        ax.set_xlabel('Frequency (kHz)', fontsize=label_fontsize)
        ax.set_ylabel('Magnitude (dB)', fontsize=label_fontsize)

        # 设置范围
        if dataset == 'widespread':
            ax.set_xlim(0, 18)
            ax.set_ylim(-45, 5)
        else:  # sonicom
            ax.set_xlim(0, 20)
            ax.set_ylim(-45, 5)
            if dir_label == r'(90°, 0°)':
                ax.set_ylim(-65, -5)

        # 网格和刻度
        ax.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize,
                      direction='in', width=1.0)

        # 图例（每个子图都显示）
        ax.legend(fontsize=legend_fontsize, loc='lower left',
                frameon=True, edgecolor='black', fancybox=False,
                ncol=1)

    plt.tight_layout()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'hrtf_per_direction_{dataset}.pdf')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")

    plt.show()

def main():
    base_dir = "results/data"

    for dataset in ['widespread', 'sonicom']:
        print(f"Processing {dataset}...")
        plot_hrtf_comparison(base_dir, dataset)
        print()

if __name__ == "__main__":
    main()
