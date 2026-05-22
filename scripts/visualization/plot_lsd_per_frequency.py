import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# 字体与字号全局设置
# -------------------------
plt.rcParams['font.family'] = 'Times New Roman'
title_fontsize = 22    # 标题字号
label_fontsize = 20    # 坐标轴标签字号
tick_fontsize = 16     # 刻度字号
legend_fontsize = 14   # 图例字号
linewidth = 1.2          # 线条粗细
# -------------------------

def load_freq_data(folder_path):
    """
    加载某个模型在某个数据集上的所有 fold 的频率 LSD 数据

    Returns:
        dict: {'frequencies': array, 'lsd_per_fold': [array, ...]}
    """
    data = []
    freq_data = None

    for entry in sorted(os.listdir(folder_path)):
        if not entry.startswith('res_'):
            continue

        res_path = os.path.join(folder_path, entry)
        if not os.path.isdir(res_path):
            continue

        # 读取 lsd_per_frequency.txt
        freq_file = os.path.join(res_path, 'lsd_per_frequency.txt')
        if not os.path.exists(freq_file):
            continue

        with open(freq_file, 'r') as f:
            lines = f.readlines()

        # 跳过注释行，读取数据
        lsd_values = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    lsd_values.append(float(line))
                except ValueError:
                    continue

        data.append(np.array(lsd_values))

        freq_values_file = os.path.join(res_path, 'freq_data.txt')
        if os.path.exists(freq_values_file) and freq_data is None:
            with open(freq_values_file, 'r') as f:
                freq_lines = f.readlines()
            freq_values = []
            for line in freq_lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        freq_values.append(float(line))
                    except ValueError:
                        continue

            freq_data = np.array(freq_values)

    if len(data) == 0:
        return None

    # 所有 fold 的频率数应该相同，取第一个作为频率数组
    return {'frequencies': freq_data, 'lsd_per_fold': data}

def process_all_models(base_dir, dataset, codebook_sizes=None):
    """
    处理所有模型的数据

    Args:
        base_dir: results/data/路径
        dataset: widespread 或 sonicom
        codebook_sizes: CNN-VQVAE 的码本大小（None 表示使用默认值 16）

    Returns:
        dict: 包含各模型的数据
    """
    result = {}

    # PRTFNet
    prtfnet_path = os.path.join(base_dir, 'prtfnet', f'lsd_{dataset}')
    if os.path.exists(prtfnet_path):
        data = load_freq_data(prtfnet_path)
        if data:
            result['PRTFNet'] = data

    # VAE-DNN-CVAE
    vae_dnn_cvae_path = os.path.join(base_dir, 'vae-dnn-cvae', f'lsd_{dataset}')
    if os.path.exists(vae_dnn_cvae_path):
        data = load_freq_data(vae_dnn_cvae_path)
        if data:
            result['VAE-DNN-CVAE'] = data

    # CNN-VQVAE (2D and 3D)
    for cnn_type in ['2D', '3D']:
        cnn_vqvae_path = os.path.join(base_dir, 'vqvae', f'lsd_{cnn_type}_{dataset}')
        codebook_size = codebook_sizes.get(cnn_type)
        if not os.path.exists(cnn_vqvae_path):
            continue

        # 过滤出指定码本大小的 fold
        lsd_per_fold = []
        freq_data = None
        for entry in sorted(os.listdir(cnn_vqvae_path)):
            if not entry.startswith('res_'):
                continue

            res_path = os.path.join(cnn_vqvae_path, entry)
            config_path = os.path.join(res_path, 'config.yaml')

            # 读取 config 获取 codebook_size
            with open(config_path, 'r') as f:
                config_content = f.read()

            # 从 vqvae_config 中读取实际的 codebook_size
            import re
            vqvae_config_match = re.search(r'vqvae_config:\s*(.+)', config_content)
            if not vqvae_config_match:
                continue

            vqvae_config_path = vqvae_config_match.group(1).strip()
            if not os.path.exists(vqvae_config_path):
                continue

            with open(vqvae_config_path, 'r') as f:
                train_config = f.read()

            cs_match = re.search(r'codebook_size:\s*(\d+)', train_config)
            if not cs_match:
                continue

            actual_cs = int(cs_match.group(1))
            if codebook_size is not None and actual_cs != codebook_size:
                continue

            # 读取 lsd_per_frequency.txt
            freq_file = os.path.join(res_path, 'lsd_per_frequency.txt')
            if not os.path.exists(freq_file):
                continue

            with open(freq_file, 'r') as f:
                lines = f.readlines()

            lsd_values = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        lsd_values.append(float(line))
                    except ValueError:
                        continue

            lsd_per_fold.append(np.array(lsd_values))

            freq_values_file = os.path.join(res_path, 'freq_data.txt')
            if os.path.exists(freq_values_file) and freq_data is None:
                with open(freq_values_file, 'r') as f:
                    freq_lines = f.readlines()
                freq_values = []
                for line in freq_lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            freq_values.append(float(line))
                        except ValueError:
                            continue

                freq_data = np.array(freq_values)

        if len(lsd_per_fold) > 0:
            model_name = f'{cnn_type}-CNN'
            result[model_name] = {
                'frequencies': freq_data,
                'lsd_per_fold': lsd_per_fold
            }

    return result

def plot_lsd_comparison(models_data, dataset, output_dir="results/figure"):
    """
    绘制 LSD vs Frequency 对比图

    Args:
        models_data: dict, 模型数据
        dataset: str, 数据集名称
        output_dir: str, 输出目录
    """
    # 颜色配置
    colors = {
        'PRTFNet': "#1E9E15",           # 绿色
        'VAE-DNN-CVAE': '#0072BD',      # 蓝色
        '2D': '#D95319',       # 橙色
        '3D': '#C02222',       # 红色
    }

    linestyles = {
        'PRTFNet': '--',                # 虚线
        'VAE-DNN-CVAE': '--',           # 点划线
        '2D': '-',             # 实线
        '3D': '-',             # 实线
    }

    plt.figure(figsize=(10, 6))

    for model_name, data in models_data.items():
        lsd_per_fold = data['lsd_per_fold']
        freqs = data['frequencies']
        freqs = freqs / 1000  # 转换为 kHz

        # 计算均值和标准差
        means = np.mean(lsd_per_fold, axis=0)
        stds = np.std(lsd_per_fold, axis=0, ddof=1)

        # 获取颜色和线型
        base_name = model_name.split('-')[0] if 'CNN' in model_name else model_name
        color = colors.get(base_name, 'black')
        linestyle = linestyles.get(model_name, '-')

        # 绘制均值曲线
        plt.plot(freqs, means,
                linestyle=linestyle,
                color=color,
                linewidth=linewidth,
                label=model_name)

        # 绘制误差带（±1 std）
        plt.fill_between(freqs,
                        means - stds,
                        means + stds,
                        alpha=0.2,
                        color=color)

    # 设置坐标轴
    plt.xlabel('Frequency (kHz)', fontsize=label_fontsize, labelpad=10)
    plt.ylabel('LSD (dB)', fontsize=label_fontsize, labelpad=10)

    # 设置范围（根据实际数据调整）
    if dataset == 'widespread':
        plt.xlim(0, 18)  # 0-20 kHz
        plt.ylim(0, 6)
    elif dataset == 'sonicom':
        plt.xlim(0, 20)  # 0-20 kHz
        plt.ylim(0, 10)
    # 网格
    plt.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.8)

    # 刻度
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize,
                   direction='in', width=1.2)

    # 图例
    plt.legend(fontsize=legend_fontsize, loc='upper left', frameon=True,
              edgecolor='black', fancybox=False, ncol=2)

    # 紧凑布局
    plt.tight_layout()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'lsd_per_frequency_{dataset}.pdf')
    # output_file_png = os.path.join(output_dir, f'lsd_per_frequency_{dataset}.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    # plt.savefig(output_file_png, bbox_inches='tight', dpi=120)
    print(f"Saved: {output_file}")

    plt.show()

def main():
    base_dir = "results/data"

    # 配置：CNN-VQVAE 的码本大小
    CODEBOOK_SIZE_WI = {"3D": 32, "2D": 16}  # 修改这里选择码本大小：4, 6, 8, 16, 32, 64, 128
    CODEBOOK_SIZE_SO = {"3D": 32, "2D": 32}  # 修改这里选择码本大小：4, 6, 8, 16, 32, 64, 128
    # 处理两个数据集
    for dataset in ['widespread', 'sonicom']:
        print(f"Processing {dataset} dataset...")

        models_data = process_all_models(base_dir, dataset, codebook_sizes=CODEBOOK_SIZE_WI if dataset == 'widespread' else CODEBOOK_SIZE_SO)

        if len(models_data) < 2:
            print(f"Warning: Not enough data found for {dataset}")
            continue

        print(f"Found models: {list(models_data.keys())}")
        plot_lsd_comparison(models_data, dataset)
        print()

if __name__ == "__main__":
    main()
