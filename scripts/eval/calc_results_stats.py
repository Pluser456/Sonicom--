import os
import re
import numpy as np
from collections import defaultdict

def extract_lsd_from_summary(summary_path):
    """
    从 summary.txt 中提取 LSD 值。
    支持多种格式：
    - "Mean LSD (predicted vs true): X.XXXXXX dB" (CNN-VQVAE)
    - "Mean LSD (reconstructed vs original): X.XXXXXX dB" (VQ-VAE reconstruction)
    - "Mean LSD: X.XXXXXX dB" (PRTFNet, VAE-DNN-CVAE)
    """
    with open(summary_path, 'r') as f:
        content = f.read()

    # 尝试多种匹配模式
    patterns = [
        r'Mean LSD \(predicted vs true\): ([\d.]+)',
        r'Mean LSD \(reconstructed vs original\): ([\d.]+)',
        r'Mean LSD: ([\d.]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return float(match.group(1))
    return None

def get_codebook_size(config_path):
    """
    从评估配置中读取 vqvae_config，然后获取 codebook_size。
    仅适用于 CNN-VQVAE 模型。
    """
    with open(config_path, 'r') as f:
        eval_config = f.read()

    # 读取 vqvae_config 路径
    vqvae_config_match = re.search(r'vqvae_config:\s*(.+)', eval_config)
    if not vqvae_config_match:
        return None

    vqvae_config_path = vqvae_config_match.group(1).strip()
    if not os.path.exists(vqvae_config_path):
        return None

    # 从训练配置中读取 codebook_size
    with open(vqvae_config_path, 'r') as f:
        train_config = f.read()
    cs_match = re.search(r'codebook_size:\s*(\d+)', train_config)
    if cs_match:
        return int(cs_match.group(1))
    return None

def get_model_params(config_path):
    """
    从配置中获取模型参数（如 VAE-DNN-CVAE 的 z_ears_size 和 z_hrtf_size）。
    返回格式：{"z_ears": 64, "z_hrtf": 32}
    """
    with open(config_path, 'r') as f:
        config = f.read()

    result = {}
    z_ears_match = re.search(r'z_ears_size:\s*(\d+)', config)
    if z_ears_match:
        result["z_ears"] = int(z_ears_match.group(1))

    z_hrtf_match = re.search(r'z_hrtf_size:\s*(\d+)', config)
    if z_hrtf_match:
        result["z_hrtf"] = int(z_hrtf_match.group(1))

    return result if result else None

def process_folder(folder_path, model_type="cnn_vqvae"):
    """
    处理单个模型的结果文件夹，计算统计信息。

    Args:
        folder_path: 输入文件夹路径（如 lsd_2D_widespread）
        model_type: 模型类型 ("cnn_vqvae", "prtfnet", "vae_dnn_cvae")

    Returns:
        dict: {key: [lsd_values]}，key 根据模型类型不同：
              - cnn_vqvae: codebook_size
              - prtfnet: "all" (单一结果)
              - vae_dnn_cvae: "z_ears_x_z_hrtf" 组合键
    """
    if model_type == "cnn_vqvae":
        data = defaultdict(list)
    elif model_type == "prtfnet":
        data = {"all": []}
    else:  # vae_dnn_cvae
        data = defaultdict(list)

    for entry in sorted(os.listdir(folder_path)):
        if not entry.startswith('res_'):
            continue

        res_path = os.path.join(folder_path, entry)
        if not os.path.isdir(res_path):
            continue

        # 读取 summary.txt 获取 LSD
        summary_path = os.path.join(res_path, 'summary.txt')
        if not os.path.exists(summary_path):
            continue

        lsd = extract_lsd_from_summary(summary_path)
        if lsd is None:
            continue

        config_path = os.path.join(res_path, 'config.yaml')
        if not os.path.exists(config_path):
            continue

        if model_type == "cnn_vqvae":
            # CNN-VQVAE: 按 codebook_size 分组
            codebook_size = get_codebook_size(config_path)
            if codebook_size is None:
                continue
            data[codebook_size].append(lsd)

        elif model_type == "prtfnet":
            # PRTFNet: 所有结果放在一组
            data["all"].append(lsd)

        else:  # vae_dnn_cvae
            # VAE-DNN-CVAE: 按 latent dimension 分组
            params = get_model_params(config_path)
            if params:
                key = f"{params['z_ears']}x{params['z_hrtf']}"
                data[key].append(lsd)

    return data

def save_stats(data, output_file, model_type="cnn_vqvae", extra_info=None):
    """
    保存统计结果到文件。

    Args:
        data: {key: [values]} 字典
        output_file: 输出文件路径
        model_type: 模型类型
        extra_info: dict, 额外信息（如 cnn_type, dataset）
    """
    header_parts = []
    if extra_info:
        for k, v in extra_info.items():
            header_parts.append(f"{k}: {v}")
    header = ", ".join(header_parts)

    with open(output_file, 'w') as f:
        f.write(f"# {header}\n")

        if model_type == "cnn_vqvae":
            f.write("Codebook Size,Mean LSD (dB),Std LSD (dB),Num Folds\n")
            for cs in sorted(data.keys()):
                values = data[cs]
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                f.write(f"{cs},{mean_val:.6f},{std_val:.6f},{len(values)}\n")

        elif model_type == "prtfnet":
            f.write("Model,Mean LSD (dB),Std LSD (dB),Num Folds\n")
            values = data["all"]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
            f.write(f"PRTFNet,{mean_val:.6f},{std_val:.6f},{len(values)}\n")

        else:  # vae_dnn_cvae
            f.write("Latent Dim (z_ears x z_hrtf),Mean LSD (dB),Std LSD (dB),Num Folds\n")
            for key in sorted(data.keys()):
                values = data[key]
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                f.write(f"{key},{mean_val:.6f},{std_val:.6f},{len(values)}\n")

        f.write("\n")

def process_model_results(base_dir, model_type):
    """
    处理单个模型的所有结果。

    Args:
        base_dir: 基础目录（如 results/data/prtfnet）
        model_type: 模型类型 ("cnn_vqvae", "prtfnet", "vae_dnn_cvae")
    """
    if model_type == "cnn_vqvae":
        folders = [
            ("lsd_2D_widespread", "2D", "widespread"),
            ("lsd_2D_sonicom", "2D", "sonicom"),
            ("lsd_3D_widespread", "3D", "widespread"),
            ("lsd_3D_sonicom", "3D", "sonicom"),
        ]
        output_file = os.path.join(base_dir, "cnn_vqvae_lsd_stats.txt")
    else:
        folders = [
            ("lsd_widespread", None, "widespread"),
            ("lsd_sonicom", None, "sonicom"),
        ]
        output_file = os.path.join(base_dir, f"{model_type}_lsd_stats.txt")

    # 收集所有数据
    all_data = {}
    for folder_name, cnn_type, dataset in folders:
        folder_path = os.path.join(base_dir, folder_name)

        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found!")
            continue

        print(f"Processing {model_type}/{folder_name}...")
        data = process_folder(folder_path, model_type=model_type)

        # 打印摘要
        if model_type == "cnn_vqvae":
            print(f"Found codebook sizes: {sorted(data.keys())}")
            for cs in sorted(data.keys()):
                values = data[cs]
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                print(f"  Codebook size {cs}: {len(values)} folds, mean={mean_val:.4f}, std={std_val:.4f}")
        elif model_type == "prtfnet":
            values = data["all"]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
            print(f"  {len(values)} folds, mean={mean_val:.4f}, std={std_val:.4f}")
        else:  # vae_dnn_cvae
            print(f"Found latent dims: {sorted(data.keys())}")
            for key in sorted(data.keys()):
                values = data[key]
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                print(f"  {key}: {len(values)} folds, mean={mean_val:.4f}, std={std_val:.4f}")
        print()

        # 收集数据用于汇总文件
        key = (cnn_type, dataset) if cnn_type else dataset
        all_data[key] = data

    # 写入汇总文件（包含所有数据集的结果）
    with open(output_file, 'w') as f:
        for key, data in all_data.items():
            if isinstance(key, tuple):
                extra_info = {"cnn_type": key[0], "dataset": key[1]}
            else:
                extra_info = {"dataset": key}

            header_parts = [f"{k}: {v}" for k, v in extra_info.items()]
            header = ", ".join(header_parts)
            f.write(f"# {header}\n")

            if model_type == "cnn_vqvae":
                f.write("Codebook Size,Mean LSD (dB),Std LSD (dB),Num Folds\n")
                for cs in sorted(data.keys()):
                    values = data[cs]
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                    f.write(f"{cs},{mean_val:.6f},{std_val:.6f},{len(values)}\n")
            elif model_type == "prtfnet":
                f.write("Model,Mean LSD (dB),Std LSD (dB),Num Folds\n")
                values = data["all"]
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                f.write(f"PRTFNet,{mean_val:.6f},{std_val:.6f},{len(values)}\n")
            else:  # vae_dnn_cvae
                f.write("Latent Dim (z_ears x z_hrtf),Mean LSD (dB),Std LSD (dB),Num Folds\n")
                for latent_key in sorted(data.keys()):
                    values = data[latent_key]
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                    f.write(f"{latent_key},{mean_val:.6f},{std_val:.6f},{len(values)}\n")
            f.write("\n")

    print(f"Saved statistics to: {output_file}")

def main():
    # 处理三个模型的结果
    models = [
        ("results/data/prtfnet", "prtfnet"),
        ("results/data/vae-dnn-cvae", "vae_dnn_cvae"),
        ("results/data/vqvae", "cnn_vqvae"),
    ]

    for base_dir, model_type in models:
        if os.path.exists(base_dir):
            print(f"=" * 60)
            print(f"Processing {model_type}...")
            print(f"=" * 60)
            process_model_results(base_dir, model_type)
            print()
        else:
            print(f"Warning: {base_dir} not found!")

if __name__ == "__main__":
    main()
