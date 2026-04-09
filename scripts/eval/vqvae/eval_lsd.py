"""
LSD 评估脚本 - 计算 CNN-VQVAE 生成 HRTF 的 LSD 指标
用法:
    python scripts/eval/vqvae/eval_lsd.py --config configs/eval/cnn-vqvae-eval.yaml
"""
import argparse
from pathlib import Path
import torch
import numpy as np

from src.utils.config import load_config, save_config
from src.models.utils import (
    load_pretrained_vqvae,
    load_pretrained_cnn,
    prepare_vqvae_dataset,
    create_single_subject_dataloader,
    infer_one_hrtf,
    get_freq_list
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LSD Evaluation')
    parser.add_argument('--config', type=str, default='configs/eval/cnn-vqvae-eval.yaml',
                        help='Path to config file')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file {args.config} not found.")
    config = load_config(args.config)

    # 设置设备
    device = torch.device(config.evaluation.device)

    # 加载模型
    cnnmodel = load_pretrained_cnn(config)
    vqvae = load_pretrained_vqvae(config)

    # 准备数据集
    dataset_paths, train_dataset, log_mean_hrtf_left, log_mean_hrtf_right = prepare_vqvae_dataset(config)

    # 逐样本 LSD 计算
    res_list = []
    pred_list = []
    true_list = []

    usediff = config.dataset.use_diff
    ear_field = "right_voxel" if config.dataset.mode == "right" else "left_voxel"

    for hrtfid in range(1, len(dataset_paths["test_hrtf_list"]) + 1):
        dataloader = create_single_subject_dataloader(
            dataset_paths, hrtfid, config,
            log_mean_hrtf_left, log_mean_hrtf_right
        )
        pred_log_hrtf, true_log_hrtf = infer_one_hrtf(cnnmodel, vqvae, dataloader, usediff, ear_field, device)
        pred_log_hrtf, true_log_hrtf = torch.abs(pred_log_hrtf), torch.abs(true_log_hrtf)

        pred_list.append(pred_log_hrtf)
        true_list.append(true_log_hrtf)

        lsd = torch.sqrt(torch.mean((pred_log_hrtf - true_log_hrtf) ** 2)).item()
        res_list.append(lsd)
        print(f"LSD of HRTF {hrtfid}: {lsd}")

    print(f"Mean LSD: {np.mean(res_list)}")

    pred_tensor = torch.cat(pred_list, dim=0)
    true_tensor = torch.cat(true_list, dim=0)

    # 获取频率列表
    freq_list = get_freq_list(config.dataset.name)

    # 逐频率点的 LSD
    avg_lsd_per_freq = np.zeros(len(freq_list))
    for freq_idx in range(len(freq_list)):
        LSDvec = torch.sqrt(torch.mean((pred_tensor[:, :, freq_idx] - true_tensor[:, :, freq_idx]) ** 2, dim=1))
        avg_lsd_per_freq[freq_idx] = torch.mean(LSDvec).item()

    # ---- 与 mean HRTF 的对比 ----
    print("\n-----------------contrast with mean HRTF-----------------\n")
    res_list_mean = []
    log_mean_hrtf_right = torch.tensor(np.abs(log_mean_hrtf_right), dtype=torch.float32).cpu()
    log_mean_hrtf_right = log_mean_hrtf_right.unsqueeze(0)

    for hrtfid in range(1, len(dataset_paths["test_hrtf_list"]) + 1):
        lsd_of_mean = torch.sqrt(torch.mean((log_mean_hrtf_right - true_tensor[hrtfid - 1, :, :]) ** 2)).item()
        res_list_mean.append(lsd_of_mean)
        print(f"LSD between mean HRTF and HRTF {hrtfid}: {lsd_of_mean}")

    print(f"Mean LSD of mean HRTF: {np.mean(res_list_mean)}")

    avg_lsd_per_freq_of_mean = np.zeros(len(freq_list))
    for freq_idx in range(len(freq_list)):
        LSDvec = torch.sqrt(torch.mean((log_mean_hrtf_right[:, :, freq_idx] - true_tensor[:, :, freq_idx]) ** 2, dim=1))
        avg_lsd_per_freq_of_mean[freq_idx] = torch.mean(LSDvec).item()

    # ---- 创建结果目录 ----
    input_type = '2D' if config.cnn.model_type in ['2DResNet', '2DResNetANP'] else '3D'
    dataset_name = config.dataset.name
    run_name = f"lsd_{input_type}_{dataset_name}"

    result_base = Path(config.paths.result_dir) / run_name
    result_base.mkdir(parents=True, exist_ok=True)

    # 像 create_experiment 一样创建 res_xxx 文件夹
    existing_dirs = [d for d in result_base.iterdir() if d.is_dir() and d.name.startswith('res_')]
    res_numbers = []
    for d in existing_dirs:
        try:
            res_numbers.append(int(d.name.split('_')[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(res_numbers) + 1 if res_numbers else 0
    result_dir = result_base / f"res_{next_num:03d}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置文件副本
    save_config(config, result_dir / "config.yaml")

    # 保存结果
    np.savetxt(result_dir / "lsd_per_sample.txt", np.array(res_list), fmt='%.6f', header='LSD per sample (dB)')
    np.savetxt(result_dir / "lsd_per_sample_mean.txt", np.array(res_list_mean), fmt='%.6f', header='LSD of mean HRTF per sample (dB)')
    np.savetxt(result_dir / "lsd_per_frequency.txt", avg_lsd_per_freq, fmt='%.3f', header='LSD per frequency (dB)')
    np.savetxt(result_dir / "lsd_per_frequency_mean.txt", avg_lsd_per_freq_of_mean, fmt='%.3f', header='LSD per frequency of mean HRTF (dB)')
    np.savetxt(result_dir / "freq_data.txt", freq_list, fmt='%.1f', header='Frequency (Hz)')

    # 保存汇总统计
    with open(result_dir / "summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"Mean LSD (predicted vs true): {np.mean(res_list):.6f} dB\n")
        f.write(f"Mean LSD (mean HRTF vs true): {np.mean(res_list_mean):.6f} dB\n")
        f.write(f"Number of samples: {len(res_list)}\n")
        f.write(f"Number of frequency bins: {len(freq_list)}\n")
        f.write(f"Config used: {args.config}\n")

    print(f"\nResults saved to {result_dir}")


if __name__ == "__main__":
    main()
